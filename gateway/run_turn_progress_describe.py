"""LLM describe layer for terminal-tool progress previews.

A terminal preview posts immediately with the mechanical (sleep-badge,
tail-biased) line; this lane then asks a small auxiliary model for a short
human caption and rewrites the line inside the editable progress bubble
(chat.update). Pure waits and boilerplate poll loops read as prose instead of
truncated shell text. Never blocks the agent thread: the describe runs as an
async task on the gateway loop; any failure or slow response leaves the
mechanical line standing. Consecutive identical commands produce identical
captions, so the bubble lane's existing consecutive-dedup collapses poll loops
into one "×N" line. Captions cache by digit-normalized command (counters and
sleep durations collapse), so a 40-iteration watch pays for ~one call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

logger = logging.getLogger(__name__)

_DESCRIBE_CACHE: dict[str, str] = {}
_DESCRIBE_CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()

_DESCRIBE_TIMEOUT_S = 30.0  # cold auto-chain (failed provider attempts + fallback) measured ~14s
_DESCRIBE_INSTRUCTIONS = (
    "You write progress-indicator captions for shell commands. Describe what the command DOES "
    "in at most 12 words, present tense, plain English, as ONE single line. Name the concrete "
    "target (file, service, count) when it is the point of the command; ignore sleeps and "
    "repeated polling. Return ONLY the caption — no quotes, no trailing punctuation, no "
    "markdown, no line breaks."
)


def normalize_for_cache(command: str) -> str:
    """Digit-collapsed cache key: ``sleep 400; grep -c x /tmp/run.1`` == ``sleep 5; grep -c x /tmp/run.9``."""
    return re.sub(r"[0-9]+", "N", re.sub(r"\s+", " ", command or "")).strip().lower()


def should_describe(command: str) -> bool:
    """Only opaque commands need a caption: multi-line, chained, or sleep-prefixed.

    A plain short command is its own caption — no model call, no added latency.
    """
    text = (command or "").strip()
    if "\n" in text:
        return True
    if len(text) < 24:
        return False
    return bool(re.search(r"\s*(?:;|&&|\|\|)\s*", text)) or bool(
        re.match(r"^\s*sleep\b", text, re.IGNORECASE))


def caption_line(caption: str) -> str:
    """The annotation line rendered BELOW its command preview: the command stays the ground
    truth, the caption rides under it as prose."""
    return f"💡 {caption}"


def _cache_get(key: str) -> str | None:
    with _CACHE_LOCK:
        return _DESCRIBE_CACHE.get(key)


def _cache_put(key: str, value: str) -> None:
    with _CACHE_LOCK:
        if len(_DESCRIBE_CACHE) >= _DESCRIBE_CACHE_MAX:
            _DESCRIBE_CACHE.pop(next(iter(_DESCRIBE_CACHE)))
        _DESCRIBE_CACHE[key] = value


async def describe_command(command: str, *, main_runtime: dict | None = None) -> str | None:
    """Cached ≤12-word caption for ``command``; None on any failure or slow response."""
    key = normalize_for_cache(command)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    def _call() -> str | None:
        from agent.oneshot import run_oneshot
        return run_oneshot(
            task="progress_describe",
            instructions=_DESCRIBE_INSTRUCTIONS,
            user_input=command[:2000],
            max_tokens=48,
            temperature=0.2,
            timeout=25.0,
            main_runtime=main_runtime,
        ) or None

    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_call), timeout=_DESCRIBE_TIMEOUT_S)
    except Exception as exc:  # timeout, provider error, no provider configured
        # INFO, not DEBUG: a caption that never lands is invisible otherwise — the mechanical
        # line just stands, and the operator cannot tell the layer is on but failing.
        logger.info("progress caption unavailable (%.60s...): %s", command, exc)
        return None
    caption = re.sub(r"\s+", " ", raw or "").strip().strip('"\''"'"'').strip()
    if not caption or len(caption) > 120:
        # Temporary observability: the provider chain returned something unusable.
        logger.info("progress caption rejected (%.60s...): %r", command, (raw or "")[:100])
        return None
    _cache_put(key, caption)
    return caption