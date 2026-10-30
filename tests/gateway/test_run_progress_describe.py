"""Terminal progress previews: sleep-aware mechanical lines + the LLM describe layer.

The mechanical line (``agent/display.py``) must spend the preview budget on the work,
not on a leading ``sleep``; the describe layer (``gateway/run_turn_progress_describe.py``)
gates, caches and fails soft; the bubble lane rewrites the captioned line in place.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from agent.display import terminal_preview_line, terminal_wait_badge
from gateway import run_turn_progress_describe as describe_mod
from gateway.run_turn_progress import GatewayTurnProgressMixin


class TestTerminalPreviewLine:
    def test_sleep_chain_shows_work_not_wait(self):
        cmd = 'sleep 400; grep -c "done: cards" /tmp/lsh-fanout-run.log'
        out = terminal_preview_line(cmd, cap=40)
        assert out.startswith("⏳ ~7m → ")
        assert "grep -c" in out  # the budget went to the work
        assert "sleep" not in out.split("→", 1)[1]

    def test_pure_sleep_is_a_wait_line(self):
        assert terminal_preview_line("sleep 240", cap=40) == "⏳ waiting ~4m"

    def test_short_plain_command_passthrough(self):
        assert terminal_preview_line("gh pr view 278", cap=40) == "gh pr view 278"

    def test_long_plain_command_head_truncated(self):
        out = terminal_preview_line("git log --oneline " + "x" * 60, cap=40)
        assert len(out) == 40 and out.endswith("...")

    def test_multiline_first_line_plus_count(self):
        out = terminal_preview_line("echo start\npython build.py\necho done", cap=40)
        assert out == "echo start ...+2 lines"

    def test_chain_previews_last_non_sleep_segment(self):
        out = terminal_preview_line("sleep 30 && npm test && npm run build", cap=60)
        assert "npm run build" in out

    def test_quoted_sql_semicolon_is_not_a_chain_split(self):
        cmd = ('echo "=== ids ==="; sqlite3 "file:db" '
               '"SELECT id FROM tasks ORDER BY id;" 2>&1 | head -55')
        out = terminal_preview_line(cmd, cap=64)
        # The in-SQL semicolon must not orphan the tail: the preview leads with the
        # command, not with the post-quote redirect (`" 2>&1 | head -55`).
        assert out.startswith("sqlite3") and len(out) <= 64

    def test_quoted_sql_concat_operator_is_not_a_chain_split(self):
        cmd = 'psql -c "SELECT a || b FROM t;"'
        assert terminal_preview_line(cmd, cap=64) == cmd

    def test_wait_badge_survives_quoted_semicolons(self):
        assert terminal_wait_badge('sleep 30; sqlite3 "db" "SELECT 1;"') == "⏳ 30s → "

    def test_wait_badge_sums_all_sleeps(self):
        assert terminal_wait_badge("sleep 60 && run.sh && sleep 60") == "⏳ ~2m → "

    def test_wait_badge_ignores_short_waits(self):
        assert terminal_wait_badge("run.sh") == ""


class TestDescribeLayer:
    def test_should_describe_gates(self):
        assert describe_mod.should_describe("sleep 400; grep -c x /tmp/fanout.log")
        assert describe_mod.should_describe("echo hi\npython build.py")
        assert not describe_mod.should_describe("gh pr view 278")
        assert not describe_mod.should_describe("")
        assert not describe_mod.should_describe("short")

    def test_cache_key_collapses_counters_and_durations(self):
        a = describe_mod.normalize_for_cache("sleep 400; grep -c 3 /tmp/run.1.log")
        b = describe_mod.normalize_for_cache("sleep 900; grep -c 17 /tmp/run.9.log")
        assert a == b

    @pytest.mark.asyncio
    async def test_describe_caches_and_reuses(self, monkeypatch):
        calls = []

        def fake_oneshot(**kw):
            calls.append(kw)
            return "checking card completion"

        monkeypatch.setattr("agent.oneshot.run_oneshot", fake_oneshot)
        first = await describe_mod.describe_command("sleep 400; grep -c x /tmp/f.log")
        second = await describe_mod.describe_command("sleep 400; grep -c x /tmp/f.log")
        assert first == "checking card completion"
        assert second == first
        assert len(calls) == 1  # second hit came from the cache

    @pytest.mark.asyncio
    async def test_describe_failure_returns_none(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("no provider configured")

        monkeypatch.setattr("agent.oneshot.run_oneshot", boom)
        assert await describe_mod.describe_command("sleep 400; x") is None

    @pytest.mark.asyncio
    async def test_describe_timeout_returns_none(self, monkeypatch):
        import time as _time

        monkeypatch.setattr(describe_mod, "_DESCRIBE_TIMEOUT_S", 0.05)

        def slow(**kw):
            _time.sleep(0.5)
            return "never"

        monkeypatch.setattr("agent.oneshot.run_oneshot", slow)
        assert await describe_mod.describe_command("sleep 400; x") is None

    def test_described_line_is_an_annotation(self):
        # The command preview stays the ground truth; the caption rides below as prose.
        assert describe_mod.caption_line("counting finished cards") == "💡 counting finished cards"


class TestBubbleLaneIntegration:
    def test_absorb_unwraps_describe_tuple(self):
        st = SimpleNamespace(progress_lines=[])
        line = GatewayTurnProgressMixin()._progress_absorb(st, ("__describe__", "mech line", "cmd"))
        assert line == "mech line"
        assert st.progress_lines == ["mech line"]

    def test_absorb_still_handles_dedup(self):
        st = SimpleNamespace(progress_lines=["first"])
        line = GatewayTurnProgressMixin()._progress_absorb(st, ("__dedup__", "first", 1))
        assert "first" in line and "×2" in line

    def test_describe_gate(self):
        runner = GatewayTurnProgressMixin()
        runner._ctx = SimpleNamespace(_progress_describe_enabled=False)
        assert runner._describe_gate("terminal", {"command": "sleep 400; x"}) is None
        runner._ctx = SimpleNamespace(_progress_describe_enabled=True)
        assert runner._describe_gate("web_search", {"query": "x"}) is None
        assert runner._describe_gate("terminal", {"command": "ls"}) is None  # short = own caption
        cmd = "sleep 400; grep -c x /tmp/fanout-run.log"
        assert runner._describe_gate("terminal", {"command": cmd}) == cmd

    @pytest.mark.asyncio
    async def test_rewrite_annotates_without_replacing(self, monkeypatch):
        runner = GatewayTurnProgressMixin()
        runner._ctx = SimpleNamespace(_run_still_current=lambda: True)
        edits = []

        async def fake_edit(st, message_id, text):
            edits.append((message_id, text))
            return SimpleNamespace(success=True)

        monkeypatch.setattr(runner, "_edit_progress_message", fake_edit)
        async def fake_describe(command, **kw):
            return "counting finished cards"

        monkeypatch.setattr("gateway.run_turn_progress_describe.describe_command", fake_describe)
        st = SimpleNamespace(progress_lines=["mech line"], progress_msg_id="42", can_edit=True)
        runner._schedule_describe_rewrite(st, "sleep 400; grep x", "mech line")
        await asyncio.sleep(0.05)  # fire-and-forget task lands
        # Command stays (ground truth), caption lands as its own line below it.
        assert st.progress_lines == ["mech line", "💡 counting finished cards"]
        assert edits and edits[0][0] == "42"

    def test_dedup_marker_lands_on_command_not_caption(self):
        runner = GatewayTurnProgressMixin()
        st = SimpleNamespace(progress_lines=["mech line", "💡 counting finished cards"])
        line = runner._progress_absorb(st, ("__dedup__", "mech line", 1))
        assert line.startswith("mech line") and "×2" in line
        assert st.progress_lines[1] == "💡 counting finished cards"  # caption untouched

    @pytest.mark.asyncio
    async def test_superseded_bubble_still_gets_its_caption(self, monkeypatch):
        # Fast command: the bubble closed (reset by an interim message) before the caption
        # arrived — the frozen message must still be annotated.
        runner = GatewayTurnProgressMixin()
        runner._ctx = SimpleNamespace(_run_still_current=lambda: True)
        edits = []

        async def fake_edit(st, message_id, content):
            edits.append((message_id, content))
            return SimpleNamespace(success=True)

        monkeypatch.setattr(runner, "_edit_progress_message", fake_edit)

        async def fake_describe(command, **kw):
            await asyncio.sleep(0.02)
            return "checking gateway start time"

        monkeypatch.setattr("gateway.run_turn_progress_describe.describe_command", fake_describe)
        dead_lines = ["mech line", "later command line"]
        st = SimpleNamespace(progress_lines=[], progress_msg_id=None, can_edit=True,
                             superseded_bubbles=[("msg-dead-1", dead_lines)])
        runner._schedule_describe_rewrite(st, "ps -o pid,lstart", "mech line")
        await asyncio.sleep(0.08)
        assert dead_lines == ["mech line", "💡 checking gateway start time", "later command line"]
        assert edits and edits[0][0] == "msg-dead-1"
        assert "later command line" in edits[0][1]  # clobber-safe: later lines preserved

    @pytest.mark.asyncio
    async def test_rewrite_leaves_missing_line_alone(self, monkeypatch):
        runner = GatewayTurnProgressMixin()
        runner._ctx = SimpleNamespace(_run_still_current=lambda: True)
        edited = []

        async def fake_edit(st, message_id, text):
            edited.append(text)
            return SimpleNamespace(success=True)

        monkeypatch.setattr(runner, "_edit_progress_message", fake_edit)

        async def fake_describe(command, **kw):
            return "counting finished cards"

        monkeypatch.setattr("gateway.run_turn_progress_describe.describe_command", fake_describe)
        st = SimpleNamespace(progress_lines=["a different line"], progress_msg_id="42", can_edit=True)
        runner._schedule_describe_rewrite(st, "sleep 400; grep x", "mech line")
        await asyncio.sleep(0.05)
        assert st.progress_lines == ["a different line"]  # rolled/deduped meanwhile
        assert not edited