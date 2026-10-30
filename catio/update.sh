#!/usr/bin/env bash
# Fork-aware update for hosts seated on catio-custom (run as the hermes user).
# NEVER use stock `hermes update` on these hosts: its flow assumes upstream
# checkouts and can move the checkout off catio-custom, dropping the custom
# layer. This script: fetch upstream -> merge onto catio-custom -> rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."
git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/NousResearch/hermes-agent.git
git fetch origin catio-custom
git fetch upstream main
git checkout catio-custom
git merge --no-edit upstream/main
echo "== verification sweep"
FAIL=0
for f in $(git diff --name-only HEAD@{1}..HEAD -- '*.py' 2>/dev/null); do
  python3 -m py_compile "$f" || FAIL=1
done
[ "$FAIL" = 0 ] || { echo "py_compile FAILED — resolve before pushing/continuing"; exit 1; }
echo "== rebuilding env"
bash scripts/install.sh
echo "== done. If this host runs a gateway, restart it now (hermes gateway restart or systemctl --user)."
