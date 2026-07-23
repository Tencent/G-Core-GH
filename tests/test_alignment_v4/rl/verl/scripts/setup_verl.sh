#!/usr/bin/env bash
# One-time setup of the verl comparison checkout for the gcore<->verl alignment:
# clone verl (main), pin it to the commit this alignment is based on, and apply the
# single required (non-dump) patch (agent_loop.py GRPO rollout-seed fix).
#
# Idempotent: re-running detects an already-patched checkout and exits. The clone
# path MUST match what common/verl/env.sh uses as VERL_PATH (default
# /work/wepsdl/projects/verl); override both with the same VERL_PATH if you move it.
#
# Usage:
#   bash tests/test_alignment_v4/rl/verl/scripts/setup_verl.sh
#   VERL_PATH=/some/where bash tests/test_alignment_v4/rl/verl/scripts/setup_verl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_VERL_DIR="$(cd "$SCRIPT_DIR/../../../common/verl" && pwd)"
PATCH="$COMMON_VERL_DIR/patches/agent_loop.patch"

# Must match common/verl/env.sh's VERL_PATH default so the runtime finds this clone.
VERL_PATH="${VERL_PATH:-/work/wepsdl/projects/verl}"
VERL_REPO_URL="${VERL_REPO_URL:-https://github.com/volcengine/verl.git}"
VERL_ALIGN_PATCH_COMMIT="${VERL_ALIGN_PATCH_COMMIT:-ed89419c23653730e95c43954c00e6c24277e1c8}"

[ -f "$PATCH" ] || { echo "[setup_verl] ERROR: patch not found: $PATCH" >&2; exit 1; }

# 1. clone if missing
if [ ! -e "$VERL_PATH/.git" ]; then
    echo "[setup_verl] cloning verl -> $VERL_PATH"
    git clone "$VERL_REPO_URL" "$VERL_PATH"
fi

# 2. already patched? then we're done (idempotent)
if git -C "$VERL_PATH" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[setup_verl] agent_loop.patch already applied at $VERL_PATH ($(git -C "$VERL_PATH" rev-parse --short HEAD)); nothing to do."
    exit 0
fi

# 3. pin to the alignment commit (fetch first in case the clone is shallow/stale)
echo "[setup_verl] checking out $VERL_ALIGN_PATCH_COMMIT"
git -C "$VERL_PATH" fetch --quiet origin || true
git -C "$VERL_PATH" checkout "$VERL_ALIGN_PATCH_COMMIT"

# 4. apply the patch
git -C "$VERL_PATH" apply "$PATCH"
echo "[setup_verl] applied agent_loop.patch to $VERL_PATH ($(git -C "$VERL_PATH" rev-parse --short HEAD))."
