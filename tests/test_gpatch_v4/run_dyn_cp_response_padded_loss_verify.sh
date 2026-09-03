#!/bin/bash
# Dyn-CP response-padded sequence loss verification (CPU unit / e2e harness).
#
# Does NOT need Ray or multi-GPU. Covers:
#   - reconstruction helpers
#   - GRPO(+TIS sequence) numerical match: pure [B,S] vs THD→reconstruct→pad
#   - path probe: masked_reduce_thd_expand must not run after cu_seqlens=None
#
# Out of scope this round: wemm_video / welm_v4 (legacy THD+CP loss path).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RCDIR="$(cd "$ROOT/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/tests:$ROOT/tests/test_gpatch_v4:${RCDIR}/Megatron-LM:${RCDIR}/mbridge:${RCDIR}/Megatron-Bridge/src:${PYTHONPATH:-}"

cd "$ROOT"
python -m pytest -v -s \
  tests/test_gpatch_v4/test_dynamic_cp_loss_reconstruction.py \
  tests/test_gpatch_v4/test_dynamic_cp_response_padded_loss.py \
  "$@"
