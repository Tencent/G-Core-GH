#!/usr/bin/env bash
# gcore-only: legacy loss_factory vs new loss/ package (use_legacy_loss).
# Does not touch verl alignment. Wraps rl/gcore/scripts/run.sh with fixed overrides:
#   off-policy, BSHD (no THD pack), short seq, exit_step=10.
# LOSS maps to reproducibility/config/loss/{grpo,gspo,sapo}.yaml
# (via hydra.searchpath; independent of verl-align loss configs).
#
# Usage (from repo root):
#   LOSS=grpo LEGACY=true  bash tests/test_alignment_v4/reproducibility/scripts/run_legacy_vs_new_loss.sh
#   LOSS=grpo LEGACY=false bash tests/test_alignment_v4/reproducibility/scripts/run_legacy_vs_new_loss.sh
#   LOSS=sapo LEGACY=false PARALLEL=hybrid bash tests/test_alignment_v4/reproducibility/scripts/run_legacy_vs_new_loss.sh
#
# Env:
#   LOSS       grpo|gspo|sapo          (required; selects loss/*.yaml under repro config)
#   LEGACY     true|false              (required; maps to ppo.use_legacy_loss)
#   PARALLEL   single_gpu|hybrid       (default: single_gpu)
#   EXP_NAME / PROJECT_NAME            (optional; defaults derived below)
# Extra CLI args are forwarded to run.sh.
#
# Matrix (single_gpu / hybrid):
#   SH=tests/test_alignment_v4/reproducibility/scripts/run_legacy_vs_new_loss.sh
#   for LOSS in grpo gspo sapo; do
#     for LEGACY in true false; do
#       LOSS=$LOSS LEGACY=$LEGACY bash $SH                          # single_gpu
#       LOSS=$LOSS LEGACY=$LEGACY PARALLEL=hybrid bash $SH          # hybrid
#     done
#   done
#
# Extra knobs via CLI (examples):
#   LOSS=grpo LEGACY=true bash $SH \
#     ppo.grpo_kl_loss_beta=0.01 ppo.ppo_entropy_bonus=0.01 \
#     ppo.ppo_initial_policy_kl_penalty=0.01
#   LOSS=grpo LEGACY=true bash $SH training.train_mbs=2
#   LOSS=grpo LEGACY=true PARALLEL=hybrid bash $SH training.train_mbs=2 \
#     ppo.grpo_kl_loss_beta=0.01 ppo.ppo_entropy_bonus=0.01 \
#     ppo.ppo_initial_policy_kl_penalty=0.01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPRO_CONFIG_DIR="$(cd "$REPRO_ROOT/config" && pwd)"
GCORE_RUN_SH="$(cd "$REPRO_ROOT/../rl/gcore/scripts" && pwd)/run.sh"

LOSS="${LOSS:-}"
LEGACY="${LEGACY:-}"
PARALLEL="${PARALLEL:-single_gpu}"

if [[ -z "$LOSS" || -z "$LEGACY" ]]; then
    echo "ERROR: LOSS and LEGACY are required." >&2
    echo "  LOSS=grpo|gspo|sapo LEGACY=true|false [PARALLEL=single_gpu|hybrid]" >&2
    exit 1
fi

case "$LOSS" in
    grpo|gspo|sapo)
        ;;
    *)
        echo "ERROR: LOSS must be grpo|gspo|sapo, got: $LOSS" >&2
        exit 1
        ;;
esac

case "$LEGACY" in
    true|True|TRUE|1)
        USE_LEGACY=True
        PATH_TAG=legacy
        ;;
    false|False|FALSE|0)
        USE_LEGACY=False
        PATH_TAG=new
        ;;
    *)
        echo "ERROR: LEGACY must be true|false, got: $LEGACY" >&2
        exit 1
        ;;
esac

case "$PARALLEL" in
    single_gpu)
        PARALLELISM=single_gpu
        ROLLOUT=sglang_single_gpu
        PARA_TAG=sg
        ;;
    hybrid)
        PARALLELISM=hybrid_parallelism
        ROLLOUT=sglang
        PARA_TAG=hy
        ;;
    *)
        echo "ERROR: PARALLEL must be single_gpu|hybrid, got: $PARALLEL" >&2
        exit 1
        ;;
esac

export PROJECT_NAME="${PROJECT_NAME:-legacy_vs_new_loss}"
export EXP_NAME="${EXP_NAME:-${LOSS}_${PATH_TAG}_${PARA_TAG}}"

exec bash "$GCORE_RUN_SH" \
    "hydra.searchpath=[file://${REPRO_CONFIG_DIR}]" \
    loss="$LOSS" \
    parallelism="$PARALLELISM" \
    rollout="$ROLLOUT" \
    staleness=off_policy \
    policy.ppo_pack_seq=False \
    training.seq_length=2048 \
    +training.exit_step=10 \
    "sampler.infer_engine_configs.0.generate_max_tokens=1024" \
    "+ppo.use_legacy_loss=${USE_LEGACY}" \
    "$@"
