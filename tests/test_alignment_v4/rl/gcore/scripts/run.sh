#!/usr/bin/env bash
# gcore RL alignment / reproducibility launcher. Run from the repo root:
#   bash tests/test_alignment_v4/rl/gcore/scripts/run.sh [hydra overrides]
#
# The default groups compose step1 (single_gpu + on_policy + grpo + dense + sglang).
# Pick a different point in the matrix via CLI overrides, e.g.:
#   ... run.sh parallelism=hybrid_parallelism staleness=off_policy
# rollout selects the inference backend (rollout=sglang default; vllm later).
# MoE reproducibility (gcore-only, run twice and diff):
#   CONFIG_NAME=repro_moe bash tests/test_alignment_v4/rl/gcore/scripts/run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../config" && pwd)"
COMMON_DIR="$(cd "$SCRIPT_DIR/../../../common" && pwd)"

source "$COMMON_DIR/gcore/env.sh"

CONFIG_NAME="${CONFIG_NAME:-align}"

# Per-experiment knobs (change per run). Only injected for the align entry; load
# from a fresh (non-existent) ckpt path so gcore builds weights from HF base and
# prev_ppo_step=0 (no auto-resume). repro_moe carries its own checkpoint/report.
EXP_OVERRIDES=()
if [ "$CONFIG_NAME" = align ]; then
    CKPT=${CKPT:-save_step1_qwen3_0_6b_align_fresh}
    PROJECT_NAME=${PROJECT_NAME:-alignment_qwen3_0_6b}
    EXP_NAME=${EXP_NAME:-align_step1_gcore}
    # '+' prefix: these keys are not in base.yaml's checkpoint/report dicts, so
    # hydra (struct mode) needs append semantics to add them.
    EXP_OVERRIDES=(
        +checkpoint.load_ckpt_path=null
        +checkpoint.save_ckpt_path="ckpt/$EXP_NAME"
        +report.wandb_project="$PROJECT_NAME"
        +report.wandb_exp_name="$EXP_NAME"
        +report.log_dir="logs/$EXP_NAME"
    )
fi

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="$CONFIG_DIR" \
    --config-name="$CONFIG_NAME" \
    "${EXP_OVERRIDES[@]}" \
    "$@"

# single gpu: EXP_NAME=gcore_single_gpu bash tests/test_alignment_v4/rl/gcore/scripts/run.sh rollout=sglang_single_gpu parallelism=single_gpu 2>&1 | tee logs/single_gpu_gcore.log
# parallel: EXP_NAME=gcore_parallel_lr0 bash tests/test_alignment_v4/rl/gcore/scripts/run.sh optimizer.lr=0 2>&1 | tee logs/parallel_gcore_lr0.log