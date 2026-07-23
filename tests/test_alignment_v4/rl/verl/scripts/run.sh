#!/usr/bin/env bash
# verl RL alignment launcher (GRPO | Qwen3-0.6B | gsm8k | SGLang | Megatron).
# Run from the repo root:
#   bash tests/test_alignment_v4/rl/verl/scripts/run.sh [hydra overrides]
#
# The default groups compose step1 (single_gpu + on_policy + grpo); launch it with a
# single GPU (the default NGPUS_PER_NODE=1). Pick a different point in the matrix:
#   NGPUS_PER_NODE=8 bash .../run.sh parallelism=hybrid_parallelism staleness=off_policy
# Multi-node (run ONLY on the head machine; bootstraps Ray across $HOSTFILE).
# nnodes is no longer auto-passed, so set trainer.nnodes=N explicitly:
#   LAUNCHER=mpirun NGPUS_PER_NODE=8 bash .../run.sh parallelism=hybrid_parallelism staleness=off_policy trainer.nnodes=2
#
# Regenerate the verl parquets first (once) with:
#   python3 tests/test_alignment_v4/common/verl/data/prepare_gsm8k_data.py
set -xeuo pipefail
ulimit -n 32768

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../config" && pwd)"
COMMON_VERL_DIR="$(cd "$SCRIPT_DIR/../../../common/verl" && pwd)"

source "$COMMON_VERL_DIR/env.sh"

# Data/model/reward paths and trainer.nnodes/n_gpus_per_node now live in the yaml
# (repo-root-relative; resolved via hydra.job.chdir=False below). Only the wandb
# names and the ray-bootstrap GPU-per-node count stay here.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
PROJECT_NAME=${PROJECT_NAME:-alignment_qwen3_0_6b}
EXP_NAME=${EXP_NAME:-verl_grpo_qwen3_0_6b}

# multi-node ray bootstrap (sets NNODES + RAY_ADDRESS when LAUNCHER=mpirun)
verl_ray_bootstrap "$NGPUS_PER_NODE"

INFRA=(
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXP_NAME"
)

CONFIG_NAME="${CONFIG_NAME:-align}"

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    hydra.searchpath="[pkg://verl.trainer.config]" \
    hydra.job.chdir=False \
    "${INFRA[@]}" \
    "$@"
