# Shared gcore env for the alignment / reproducibility runs. Source this from a
# per-algorithm run.sh. Must be run from the repo root (PYTHONPATH uses $PWD and
# the Ray bootstrap is at tasks/math_rl_v4/scripts/mpirun-init-ray.sh).
MCORE_PATH="${MCORE_PATH:-/root/Megatron-LM/}"
MBRIDGE_PATH="${MBRIDGE_PATH:-/work/wepsdl/mbridge}"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/root/Megatron-Bridge}"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:${PYTHONPATH:-}"

source tasks/math_rl_v4/scripts/mpirun-stop-ray.sh
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh
