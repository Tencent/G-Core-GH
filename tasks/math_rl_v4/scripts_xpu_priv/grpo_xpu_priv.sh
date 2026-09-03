LIB_PATH="../"
MCORE_PATH="${LIB_PATH}/Megatron-LM/"
MBRIDGE_PATH="${LIB_PATH}/mbridge"
MEGATRON_BRIDGE_PATH="${LIB_PATH}/Megatron-Bridge"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/xpu_env_priv.sh"


export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts_xpu_priv/xpu_env_priv.sh
source tasks/math_rl_v4/scripts_xpu_priv/mpirun-init-ray-xpu-priv.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config.yaml"
