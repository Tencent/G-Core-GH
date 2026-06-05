DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../..
CUR_DIR=$(pwd)
MCORE_PATH="${CUR_DIR}/../../Megatron-LM/"
MBRIDGE_PATH="${CUR_DIR}/../../mbridge"
MEGATRON_BRIDGE_PATH="${CUR_DIR}/../../Megatron-Bridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="${CUR_DIR}/tasks/math_rl_v4_ppo" --config-name="rl_config.yaml"
