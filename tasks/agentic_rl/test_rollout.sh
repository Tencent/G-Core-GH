
DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../..
CUR_DIR=$(pwd)
pkill -9 -f python
readonly MCORE_PATH="${CUR_DIR}/../../Megatron-LM"
readonly mbridge_path="${CUR_DIR}/../../mbridge"
export PYTHONPATH="${CUR_DIR}:$MCORE_PATH:${mbridge_path}"
export WANDB_BASE_URL=
export WANDB_API_KEY=
export _CHECK_PEFT=0
export SGLANG_DISABLE_CUDNN_CHECK=1
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh
python3 -u tasks/agentic_rl/test_agentic.py \
    --config-path="." --config-name="rl_config.yaml"
