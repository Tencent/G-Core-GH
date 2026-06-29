
DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../..
CUR_DIR=$(pwd)
pkill -9 -f python
readonly MCORE_PATH1="${CUR_DIR}/../Megatron-LM"
readonly mbridge_path1="${CUR_DIR}/../mbridge"
export PYTHONPATH="${CUR_DIR}:$MCORE_PATH1:${mbridge_path1}"
export WANDB_BASE_URL=
export WANDB_API_KEY=
export _CHECK_PEFT=0
export SGLANG_DISABLE_CUDNN_CHECK=1
export WXMS_MANAGER_API_KEY=g_core
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh
python3 -u tasks/agentic_rl/test_agentic.py \
    --config-path="." --config-name="mini_program.yaml"
