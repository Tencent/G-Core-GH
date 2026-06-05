
DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../..
CUR_DIR=$(pwd)
pkill -9 -f python
MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

export WANDB_BASE_URL=
export WANDB_API_KEY=

export _CHECK_PEFT=0
export SGLANG_DISABLE_CUDNN_CHECK=1
readonly LOG_DIR=../log/agentic_rl/sokoban_test
# 检查log目录是否存在，不存在则创建
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh
python3 -u tasks/agentic_rl/test_agentic.py \
    --config-path="." --config-name="sokoban_rl_config.yaml" >$LOG_DIR/all_rank.log 2>&1 &
