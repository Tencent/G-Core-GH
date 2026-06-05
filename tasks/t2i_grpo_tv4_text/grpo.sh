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

source tasks/t2i_grpo_tv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_bagel_t2i_grpo.py --config-path="${CUR_DIR}/tasks/t2i_grpo_tv4_text/" --config-name="oteam4_4_rl.yaml"




