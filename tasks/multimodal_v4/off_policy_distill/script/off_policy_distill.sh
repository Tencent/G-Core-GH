MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MLM_BRIDGE_PATH="/root/Megatron-Bridge/src"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MLM_BRIDGE_PATH:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

source tasks/multimodal_v4/off_policy_distill/script/mpirun_run_once.sh \
    tasks/multimodal_v4/off_policy_distill/script/run_once.sh

python3 -u gpatch_v4/entry/train_off_policy_distill.py \
    --config-path="../../tasks/multimodal_v4/off_policy_distill/yaml" \
    --config-name="off_policy_distill_demo"
