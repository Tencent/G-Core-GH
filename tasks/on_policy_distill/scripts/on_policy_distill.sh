MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

source tasks/on_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_on_policy_distill.py \
    --config-path="../../tasks/on_policy_distill/yamls" --config-name="distill_demo"
