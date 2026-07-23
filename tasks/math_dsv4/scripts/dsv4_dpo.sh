MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MLM_BRIDGE_PATH="/root/Megatron-Bridge/"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MLM_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

export GCORE_NNODES=16
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_dpo.py \
    --config-path="../../tasks/math_dsv4/yaml" \
    --config-name="dsv4_dpo.yaml"
