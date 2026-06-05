MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

source tasks/off_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_off_policy_distill.py \
    --config-path="../../tasks/off_policy_distill/yaml" --config-name="offpd_test_multi_data.yaml"
