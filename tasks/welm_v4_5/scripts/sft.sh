MCORE_PATH="../Megatron-LM/"
MBRIDGE_PATH="../mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

source tasks/welm_v4_5/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/welm_v4_5/yaml" --config-name="sft.yaml"
