MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MLM_BRIDGE_PATH="/root/Megatron-Bridge/src"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MLM_BRIDGE_PATH:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

source tasks/multimodal_v4/finetune/scripts/mpirun_run_once.sh \
    tasks/multimodal_v4/finetune/scripts/run_once.sh

# 启动datasetv4的lmdb
source tasks/multimodal_v4/finetune/scripts/mpirun_run_once.sh \
    tasks/multimodal_v4/finetune/scripts/lmdb_run.sh

python3 -u gpatch_v4/entry/verify_dataloader.py \
    --config-path="../../tasks/multimodal_v4/finetune/yaml" \
    --config-name="qwen3vl_multi_turn.yaml"
