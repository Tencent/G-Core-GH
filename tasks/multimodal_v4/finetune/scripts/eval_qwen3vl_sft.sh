MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MLM_BRIDGE_PATH="/root/Megatron-Bridge/src"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MLM_BRIDGE_PATH:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

ray stop --force

DATASET_NAME="filter_4k_qwen3vl"

source tasks/multimodal_v4/finetune/scripts/mpirun_run_once.sh \
    tasks/multimodal_v4/finetune/scripts/run_once.sh

source tasks/multimodal_v4/finetune/scripts/mpirun_run_once.sh \
    tasks/multimodal_v4/finetune/scripts/lmdb_run.sh $DATASET_NAME

python3 -u gpatch_v4/entry/eval_entry.py \
    --config-path="../../tasks/multimodal_v4/finetune/yaml" \
    --config-name="eval_qwen3vl_sft.yaml"
