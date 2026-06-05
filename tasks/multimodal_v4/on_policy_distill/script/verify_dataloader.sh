MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_DEVICE_MAX_CONNECTIONS=1

source tasks/multimodal_v4/on_policy_distill/script/mpirun_run_once.sh \
    tasks/multimodal_v4/on_policy_distill/script/run_once.sh

python3 -u gpatch_v4/entry/verify_dataloader.py \
    --config-path="../../tasks/multimodal_v4/on_policy_distill/yaml" \
    --config-name="qwen3vl_distill"
