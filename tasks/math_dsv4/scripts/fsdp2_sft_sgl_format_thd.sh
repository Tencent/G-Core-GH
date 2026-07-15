MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

export GCORE_GPU=16
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_dsv4/yaml" --config-name="math_sft_thd_fsdp2.yaml" \
    policy.hf_model_path="hf-hub/sgl-project/DeepSeek-V4-Flash-FP8" \
    policy.hf_tokenizer_path="hf-hub/sgl-project/DeepSeek-V4-Flash-FP8" \
    optimizer.lr=0.0 \
    training.save_interval=1
