MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_rl_v4/yaml/" --config-name="math_sft.yaml" \
    +checkpoint.convert_mcore_to_hf_offline=True \
    +checkpoint.convert_target_step=3357 \
    +checkpoint.export_hf_save_path=sft_convert/hf \
    +policy.dist_config.nnodes=1 \
    checkpoint.load_ckpt_path=./save_qwen_2_5_1_5b_sft \
    checkpoint.save_ckpt_path=./save_qwen_2_5_1_5b_sft \
    checkpoint.mbridge_distributed_filesystem=True \
    +'checkpoint.override_tokenizer_special_token={eos_token: "<|im_end|>"}'