MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Reuse the shared mpirun + Ray bootstrap from math_rl_v4 — it's not task-specific.
export GCORE_GPU=16
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_dsv4/yaml" --config-name="math_dsv4_grpo_sgl_colocated.yaml" \
    debug.debug_engine_update_weight=True \
    policy.hf_model_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang/hf_fp4_fp8_e8m0/150 \
    policy.hf_tokenizer_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang/hf_fp4_fp8_e8m0/150 \
    policy.ref_hf_model_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang/hf_fp4_fp8_e8m0/150

    # policy.hf_model_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang//hf_fp4_fp8/150/ \
    # policy.hf_tokenizer_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang//hf_fp4_fp8/150/ \
    # policy.ref_hf_model_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang//hf_fp4_fp8/150/

    # policy.hf_model_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang//hf/150/ \
    # policy.hf_tokenizer_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang//hf/150/ \
    # policy.ref_hf_model_path=/mnt/geminigmceph/user_yyyuuuzhang/code/gcore-dev-0710/ckpt_miniprogram_grpo_sglang//hf/150/
