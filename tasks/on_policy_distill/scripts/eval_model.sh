MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/on_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/eval_entry.py \
    --config-path="../../tasks/on_policy_distill/yamls" --config-name="evaluate_math" \
    training.seq_length=16384 \
    sampler.infer_engine_configs.0.generate_max_tokens=14336 \
    policy.hf_tokenizer_path="save_onp_distill_qwen3_1.7b_base_after_sft/hf/100" \
    sampler.model_info.0.hf_model_path="save_onp_distill_qwen3_1.7b_base_after_sft/hf/100" \
    evaluate_result.output_prefix="onpd_qwen3_1.7b_base_after_sft_100"
