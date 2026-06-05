MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/off_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/eval_entry.py \
    --config-path="../../tasks/off_policy_distill/yaml" --config-name="evaluate_math" \
    policy.hf_tokenizer_path="qwen3_1-7b_base_mlm/hf_converted_800/" \
    sampler.model_info.0.hf_model_path="qwen3_1-7b_base_mlm/hf_converted_800/" \
    evaluate_result.output_prefix="old_code_sft_800" \

    # policy.hf_tokenizer_path="hf-hub/Qwen/Qwen3-32B/" \
    # sampler.model_info.0.hf_model_path="hf-hub/Qwen/Qwen3-32B/" \
    # evaluate_result.output_prefix="qwen3-32b_raw_v1" \

    # policy.hf_tokenizer_path="hf-hub/Qwen/Qwen3-1.7B" \
    # sampler.model_info.0.hf_model_path="hf-hub/Qwen/Qwen3-1.7B" \
    # evaluate_result.output_prefix="qwen3-1.7b_raw_v3"
