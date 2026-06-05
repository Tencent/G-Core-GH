export PYTHONPATH="$PWD:$PYTHONPATH"

# 处理成 jsonl 方便点

python3 tasks/math_rl_v3/qwen/preprocess_sft_data_phase_1.py \
    $PWD/hf-hub/AI-MO/NuminaMath-CoT/data \
    $PWD/hf-hub/AI-MO/NuminaMath-CoT-jsonl \

python3 tasks/math_rl_v3/qwen/preprocess_sft_data_phase_1.py \
    $PWD/hf-hub/openai/gsm8k/main \
    $PWD/hf-hub/openai/gsm8k-jsonl \

python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $PWD/hf-hub/AI-MO/NuminaMath-CoT-jsonl/train \
    --data_file_postfix 'jsonl' \
    --domain_name 'NuminaMath'

python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $PWD/hf-hub/AI-MO/NuminaMath-CoT-jsonl/eval \
    --data_file_postfix 'jsonl' \
    --domain_name 'NuminaMath'

python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $PWD/hf-hub/openai/gsm8k-jsonl/train \
    --data_file_postfix 'jsonl' \
    --domain_name 'gsm8k'

python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $PWD/hf-hub/openai/gsm8k-jsonl/eval \
    --data_file_postfix 'jsonl' \
    --domain_name 'gsm8k'