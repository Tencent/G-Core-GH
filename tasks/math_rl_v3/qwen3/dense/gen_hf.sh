export PYTHONPATH="$PWD:$PYTHONPATH"

export TOKENIZERS_PARALLELISM=false
    # --hf_ckpt /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_xiaotaoliu/code/test_hf_load/welm_moe_1.5B/ \
    # --hf_ckpt /mnt/gemininjceph/geminicephfs/mm-base-plt2/user_erikfu/moe/welm_moe_32b_8k-sft_20240812/ \

# 1.7B
# MODEL_PATH="/mnt/gemininjceph2/geminicephfs/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen3-1.7B"
# 32B
MODEL_PATH="/mnt/gemininjceph2/geminicephfs/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen3-32B"

python3 -u tools/align_loss/generate_hf.py \
    --hf_ckpt $MODEL_PATH \
    --max_new_tokens 32 \
    --prompts \
        '1+1的结果是' \
        '3-4的结果乘以5等于' \
        '5-3>0是对的吗' \
        李白，字太白 \
