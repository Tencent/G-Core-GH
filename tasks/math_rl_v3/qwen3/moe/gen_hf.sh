export PYTHONPATH="$PWD:$PYTHONPATH"

export TOKENIZERS_PARALLELISM=false


# moe 30B-A3B
MODEL_PATH="/mnt/gemininjceph2/geminicephfs/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen3-30B-A3B/"

python3 -u tools/align_loss/generate_hf.py \
    --hf_ckpt $MODEL_PATH \
    --max_new_tokens 32 \
    --prompts \
        '1+1的结果是' \
        '3-4的结果乘以5等于' \
        '5-3>0是对的吗' \
        李白，字太白 \
