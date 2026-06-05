export PYTHONPATH="$PWD:$PYTHONPATH"

export TOKENIZERS_PARALLELISM=false


python3 -u tools/align_loss/generate_hf.py \
    --model_module tasks.deepseek.V2_Lite.align_ckpt.modeling_deepseek \
    --model_class DeepseekV2ForCausalLM \
    --hf_ckpt /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/deepseek-ai/DeepSeek-V2-Lite \
    --max_new_tokens 100 \
    --prompts \
        '李白，字太白' \

    # --prompts \
    #     '1+1的结果是' \
    #     '3-4的结果乘以5等于' \
    #     '5-3>0是对的吗' \
    #     李白，字太白 \
