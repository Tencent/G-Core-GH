export PYTHONPATH="$PWD/tools/convert_fp8:$PYTHONPATH:../Megatron-LM"

python tools/convert_fp8/bf16_cast_fp8.py \
  --input-bf16-hf-path /mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/gcore-dev/deepseek-v3-sft-bf16/hf_iter_69 \
  --output-fp8-hf-path ./DeepSeek-sft-bf16-FP8E4M3_block128x128-fp8-gate-test \
   --input-fp8-hf-path /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/deepseek-ai/DeepSeek-V3

# python tools/convert_deepseek/bf16_cast_fp8.py \
#   --input-bf16-hf-path /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_jeffhong/ai-search/hf_ckpt/lesliejiang-iter_0000060 \
#   --output-fp8-hf-path /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_jeffhong/ai-search/hf_ckpt/lesliejiang-bf16-FP8E4M3_block128x128-fp8 \
#   --input-fp8-hf-path /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_jeffhong/hf-hub/deepseek-ai/DeepSeek-V3

# python tools/convert_fp8/bf16_cast_fp8.py \
#   --input-bf16-hf-path /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Qwen/Qwen2.5-Math-1.5B/ \
#   --output-fp8-hf-path /mnt/ceph-hz1-csp/mm-base-plt2/user_jeffhong/workspace/wepsdl-dev/gcore-dev/Qwen2.5-Math-1.5B-FP8