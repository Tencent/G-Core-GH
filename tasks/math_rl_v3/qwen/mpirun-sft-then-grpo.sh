echo 'sft'
bash tasks/math_rl_v3/qwen/mpirun-sft.sh

echo 'prepare for grpo'
mkdir qwen_2_5_1_5b
cp -r qwen_2_5_1_5b_sft/iter_0004476 qwen_2_5_1_5b/release
echo 'release' >qwen_2_5_1_5b/latest_checkpointed_iteration.txt
cp -r qwen_2_5_1_5b qwen_2_5_1_5b_ref

echo 'grpo'
bash tasks/math_rl_v3/qwen/mpirun-grpo.sh
