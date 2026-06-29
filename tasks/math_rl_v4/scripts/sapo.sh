MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

# Qwen3-4B-Base (dense) | sglang rollout + mcore training | DAPO-Math-17k
# 全部配置见 tasks/math_rl_v4/yaml/rl_qwen3_4b_sapo.yaml。
#   - 跑 30B：把 --config-name 改为 rl_qwen3_30b_a3b_sapo.yaml
#   - 临时覆盖：直接在命令行追加 hydra override（会经由 "$@" 透传）

# ============================ SAPO（默认启用）============================
python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" \
    --config-name="rl_qwen3_4b_sapo.yaml" \
    "$@"

# # ============================ GSPO（对比，需手动取消注释）============================
# python3 -u gpatch_v4/entry/train_lm_grpo.py \
#     --config-path="../../tasks/math_rl_v4/yaml" \
#     --config-name="rl_qwen3_4b_sapo.yaml" \
#     ppo.loss_func=gspo \
#     checkpoint.load_ckpt_path=ckpt_qwen3_4b_gspo \
#     checkpoint.save_ckpt_path=ckpt_qwen3_4b_gspo \
#     report.wandb_exp_name=gspo_qwen3_4b \
#     report.log_dir=logs/gspo_qwen3_4b \
#     "$@"

# # ============================ GRPO（对比，需手动取消注释）============================
# python3 -u gpatch_v4/entry/train_lm_grpo.py \
#     --config-path="../../tasks/math_rl_v4/yaml" \
#     --config-name="rl_qwen3_4b_sapo.yaml" \
#     ppo.loss_func=grpo \
#     checkpoint.load_ckpt_path=ckpt_qwen3_4b_grpo_v2 \
#     checkpoint.save_ckpt_path=ckpt_qwen3_4b_grpo_v2 \
#     report.wandb_exp_name=grpo_qwen3_4b \
#     report.log_dir=logs/grpo_qwen3_4b \
#     "$@"
