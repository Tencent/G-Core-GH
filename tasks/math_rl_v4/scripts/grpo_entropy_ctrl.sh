# 配置参考 
# 1. verl/recipe/entropy
# 2. https://arxiv.org/pdf/2505.22617#page=19.24
MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

# # https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/recipe/dapo/7b_baseline.sh
# # bash tasks/math_rl_v4/scripts/grpo_entropy_ctrl.sh > logs/offpolicy_baseline.log 2>&1 &
# readonly EXP_NAME="offpolicy_baseline"
# python3 -u gpatch_v4/entry/train_lm_grpo.py \
#     --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_entropy_ctrl.yaml" \
#     checkpoint.load_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     checkpoint.save_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     training.train_mbs=1 \
#     +training.exit_step=1000 \
#     report.wandb_exp_name=math_rl_v4_qwen2_7b_${EXP_NAME} \
#     report.log_dir="/mnt/ceph-hz1-csp/mm-base-plt2/user_yeazhao/log/entropy_ctrl/math_rl_v4_qwen2_7b_${EXP_NAME}" \
#     +ppo.ppo_clip_ratio_low=0.2 \
#     +ppo.ppo_clip_ratio_high=0.2

# # https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/recipe/dapo/7b_kl_cov.sh
# # bash tasks/math_rl_v4/scripts/grpo_entropy_ctrl.sh > logs/offpolicy_global_kl_cov.log 2>&1 &
# readonly EXP_NAME="offpolicy_global_kl_cov"
# python3 -u gpatch_v4/entry/train_lm_grpo.py \
#     --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_entropy_ctrl.yaml" \
#     checkpoint.load_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     checkpoint.save_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     training.train_mbs=1 \
#     +training.exit_step=1000 \
#     report.wandb_exp_name=math_rl_v4_qwen2_7b_${EXP_NAME} \
#     report.log_dir="/mnt/ceph-hz1-csp/mm-base-plt2/user_yeazhao/log/entropy_ctrl/math_rl_v4_qwen2_7b_${EXP_NAME}" \
#     +ppo.ppo_clip_ratio_low=1 \
#     +ppo.ppo_clip_ratio_high=1 \
#     +ppo.ppo_logps_ratio_clamp=20.0 \
#     +ppo.ppo_kl_cov_coef=1 \
#     +ppo.ppo_kl_cov_ratio=0.002 \
#     +ppo.ppo_entropy_global_cov=true \
#     +ppo.ppo_entropy_regularization_type=kl-cov

# # https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/recipe/dapo/7b_clip_cov.sh
# # bash tasks/math_rl_v4/scripts/grpo_entropy_ctrl.sh > logs/offpolicy_global_clip_cov.log 2>&1 &
# readonly EXP_NAME="offpolicy_global_clip_cov"
# python3 -u gpatch_v4/entry/train_lm_grpo.py \
#     --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_entropy_ctrl.yaml" \
#     checkpoint.load_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     checkpoint.save_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     training.train_mbs=1 \
#     +training.exit_step=1000 \
#     report.wandb_exp_name=math_rl_v4_qwen2_7b_${EXP_NAME} \
#     report.log_dir="/mnt/ceph-hz1-csp/mm-base-plt2/user_yeazhao/log/entropy_ctrl/math_rl_v4_qwen2_7b_${EXP_NAME}" \
#     +ppo.ppo_clip_ratio_low=1 \
#     +ppo.ppo_clip_ratio_high=1 \
#     +ppo.ppo_kl_cov_coef=1 \
#     +ppo.ppo_clip_cov_ratio=0.0002 \
#     +ppo.ppo_entropy_global_cov=true \
#     +ppo.ppo_entropy_regularization_type=clip-cov

# # https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/recipe/dapo/7b_kl_cov.sh
# # bash tasks/math_rl_v4/scripts/grpo_entropy_ctrl.sh > logs/offpolicy_local_kl_cov.log 2>&1 &
# readonly EXP_NAME="offpolicy_local_kl_cov"
# python3 -u gpatch_v4/entry/train_lm_grpo.py \
#     --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_entropy_ctrl.yaml" \
#     checkpoint.load_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     checkpoint.save_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
#     training.train_mbs=4 \
#     +training.exit_step=1000 \
#     report.wandb_exp_name=math_rl_v4_qwen2_7b_${EXP_NAME} \
#     report.log_dir="/mnt/ceph-hz1-csp/mm-base-plt2/user_yeazhao/log/entropy_ctrl/math_rl_v4_qwen2_7b_${EXP_NAME}" \
#     +ppo.ppo_clip_ratio_low=1 \
#     +ppo.ppo_clip_ratio_high=1 \
#     +ppo.ppo_kl_cov_coef=1 \
#     +ppo.ppo_kl_cov_ratio=0.002 \
#     +ppo.ppo_entropy_global_cov=false \
#     +ppo.ppo_entropy_regularization_type=kl-cov

# https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/recipe/dapo/7b_clip_cov.sh
# bash tasks/math_rl_v4/scripts/grpo_entropy_ctrl.sh > logs/offpolicy_local_clip_cov.log 2>&1 &
readonly EXP_NAME="offpolicy_local_clip_cov"
python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_entropy_ctrl.yaml" \
    checkpoint.load_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
    checkpoint.save_ckpt_path=save_qwen_2_5_7b_${EXP_NAME} \
    training.train_mbs=4 \
    +training.exit_step=1000 \
    report.wandb_exp_name=math_rl_v4_qwen2_7b_${EXP_NAME} \
    report.log_dir="/mnt/ceph-hz1-csp/mm-base-plt2/user_yeazhao/log/entropy_ctrl/math_rl_v4_qwen2_7b_${EXP_NAME}" \
    +ppo.ppo_clip_ratio_low=1 \
    +ppo.ppo_clip_ratio_high=1 \
    +ppo.ppo_clip_cov_ratio=0.0002 \
    +ppo.ppo_entropy_global_cov=false \
    +ppo.ppo_entropy_regularization_type=clip-cov