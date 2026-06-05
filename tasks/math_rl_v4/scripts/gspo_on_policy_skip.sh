MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_on_policy_skip.yaml" \
    ppo.loss_func=gspo \
    ppo.advantage_type=grpo \
    ppo.skip_prev_logps=true \
    checkpoint.load_ckpt_path=save_qwen_2_5_1_5b_gspo_on_policy_skip \
    checkpoint.save_ckpt_path=save_qwen_2_5_1_5b_gspo_on_policy_skip \
    report.wandb_exp_name=math_rl_v4_gspo_on_policy_skip \
    report.log_dir=logs/math_rl_v4_gspo_on_policy_skip
