MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_on_policy_skip.yaml" \
    +training.im_end_metrics_enable=true \
    checkpoint.load_ckpt_path=save_qwen_2_5_1_5b_grpo_im_end_metrics \
    checkpoint.save_ckpt_path=save_qwen_2_5_1_5b_grpo_im_end_metrics \
    report.wandb_exp_name="math_rl_v4_grpo_im_end_metrics" \
    report.log_dir="logs/math_rl_v4_grpo_im_end_metrics"
