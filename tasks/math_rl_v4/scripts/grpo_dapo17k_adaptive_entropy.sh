MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_grpo_dapo17k_dyn_cp.yaml" \
    checkpoint.load_ckpt_path=qwen3_4b_dapo17k_dyn_cp_adaptive_entropy \
    checkpoint.save_ckpt_path=qwen3_4b_dapo17k_dyn_cp_adaptive_entropy \
    report.wandb_exp_name=grpo_dapo17k_dyn_cp_adaptive_entropy \
    +ppo.feature_store_enable=true \
    +ppo.use_adaptive_entropy=true \
    ppo.ppo_entropy_bonus=0.01 \
    +ppo.entropy_target=0.2 \
    +ppo.entropy_coef_delta=0.005 \
    +ppo.entropy_coef_min=0.0 \
    +ppo.entropy_coef_max=1.0
