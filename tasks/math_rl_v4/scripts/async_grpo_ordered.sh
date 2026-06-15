MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo_single_ctrl.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config_async.yaml" \
    +training.rollout_ordered_collection=True \
    checkpoint.save_ckpt_path=save_qwen_2_5_1_5b_grpo_async_rollout_ordered \
    report.wandb_exp_name=math_rl_v4_grpo_async_ordered
