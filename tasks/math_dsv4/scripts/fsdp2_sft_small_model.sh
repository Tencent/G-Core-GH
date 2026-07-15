MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

# Reuse the shared mpirun + Ray bootstrap from math_rl_v4 — it's not task-specific.
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_dsv4/yaml" --config-name="math_sft_with_mtp_fsdp2.yaml" \
    debug.debug_truncate_num_hidden_layers=4 \
    training.auto_load_from_save_ckpt=False \
    training.save_interval=1000 \
    training.enable_mtp=True \
    training.exit_step=50 \
    +report.log_level=debug \
    optimizer.lr=1e-5 \
    checkpoint.save_ckpt_path=save_dsv4_sft_cp2_l4_wo_mtp \
    report.wandb_exp_name=math_dsv4_cp2_l4_wo_mtp
