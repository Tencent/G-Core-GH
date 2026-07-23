MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Reuse the shared mpirun + Ray bootstrap from math_rl_v4 — it's not task-specific.
export GCORE_NNODES=16
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_dsv4/yaml" --config-name="math_dsv4_grpo_sgl_colocated.yaml" \
    policy.ppo_pack_seq=True \
    training.auto_load_from_save_ckpt=False \
    +policy.fp8_qat=True \
    +training.linear_ce_backend=separate \
    +training.use_linear_ce=True

    # debug.debug_engine_update_weight=True \
    # +policy.fp8_qat=True \
    # +policy.fp4_qat=True

    # +training.ppo_dump_metrics_interval=1 \
    # +training.ppo_dump_metrics_dir=debug-tmp/grpo_thd_alignment \