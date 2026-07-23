MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Reuse the shared mpirun + Ray bootstrap from math_rl_v4 — it's not task-specific.
export GCORE_NNODES=4
export VLLM_ENABLE_CUDA_COMPATIBILITY=1

source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_dsv4/yaml" --config-name="test_dsv4_flash_small.yaml" \
    +policy.ppo_pack_seq=True \
    +ppo.loss_func=grpo \
    +training.ppo_dump_metrics_interval=1 \
    +training.ppo_dump_metrics_dir=debug-tmp/grpo_thd_alignment \
    +training.exit_step=1 \
    training.auto_load_from_save_ckpt=False \
    sampler.infer_engine_configs.0.generate_max_tokens=2048 \
    debug.debug_engine_update_weight=False \
    +debug.debug_truncate_num_hidden_layers=4


# python3 -u gpatch_v4/entry/train_lm_grpo_single_ctrl.py \
#     --config-path="../../tasks/math_dsv4/yaml" --config-name="test_dsv4_flash_small_disagg.yaml" \
#     debug.debug_engine_update_weight=True \
#     +debug.debug_truncate_num_hidden_layers=4

# PYTHONPATH="$PWD:/root/Megatron-LM:$PYTHONPATH" python -m gpatch_v4.utils.grpo_thd_alignment --require-multi-segment debug-tmp/grpo_thd_alignment/tmp/PpoStep0_SubEpoch0_*/dp*_rank*.pt