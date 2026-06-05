MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo_single_ctrl.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_config.yaml" \
    +placement_type=colocate \
    +training.single_controller=True \
    +training.async_rollout=False \
    checkpoint.load_ckpt_path=debug_0512 \
    checkpoint.save_ckpt_path=debug_0512
