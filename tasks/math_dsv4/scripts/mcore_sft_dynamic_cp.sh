MCORE_PATH=${MCORE_PATH:-"/root/Megatron-LM/"}
MBRIDGE_PATH=${MBRIDGE_PATH:-"/root/mbridge"}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-"/root/Megatron-Bridge"}

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export WANDB_KEY=${WANDB_KEY:-"local-f55b2c9b6471378669c57300953758181e49b7ae"}
export WANDB_HOST=${WANDB_HOST:-"http://1004.wandb.gcore.woa.com"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reuse the shared mpirun + Ray bootstrap from math_rl_v4.
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh
# Wait for the Ray cluster to start to avoid startup races.
sleep 10

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_dsv4/yaml" \
    --config-name="math_sft_mcore.yaml" \
    training.enable_mtp=False \
    +training.use_dynamic_mbs=False \
    +training.eval_interval=0 \
    +training.eval_before_train=False \
    +training.attention_backend=flash \
    policy.dist_config.context_parallel_size=1 \
    +policy.dist_config.dynamic_context_parallel=True \
    +policy.dist_config.dynamic_cp_scheduler_type=default \
    +policy.dist_config.max_seqlen_per_dp_cp_rank=4096 \
    +policy.dist_config.min_dynamic_context_parallel_size=1 \
    +policy.override_transformer_config.calculate_per_token_loss=True \
    +policy.override_transformer_config.cross_entropy_loss_fusion=True \
    +policy.override_transformer_config.cross_entropy_fusion_impl=native \
    report.wandb_key=${WANDB_KEY} \
    report.wandb_host=${WANDB_HOST} \
    report.wandb_project="math_dsv4_sft" \
    report.wandb_exp_name="ep8_dynamic_cp"
