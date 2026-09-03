MCORE_PATH=${MCORE_PATH:-"/root/Megatron-LM/"}
MBRIDGE_PATH=${MBRIDGE_PATH:-"/root/mbridge"}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-"/root/Megatron-Bridge"}

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export WANDB_KEY="local-f55b2c9b6471378669c57300953758181e49b7ae"
export WANDB_HOST="http://1004.wandb.gcore.woa.com"


export CUDA_LAUNCH_BLOCKING=1

sleep 10
# Reuse the shared mpirun + Ray bootstrap from math_rl_v4 — it's not task-specific.
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_dsv4/yaml" \
    --config-name="math_dsv4_grpo_mcore_sgl.yaml" \
    debug.debug_engine_update_weight=False \
    +policy.override_transformer_config.cross_entropy_loss_fusion=True \
    +policy.override_transformer_config.cross_entropy_fusion_impl=native \
    +policy.override_transformer_config.sequence_packing_scheduler=dp_balanced \
    +policy.ppo_pack_seq=True \
    report.wandb_key=${WANDB_KEY} \
    report.wandb_host=${WANDB_HOST} \
    report.wandb_project="math_dsv4_grpo" \
    report.wandb_exp_name="ep8_cp2" \
    policy.hf_model_path="/data/DeepSeek-V4-Flash-FP8" \
    training.early_swap_model=False \



    #+policy.ppo_pack_seq=True \

#+policy.override_transformer_config.pipeline_model_parallel_layout="Ettt|tL" \
#policy.dist_config.pipeline_model_parallel_size=2 \
