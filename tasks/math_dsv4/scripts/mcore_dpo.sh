MCORE_PATH=${MCORE_PATH:-"/root/Megatron-LM/"}
MBRIDGE_PATH=${MBRIDGE_PATH:-"/root/mbridge"}
MEGATRON_BRIDGE_PATH=${MEGATRON_BRIDGE_PATH:-"/root/Megatron-Bridge"}

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

export WANDB_KEY="local-f55b2c9b6471378669c57300953758181e49b7ae"
export WANDB_HOST="http://1004.wandb.gcore.woa.com"

source tasks/math_dsv4/scripts/mpirun-init-ray.sh
sleep 10

python3 -u gpatch_v4/entry/train_dpo.py \
    --config-path="../../tasks/math_dsv4/yaml" \
    --config-name="dsv4_dpo_mcore_small.yaml" \
    training.enable_mtp=False \
    +policy.override_transformer_config.sequence_packing_scheduler=dp_balanced \
    policy.dist_config.context_parallel_size=8 \
    report.wandb_key=${WANDB_KEY} \
    report.wandb_host=${WANDB_HOST} \
    report.wandb_project="math_dsv4_dpo" \
    report.wandb_exp_name="ep8_cp8" \
