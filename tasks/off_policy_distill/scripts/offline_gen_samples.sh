MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/off_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/eval_entry.py \
    --config-path="../../tasks/off_policy_distill/yaml" --config-name="offline_gen_samples" \
    policy.dist_config.nnodes=2 \
    sampler.dist_config.nnodes=2 \
    sampler.infer_engine_configs.0.dist_config.nnodes=2 \
    evaluate_result.output_prefix="qwen3_16_nodes_gen_samples" \
