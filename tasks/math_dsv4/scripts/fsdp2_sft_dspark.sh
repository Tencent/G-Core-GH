MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

# Optional: pin cluster size, e.g. export GCORE_NNODES=4
export GCORE_NNODES=16
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_dsv4/yaml" --config-name="math_sft_dspark_fsdp2.yaml" \
    policy.ppo_pack_seq=True \
    policy.dist_config.context_parallel_size=8
