MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/off_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_off_policy_distill.py \
    --config-path="../../tasks/off_policy_distill/yaml" --config-name="off_policy_distill_demo" \
    +checkpoint.convert_mcore_to_hf_offline=True \
    +checkpoint.convert_target_step=699 \
    checkpoint.mbridge_distributed_filesystem=True \
