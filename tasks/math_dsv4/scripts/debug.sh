MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"
# EXTRA_PATH="/mnt/geminigmceph/user_jingtxu/code/gcore_test_dsv4_0713/gcore_patch/user_task_v4"

export PYTHONPATH="$EXTRA_PATH:$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

export GCORE_NNODES=16
source tasks/math_dsv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../" --config-name="sft_fsdp2.yaml" \
