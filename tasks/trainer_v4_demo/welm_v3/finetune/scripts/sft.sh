#!/bin/bash
# WeLM v3 258B SFT demo launch script
#
# Usage:
#   cd gcore-dev && bash tasks/trainer_v4_demo/welm_v3/finetune/scripts/sft.sh
#
# Prerequisites:
#   - provide pathes of Megatron-LM, mbridge, and Megatron-Bridge

MCORE_PATH="/root/Megatron-LM"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

source tasks/trainer_v4_demo/welm_v3/finetune/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/trainer_v4_demo/welm_v3/finetune/yaml" \
    --config-name="welm_v3_sft.yaml"
