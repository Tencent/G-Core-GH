#!/usr/bin/env bash
# PPO (actor + critic) | GSM8K | mcore backend

MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="../mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4_ppo/script/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4_ppo/yaml" --config-name="rl_config.yaml"
