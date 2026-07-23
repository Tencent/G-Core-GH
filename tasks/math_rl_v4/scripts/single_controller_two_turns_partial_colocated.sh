MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="../mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

# Two 8-GPU nodes: policy uses all 16 GPUs, while sampler/gen-RM split the same pool 8+8.
export GCORE_GPU=2
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo_single_ctrl.py \
    --config-path="../../tasks/math_rl_v4/yaml" \
    --config-name="rl_config_two_turns_partial_colocated.yaml"
