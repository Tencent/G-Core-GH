MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/off_policy_distill/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/eval_entry.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="evaluate_math.yaml"
