MCORE_PATH="../Megatron-LM"
MBRIDGE_PATH="../mbridge"
MEGATRON_BRIDGE_PATH="../Megatron-Bridge"

export PYTHONPATH="$PWD:$MEGATRON_BRIDGE_PATH/src:$MCORE_PATH:$MBRIDGE_PATH:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_rl_v4/yaml" \
    --config-name="math_sft_lora.yaml" \
    +optimizer.report_post_clip_grad_norm=True
