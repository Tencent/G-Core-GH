LIB_PATH="../"
MCORE_PATH="${LIB_PATH}/Megatron-LM/"
MBRIDGE_PATH="${LIB_PATH}/mbridge"
MEGATRON_BRIDGE_PATH="${LIB_PATH}/Megatron-Bridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

export GCORE_NNODES=2
source tasks/math_rl_v4/scripts_xpu_priv/xpu_env_priv.sh
source tasks/math_rl_v4/scripts_xpu_priv/mpirun-init-ray-xpu-priv.sh

python3 -u gpatch_v4/entry/train_lm_finetune.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="math_sft_xpu_priv.yaml" \
    +optimizer.report_post_clip_grad_norm=True
