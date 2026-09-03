MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_grpo_dapo17k_dyn_cp.yaml"

    # +report.profile.enable_profile=True \
    # +report.profile.profile_start_step=3 \
    # +report.profile.profile_end_step=4

    # --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_grpo_dapo17k_dyn_cp.yaml"
    # --config-path="../../tasks/math_rl_v4/yaml" --config-name="rl_grpo_dapo17k_config.yaml"
