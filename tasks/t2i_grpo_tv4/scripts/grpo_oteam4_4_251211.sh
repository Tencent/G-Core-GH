MCORE_PATH="/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge/src"
export PYTHONPATH="$PWD:$MCORE_PATH:$PYTHONPATH"

source tasks/t2i_grpo_tv4/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_t2i_grpo.py \
    --config-path="../../tasks/t2i_grpo_tv4/yaml" --config-name="oteam4_4_rl_config_251211.yaml"
