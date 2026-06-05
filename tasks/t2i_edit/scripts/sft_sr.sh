MCORE_PATH="/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$PYTHONPATH"

source tasks/t2i_edit/scripts/mpirun-init-ray.sh

python3 -u gpatch_v4/entry/train_t2i_edit_sft.py \
    --config-path="../../tasks/t2i_edit/yaml" --config-name="qwen_image_edit_config"