CONFIG_NAME=$1

MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$PYTHONPATH"

python3 -u gpatch_v4/entry/infer_entry.py \
    --config-path="../../tasks/infer_only" --config-name="$CONFIG_NAME"