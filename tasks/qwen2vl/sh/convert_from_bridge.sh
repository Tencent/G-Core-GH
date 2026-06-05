# !/bin/bash
# $1: convert type: hf_to_mlm/mlm_to_hf

readonly CONVERT_TYPE=$1

readonly MCORE_PATH='../Megatron-LM'
export PYTHONPATH="$PWD:$MCORE_PATH:../mbridge:$PYTHONPATH"

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/Qwen/Qwen2.5-VL-3B-Instruct"

readonly TP_SIZE=1
readonly PP_SIZE=1
readonly EP_SIZE=1
readonly CP_SIZE=1

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

if [ "$CONVERT_TYPE" == "hf_to_mlm" ]; then
    MLM_OUTPUT_DIR="${PWD}/ckpt_qwen2p5vl_mbridge"

    CKPT_ARGS="
        --convert_way hf_to_mlm \
        --hf_dir $HF_HUB_DIR \
        --load_model_path $HF_HUB_DIR \
        --save_model_path $MLM_OUTPUT_DIR \
    "
elif [ "$CONVERT_TYPE" == "mlm_to_hf" ]; then
    MLM_INPUT_DIR="${PWD}/ckpt_qwen2p5vl_mbridge/release"
    HF_OUTPUT_DIR="${PWD}/ckpt_qwen2p5vl_mbridge/hf_release"

    CKPT_ARGS="
        --convert_way mlm_to_hf \
        --hf_dir $HF_HUB_DIR \
        --load_model_path $MLM_INPUT_DIR \
        --save_model_path $HF_OUTPUT_DIR \
        --override_args_from_ckpt \
    "
else
    echo "not support this convert type:${CONVERT_TYPE}"
    exit 0
fi

torchrun $DISTRIBUTED_ARGS tools/px_ckpt_conv/convert_from_bridge.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --cp $CP_SIZE \
    --ep $EP_SIZE \
    --dist-ckpt-format torch_dist \
    $CKPT_ARGS

if [ "$CONVERT_TYPE" == "mlm_to_hf" ]; then
    FILES=(
        "chat_template.json"
        "config.json"
        "generation_config.json"
        "model.safetensors.index.json"
        "preprocessor_config.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )

    for file in "${FILES[@]}"; do
        if [ -f "$HF_OUTPUT_DIR/$file" ]; then
            echo "exist: $HF_OUTPUT_DIR/$file"
        else
            cp "$HF_HUB_DIR/$file" $HF_OUTPUT_DIR
            echo "copyed: $file $HF_OUTPUT_DIR"
        fi
    done
fi
