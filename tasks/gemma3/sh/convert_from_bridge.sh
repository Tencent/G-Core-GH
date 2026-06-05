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
readonly HF_HUB_DIR="$MYWD/hf-hub/google/gemma-3-4b-it"

readonly TP_SIZE=2
readonly PP_SIZE=2
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
    MLM_OUTPUT_DIR="${PWD}/ckpt_gemma3_mbridge"

    CKPT_ARGS="
        --convert_way hf_to_mlm \
        --hf_dir $HF_HUB_DIR \
        --load_model_path $HF_HUB_DIR \
        --save_model_path $MLM_OUTPUT_DIR \
    "
elif [ "$CONVERT_TYPE" == "mlm_to_hf" ]; then
    MLM_INPUT_DIR="${PWD}/ckpt_gemma3_mbridge/release"
    HF_OUTPUT_DIR="${PWD}/ckpt_gemma3_mbridge/hf_release"

    CKPT_ARGS="
        --convert_way mlm_to_hf \
        --hf_dir $HF_HUB_DIR \
        --load_model_path $MLM_INPUT_DIR \
        --save_model_path $HF_OUTPUT_DIR \
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
