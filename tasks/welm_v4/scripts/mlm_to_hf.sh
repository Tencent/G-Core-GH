export PYTHONPATH="$PWD:$3:$4:$PYTHONPATH"

readonly GPUS_PER_NODE=8
readonly NODE_RANK=0
readonly NNODES=1
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="$__POD_IP__"
export CUDA_DEVICE_MAX_CONNECTIONS=1
readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

MLM_INPUT_PATH=$1
HF_SAVE_BASE_PATH=$2
TOKENIZER_MODEL=$5


readonly TP_SIZE=4
readonly PP_SIZE=1
readonly EP_SIZE=1
readonly CP_SIZE=2

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"
readonly convert_way="mlm_to_hf"
# Find all iter_* directories and sort them
ITER_DIRS=($(find "$MLM_INPUT_PATH" -maxdepth 1 -type d -name "iter_*" | sort))
TOTAL_ITERS=${#ITER_DIRS[@]}

echo "MPI_RANK: $MPI_RANK, MPI_SIZE: $MPI_SIZE, TOTAL_ITERS: $TOTAL_ITERS"

# Iterate through checkpoints assigned to this rank
for ((i = MPI_RANK; i < TOTAL_ITERS; i += MPI_SIZE)); do
    ITER_DIR=${ITER_DIRS[$i]}
    ITER_NAME=$(basename "$ITER_DIR")

    echo "Rank $MPI_RANK processing: $ITER_NAME"
    
    # Create HF save path for this iteration
    HF_SAVE_PATH="${HF_SAVE_BASE_PATH}/${ITER_NAME}"

    CKPT_ARGS="
        --convert_way mlm_to_hf \
        --hf_dir $TOKENIZER_MODEL \
        --load_model_path $ITER_DIR \
        --save_model_path $HF_SAVE_PATH \
        --override_args_from_ckpt \
        "

    # export NCCL_DEBUG=INFO
    torchrun $DISTRIBUTED_ARGS tools/px_ckpt_conv/convert_from_bridge.py \
        --tp $TP_SIZE \
        --pp $PP_SIZE \
        --cp $CP_SIZE \
        --ep $EP_SIZE \
        --vpp 1 \
        --dist-ckpt-format torch_dist \
        $CKPT_ARGS >log/convert2-$convert_way-${MPI_RANK}.log 2>&1
    
    echo "Rank $MPI_RANK completed: $ITER_NAME -> $HF_SAVE_PATH"
done

echo "Rank $MPI_RANK finished all assigned checkpoints."




