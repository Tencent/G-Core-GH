#!/bin/bash
sleep 1

readonly MLM_PATH='../3rdparty/Megatron-LM'
export PYTHONPATH="$PWD:$MLM_PATH:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export NVTE_FUSED_ATTN=0

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"


# # llama3.2 1b
# readonly MODEL_YAML="gpatch/model_yamls/llama3.2-1b.yaml"
# readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
# readonly LOAD_CHECKPOINT_DIR="$PWD/llama_3_2_1b_base"
# readonly SAVE_CHECKPOINT_DIR="$PWD/llama_3_2_1b_sft"
# readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/unsloth/Llama-3.2-1B"
# readonly TB_DIR="tb/llama3_2_1b"
# readonly WANDB_DIR="wandb-save/llama3_2_1b"

# llama3.1 8b
readonly MODEL_YAML="gpatch/model_yamls/llama3.1-8b.yaml"
readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly LOAD_CHECKPOINT_DIR="$PWD/llama_3_1_8b_base"
readonly SAVE_CHECKPOINT_DIR="$PWD/llama_3_1_8b_sft"
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/unsloth/Llama-3.1-8B"
readonly TB_DIR="tb/llama3_1_8b"
readonly WANDB_DIR="wandb-save/llama3_1_8b"

readonly TP_SIZE=1
readonly PP_SIZE=1
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=8

readonly TRAIN_ITERS=1000
readonly EVAL_ITERS=0
readonly SEQ_LEN=2048

export WANDB_API_KEY=
export WANDB_BASE_URL=

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
DP_SIZE $DP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

MP_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
    --use-distributed-optimizer \
"

TRAINER_ARGS="
    --seq-length $SEQ_LEN \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr 1e-5 \
    --min-lr 0 \
    --lr-warmup-iters 0 \
    --lr-decay-style constant \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --attention-backend flash \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type single \
    --num-workers 1 \
    --data-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_data/alpaca-truncated/alpaca-truncated.jsonl \
    --use-map-dataset \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
"
    # --wandb-project huggingface \
    # --wandb-exp-name v3_1-7b_mlm_tp2_pp2 \
    # --wandb-save-dir $WANDB_DIR \

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

RUN_PY="tools/align_loss/finetune_qwen.py"
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
