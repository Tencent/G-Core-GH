#!/bin/bash

ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

set -ex

export PYTHONPATH="$PWD:../../Megatron-LM:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly DFS_FOLDER='/mnt/ceph-hz1-csp/mm-base-plt2'
# readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly LOAD_CHECKPOINT_DIR="$PWD/deepseek-v2-lite-mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/deepseek-v2-lite-sft"
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/deepseek-ai/DeepSeek-V2-Lite"

readonly TP_SIZE=1
readonly PP_SIZE=1
readonly CP_SIZE=1
readonly EP_SIZE=8
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=512
readonly SEQ_LENGTH=$((4*1024))

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# 并行度
MP_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
    --context-parallel-size $CP_SIZE \
    --expert-model-parallel-size $EP_SIZE \
    --use-distributed-optimizer \
    --attention-backend fused \
"

readonly MODEL_YAML="gpatch/model_yamls/deepseek-v2-lite.yaml"

TRAINER_ARGS="
    --use-mcore-models \
    --sequence-parallel \
    --use-flash-attn \
    --disable-bias-linear \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --no-bias-swiglu-fusion \
    --no-check-for-nan-in-loss-and-grad \
    --no-rope-fusion \
    --lr-warmup-init 1.3e-7 \
    --lr 1.3e-6 \
    --min-lr 1.3e-7 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --train-iters 500000 \
"


DATASET_PATH="/mnt/ceph-hz1-csp/mm-base-plt2/user_jeffhong/datasets"

# 数据与 tokenizer
DATA_ARGS="
    --seq-length $SEQ_LENGTH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --vocab-file $DATASET_PATH/oscar/gpt2-vocab.json \
    --merge-file $DATASET_PATH/oscar/gpt2-merges.txt \
    --data-path $DATASET_PATH/oscar-mcore/oscardata_text_document \
    --split 99,1,0 \
    --no-mmap-bin-files \
    --no-create-attention-mask-in-dataloader \
    --num-workers 6 \
"

export WANDB_BASE_URL=
export WANDB_API_KEY=

TS=$(date +%Y-%m-%d-%H-%M-%S)

EVAL_AND_OUTPUT_ARGS="
    --eval-interval 200 \
    --eval-iters 32 \
    --log-interval 1 \
    --log-throughput \
    --save-interval 500 \
    --tensorboard-dir tb/sft \
    --tensorboard-log-interval 1 \
    --wandb-project jeffhong \
    --wandb-exp-name $TS/deepseek-v2-lite-tp-${TP_SIZE}-cp-${CP_SIZE}-ep-${EP_SIZE} \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

readonly MLM_PATH=/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_guanyouhe/sync_code/Megatron-LM
export PYTHONPATH="$MLM_PATH:$PYTHONPATH"

torchrun $DISTRIBUTED_ARGS tasks/deepseek/V2_Lite/align_moe_zoo/sft.py \
    $MP_ARGS \
    $DATA_ARGS \
    $TRAINER_ARGS \
    $EVAL_AND_OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
