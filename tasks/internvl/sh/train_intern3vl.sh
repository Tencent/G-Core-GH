#!/bin/bash

export NVTE_FLASH_ATTN=1
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# export CUDA_LAUNCH_BLOCKING=1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65515
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

MYWD=$PWD
readonly HF_HUB_DIR="$MYWD/hf-hub/OpenGVLab/InternVL3-14B"


readonly LOAD_CHECKPOINT_DIR=${HF_HUB_DIR}
readonly SAVE_CHECKPOINT_DIR="${PWD}/ckpt_internvl3_2b_sft"
readonly TOKENIZER_MODEL=${HF_HUB_DIR}
readonly TB_DIR="tb/invernvl3-2B"
readonly LMDB_PATH="${MYWD}/hf-hub/BUAADreamer/llava-en-zh-300k/gcore-data/zh/img_file.lmdb"
readonly LMDB_PORT=8300
readonly MODEL_YAML="gpatch/model_yamls/internvl3-14b.yaml"

readonly TP_SIZE=4
readonly PP_SIZE=1
readonly EP_SIZE=1
readonly CP_SIZE=1
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=8

readonly TRAIN_ITERS=20000
readonly LR_WARMUP_ITERS=0
readonly EVAL_ITERS=10
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
CP_SIZE $CP_SIZE
DP_SIZE $DP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
"

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
    --use-distributed-optimizer \
    --attention-backend flash \
"

TRAINER_ARGS="
    --lr 0 \
    --min-lr 0 \
    --lr-decay-style constant \
    --optimizer adam \
    --weight-decay 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --clip-grad 1.0 \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --seq-length 4096 \
    --decoder-seq-length 4096 \
    --img-h 448 \
    --img-w 448 \
    --patch-dim 14 \
    --no-rope-fusion \
    --no-save-optim \
    --seed 42 \
    --ckpt-format torch_dist \
    --mask-history \
    --try-load-from-hf \
    --hf-model-path ${HF_HUB_DIR} \
    --attention-softmax-in-fp32 \
    --no-bias-swiglu-fusion \
    --no-persist-layer-norm \
    --no-bias-dropout-fusion \
    --export-to-hf \
    --processor-path ${HF_HUB_DIR} \
    --mbridge-distributed-filesystem \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --px-data-config-path examples/guanyouhe/llava-en-zh-300k-v3.json \
    --dataloader-type external \
    --num-workers 8 \
    --timing-log-level 1 \
    --lmdb-port $LMDB_PORT \
    --px-reset-dataloader-at-start-of-eval \
    --eod-mask-loss \
"

export WANDB_API_KEY=
export WANDB_BASE_URL=
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters $EVAL_ITERS \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --wandb-project internvl \
    --wandb-exp-name 2b/dp${DP_SIZE}/tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}/mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE} \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

nohup python megatron_datasets/tools/lmdb_read_svr.py \
    --lmdb-path $LMDB_PATH \
    --lmdb-map-size 500 \
    --lmdb-port $LMDB_PORT > svr.log 2>&1 &


export PYTHONPATH=../Megatron-LM:../mbridge:$PYTHONPATH

PYTHONPATH="${PWD}:$PYTHONPATH" torchrun $DISTRIBUTED_ARGS \
    tasks/internvl/train_internvl.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR 

pkill -f -9 lmdb_read_svr.py
