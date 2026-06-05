#!/bin/bash
sleep 1

export PYTHONPATH="$PWD:$PYTHONPATH"
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

# qwen3 moe 235B-a22b
readonly MODEL_YAML="gpatch/model_yamls/qwen3-235b-a22b-moe.yaml"
# readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly DFS_FOLDER="/mnt/ceph-sz2-csp/mm-base-plt2/"
readonly LOAD_CHECKPOINT_DIR="/mnt/ceph-sz2-csp/mm-base-plt2/user_xiaotaoliu/code/gcore-dev/qwen3_235b_a22b_moe_mlm"
readonly SAVE_CHECKPOINT_DIR="$PWD/qwen3_235b_a22b_moe_mlm_sft_save"
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/Qwen/Qwen3-235B-A22B"
readonly EXP_NAME="v3_235b_mlm"
readonly TB_DIR="tb/qwen3_235b_a22b"
readonly WANDB_DIR="wandb-save/qwen3_235b_a22b"


readonly TP_SIZE=4
readonly PP_SIZE=8
readonly EP_SIZE=4
readonly CP_SIZE=4
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=16

readonly TRAIN_ITERS=1000
readonly EVAL_ITERS=0
readonly SEQ_LEN=4096

export WANDB_API_KEY=
export WANDB_BASE_URL=

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
CP_SIZE $CP_SIZE
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
    --decoder-first-pipeline-num-layers 11 \
    --decoder-last-pipeline-num-layers 11 \
    --context-parallel-size $CP_SIZE \
    --expert-model-parallel-size $EP_SIZE \
    --expert-tensor-parallel-size 1 \
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
    --lr-warmup-iters 200 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --attention-backend flash \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-use-indexed-jsonl-dataset \
    --px-data-config-path tasks/math_rl_v3/qwen/sft_data_config.json \
    --px-shuffle-data \
    --px-shuffle-buffer-size 10000 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
    --no-save-rng \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
"
    # --wandb-project huggingface \
    # --wandb-exp-name 235b_mlm_run \
    # --wandb-save-dir wandb-save \

EXPERT_ARGS="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type allgather \
    --moe-pad-with-random-token \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

RUN_PY="tasks/math_rl_v3/sft.py"
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $TRAINER_ARGS \
    $MP_ARGS \
    $EXPERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
