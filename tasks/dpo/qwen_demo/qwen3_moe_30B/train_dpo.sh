#!/bin/bash
sleep 1

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1
export HF_DATASETS_OFFLINE=1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"


export WANDB_API_KEY=
export WANDB_BASE_URL=

USERNAME="huggingface"
WANDB_NAME="dpo_qwen3_30b_v3_policy_cp2_random_pad"
WANDB_DIR="wandb_local/qwen3_30b"

readonly DFS_PATH="/mnt/ceph-sz2-csp/mm-base-plt2"
readonly MODEL_YAML="gpatch/model_yamls/qwen3-30b-a3b-moe.yaml"
readonly LOAD_CHECKPOINT_DIR="$PWD/qwen3_30b_a3b_moe_mlm_dpo"
readonly SAVE_CHECKPOINT_DIR="$PWD/qwen3_30b_a3b_moe_mlm_dpo_save"
readonly TOKENIZER_MODEL="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-30B-A3B"
readonly TB_DIR="tb/train-dpo"

readonly DATA_CONFIG="tasks/dpo/qwen_demo/qwen3_moe_30B/dpo_data.json"
# readonly DATA_CONFIG="tasks/dpo/qwen_demo/qwen3_moe_30B/both.json"

readonly TP_SIZE=2
readonly PP_SIZE=8
readonly CP_SIZE=2
readonly EP_SIZE=2
readonly DP_SIZE=$(($GPUS_PER_NODE*$NNODES/$TP_SIZE/$PP_SIZE/$CP_SIZE))
readonly MICRO_BATCH_SIZE=$((1*2))
readonly GLOBAL_BATCH_SIZE=$((16*2))

readonly TRAIN_ITERS=1000
readonly EVAL_ITERS=0
readonly SEQ_LEN=$((8*1024))

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
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
    --context-parallel-size $CP_SIZE \
    --expert-model-parallel-size $EP_SIZE \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --use-distributed-optimizer \
"

TRAINING_ARGS="
    --seq-length $SEQ_LEN \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --init-method-std 0.02 \
    --lr 5e-7 \
    --min-lr 0 \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend flash \
"


DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-use-indexed-jsonl-dataset \
    --px-data-config-path $DATA_CONFIG \
    --px-shuffle-data \
    --px-shuffle-buffer-size 500000 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
    --wandb-project $USERNAME \
    --wandb-exp-name $WANDB_NAME \
    --wandb-save-dir  $WANDB_DIR \
"

# --dpo-policy-ref-model-cnt 1 when model-using ref or policy
# --dpo-policy-ref-model-cnt 2 when model-using both
DPO_ARGS="
    --dpo \
    --dpo-beta 0.1 \
    --dpo-label-smoothing 0. \
    --dpo-ftx-gamma 0 \
    --dpo-reward-models-cnt 0 \
    --dpo-model-using policy \
    --dpo-policy-ref-model-cnt 1 \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-save-rng \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
"

EXPERT_ARGS="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-pad-with-random-token \
"

RUN_PY="./tasks/dpo/train_dpo.py"
torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $TRAINING_ARGS \
    $EXPERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $DPO_ARGS \
    $FINETUNE_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
