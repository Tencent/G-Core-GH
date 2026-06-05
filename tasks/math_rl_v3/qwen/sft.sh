#!/bin/bash
ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly TP_SIZE=2
readonly PP_SIZE=1
readonly CP_SIZE=2

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly DFS_FOLDER=$PWD
readonly LOAD_CHECKPOINT_DIR="$PWD/qwen_2_5_1_5b_base"
readonly SAVE_CHECKPOINT_DIR="$PWD/qwen_2_5_1_5b_sft"
readonly TOKENIZER_MODEL="$DFS_FOLDER/hf-hub/Qwen/Qwen2.5-Math-1.5B/"

# 注意，建议设置自己的 system_prompt, 并且在下面 args 打开开关
#     --px-apply-chat-template \
#     --px-system-prompt "$system_prompt" \
# 如果无 system_prompt, 则只打开 --px-apply-chat-template 即可
# 比如下面填入 qwen2.5 math 的 prompt
readonly system_prompt="Please reason step by step, and put your final answer within \\boxed{}."

readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=256
readonly TRAIN_ITERS=3357
readonly SEQ_LENGTH=$((4*1024))

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
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
    --use-distributed-optimizer \
    --attention-backend fused \
"

# trainer 的 GBS、LR 等
TRAINER_ARGS="
    --seq-length $SEQ_LENGTH \
    --seed 1111 \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --init-method-std 0.02 \
    --lr 2e-5 \
    --min-lr 0 \
    --lr-warmup-iters 200 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
"

# 数据与 tokenizer
DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type external \
    --num-workers 1 \
    --px-data-config-path tasks/math_rl_v3/qwen/sft_data_config.json \
    --px-shuffle-data \
    --px-shuffle-buffer-size 10000 \
    --px-use-indexed-jsonl-dataset \
    --px-auto-cal-eval-iters \
"

export WANDB_BASE_URL=
export WANDB_API_KEY=

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --gcore-save-retain-interval 1000 \
    --gcore-save-total-limit 5 \
    --tensorboard-dir tb/sft_qwen \
    --tensorboard-log-interval 1 \
    --eval-interval 200 \
    --eval-iters 1 \
    --wandb-project mcore0_13 \
    --wandb-exp-name qwen2.5-sft-tp-${TP_SIZE}-cp-${CP_SIZE} \
    --wandb-save-dir wandb \
"

# 热启动之后去掉这些 flag
FINETUNE_ARGS="
    --finetune \
    --no-load-optim \
    --no-load-rng \
"

export PYTHONPATH="$PWD:../3rdparty/Megatron-LM:../Megatron-LM:$PYTHONPATH"

torchrun $DISTRIBUTED_ARGS tasks/math_rl_v3/sft.py \
    $TRAINER_ARGS \
    $MP_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    --px-apply-chat-template \
    --px-system-prompt "$system_prompt" \
    --cli-arg-yaml-cfgs gpatch/model_yamls/qwen2.5-math-1.5b.yaml \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR
