#!/bin/bash

readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

export PYTHONPATH="$PWD:/root/Megatron-LM:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="bond1"
export GLOO_SOCKET_IFNAME="bond1"
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0

export VLLM_USE_V1=1
export VLLM_HOST_IP=$__HOST_IP__
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

readonly DIST_TIMEOUT_MIN=300

readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly GPUS_PER_NODE=${GPUS_PER_NODE:-8}
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

readonly PLACE_CFG_FOLDER=$1
readonly PPO_ROLE=$2
readonly REWARD_TYPE="rule_only"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export RETOOL_DEBUG=1
export RETOOL_DEBUG_FIRST_N=2
# Enable gen_intervals debug logging (logs to retool_output/log/gen_intervals_debug.log)
# Set to 1 to enable, 0 to disable
# Will log first 3 samples of every 10th PPO step
export RETOOL_DEBUG_GEN_INTERVALS=1
# 每个 Step 采样数量
export RETOOL_MONITOR_SAMPLE_COUNT=10
# 是否同时打印到 stdout
export RETOOL_MONITOR_PRINT_STDOUT=0

SAMPLER_NNODES=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-nnodes`
SAMPLER_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-master-addr`
SAMPLER_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-svr-ips`
SAMPLER_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-svr-ports`
SAMPLER_DIST_INIT_ADDRS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-dist-init-addrs`
SAMPLER_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-tp-size`
SAMPLER_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn sampler-pp-size`

CRITIC_NNODES=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn critic-nnodes`
CRITIC_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn critic-master-addr`
CRITIC_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn critic-svr-ips`
CRITIC_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn critic-svr-ports`
CRITIC_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn critic-tp-size`
CRITIC_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn critic-pp-size`

ACTOR_NNODES=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-nnodes`
ACTOR_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-master-addr`
ACTOR_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-svr-ips`
ACTOR_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-svr-ports`
ACTOR_NODE_IPS=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-node-ips`
ACTOR_TP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-tp-size`
ACTOR_PP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-pp-size`
ACTOR_CP_SIZE=`python tools/auto_place.py --fn get --config-folder $PLACE_CFG_FOLDER --get-fn actor-cp-size`

if [ "$PPO_ROLE" = "actor" ]; then
    export MASTER_ADDR="${ACTOR_MASTER_ADDR:-localhost}"
    readonly MASTER_PORT=65531
elif [ "$PPO_ROLE" = "critic" ]; then
    export MASTER_ADDR="${CRITIC_MASTER_ADDR:-localhost}"
    readonly MASTER_PORT=65535
elif [ "$PPO_ROLE" = "sampler" ]; then
    export MASTER_ADDR="${SAMPLER_MASTER_ADDR:-localhost}"
    readonly MASTER_PORT=65534
else
    echo "$PPO_ROLE no support"
    exit 0
fi

export WANDB_BASE_URL=
export WANDB_API_KEY=
# export WANDB_MODE=offline
readonly USERNAME="retool-grpo-gcore"
readonly GRPO_RUN="retool-7b-internal-agent-debug"

export RETOOL_DEBUG_MESSAGES_TRAJ=1
export RETOOL_DEBUG_MESSAGES_TRAJ_STEP_INTERVAL=5
export RETOOL_DEBUG_MESSAGES_TRAJ_FIRST_N=1

readonly ACTOR_TOKENIZER_MODEL="${PWD}/hf-hub/Qwen/Qwen2.5-7B-Instruct/"
readonly RM_TOKENIZER_MODELS="${PWD}/hf-hub/Qwen/Qwen2.5-7B-Instruct/"
readonly DATA_PATH="data/retool/dataset/dapo_math_17k/train/metadata.json"
readonly EVAL_DATA_PATH="data/retool/dataset/aime_2024/train/metadata.json"

# 对每个角色设置加载路径
if [ "$PPO_ROLE" = "actor" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-7b-instruct.yaml"
  readonly LOAD_CHECKPOINT_DIR="${PWD}/../ckpts/qwen_2_5_7b_instruct"
  readonly REF_LOAD_CHECKPOINT_DIR="${LOAD_CHECKPOINT_DIR}"
  readonly SAVE_CHECKPOINT_DIR="${PWD}/../ckpts/${GRPO_RUN}"
  readonly REF_SAVE_CHECKPOINT_DIR="${PWD}/../ckpts/${GRPO_RUN}-ref"
  readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
  readonly TB_DIR="tb/grpo-actor"
  readonly WANDB_DIR="wandb_local/grpo-actor"
  readonly LR=1e-6
elif [ "$PPO_ROLE" = "critic" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-7b-instruct.yaml"
  readonly LOAD_CHECKPOINT_DIR="${PWD}/../ckpts/qwen_2_5_7b_instruct"
  readonly REF_LOAD_CHECKPOINT_DIR=${LOAD_CHECKPOINT_DIR}
  readonly SAVE_CHECKPOINT_DIR="${PWD}/qwen_2_5_rm_7b_save"
  readonly TOKENIZER_MODEL=$RM_TOKENIZER_MODELS
  readonly TB_DIR="tb/grpo-critic"
  readonly WANDB_DIR="wandb_local/grpo-critic"
  readonly LR=1e-6
elif [ "$PPO_ROLE" = "sampler" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-7b-instruct.yaml"
  readonly LOAD_CHECKPOINT_DIR="${PWD}/hf-hub/Qwen/Qwen2.5-7B-Instruct"
  readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR
  readonly SAVE_CHECKPOINT_DIR="none"
  readonly REF_SAVE_CHECKPOINT_DIR="none"
  readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
  readonly TB_DIR="tb/dqa-ppo-sampler"
  readonly WANDB_DIR="wandb_local/grpo-sampler"
  readonly LR=1e-6
else
  echo "$PPO_ROLE no support"
  exit 0
fi

# 确定 topo
readonly CRITIC_DP_SIZE=$(($GPUS_PER_NODE*$CRITIC_NNODES/$CRITIC_TP_SIZE/$CRITIC_PP_SIZE))
readonly SAMPLER_DP_SIZE=$(($GPUS_PER_NODE*$SAMPLER_NNODES/$SAMPLER_TP_SIZE/$SAMPLER_PP_SIZE))
readonly SAMPLER_MP_SIZE=$(($SAMPLER_TP_SIZE*$SAMPLER_PP_SIZE))
readonly ACTOR_DP_SIZE=$(($GPUS_PER_NODE*$ACTOR_NNODES/$ACTOR_TP_SIZE/$ACTOR_PP_SIZE/$ACTOR_CP_SIZE))

if [ "$PPO_ROLE" = "actor" ]; then
    readonly TP_SIZE=$ACTOR_TP_SIZE
    readonly PP_SIZE=$ACTOR_PP_SIZE
    readonly EP_SIZE=1
    readonly CP_SIZE=$ACTOR_CP_SIZE
    readonly DP_SIZE=$ACTOR_DP_SIZE
elif [ "$PPO_ROLE" = "critic" ]; then
    readonly TP_SIZE=$CRITIC_TP_SIZE
    readonly PP_SIZE=$CRITIC_PP_SIZE
    readonly EP_SIZE=1
    readonly CP_SIZE=1
    readonly DP_SIZE=$CRITIC_DP_SIZE
elif [ "$PPO_ROLE" = "sampler" ]; then
    readonly TP_SIZE=$SAMPLER_TP_SIZE
    readonly PP_SIZE=$SAMPLER_PP_SIZE
    readonly EP_SIZE=1
    readonly CP_SIZE=1
    readonly DP_SIZE=$SAMPLER_DP_SIZE
fi

# ============ ReTool: Use Qwen standard tool-use template (no custom system prompt) ============
# System prompt is auto-generated by Qwen tokenizer's apply_chat_template with tools parameter
# This aligns with VERL's approach

# VERL-style answer format (appended to user message)
readonly ANSWER_FORMAT=$'\nThe answer format must be: \\boxed{\'The final answer goes here.\'}'

# VERL-style tools schema (aligned with sandbox_fusion_tool_config.yaml)
readonly TOOLS_JSON='[{"type":"function","function":{"name":"code_interpreter",' \
'"description":"A tool for executing code.","parameters":{"type":"object",' \
'"properties":{"code":{"type":"string","description":"The code to execute."}},' \
'"required":["code"]}}}]'

readonly MAX_CONCURRENCY_ALL_DP=512
readonly ROLLOUT_GLOBAL_BATCH_SIZE=32 # align 512
readonly ROLLOUT_MICRO_BATCH_SIZE=1
readonly ROLLOUT_BATCHING_MAX_BATCH_SIZE=16    #没传入这个参数，retool实验中无效
readonly SHUFFLE_BUFFER_SIZE=$((256*$ROLLOUT_GLOBAL_BATCH_SIZE))
readonly MICRO_BATCH_SIZE=1
readonly GLOBAL_BATCH_SIZE=128  # align 512

# eval args
# 每 5 个 PPO step 进行一次验证 (类似 verl 的 test_freq=5)
readonly PPO_STEP_EVAL_INTERVAL=5
# 每次验证消费 PPO_EVAL_STEPS * PPO_EVAL_ROULLOUT_GLOBAL_BATCH_SIZE 条数据
readonly PPO_EVAL_STEPS=1
readonly PPO_EVAL_ROULLOUT_GLOBAL_BATCH_SIZE=64   # AIME 数据集较小，调小批次
readonly PPO_EVAL_ROULLOUT_MICRO_BATCH_SIZE=1
# 验证时每个 prompt 采样次数 (类似 verl 的 n_resp_per_prompt_val)
readonly PPO_EVAL_SAMPLING_REPEAT=1

readonly PPO_LOGPS_FWD_MICRO_BATCH_SIZE=1
readonly TRAIN_ITERS=-1
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=$((18*1024))
# RESP_SEQ_LENGTH: 所有轮次响应的累计总长度限制 (对齐 verl 的 max_response_length)
# 注意: 这不是每轮的限制，而是 8 轮加起来最多 16384 tokens
readonly RESP_SEQ_LENGTH=$((16*1024))
readonly MAX_SAMPLING_RETRIES=8

# ============ ReTool 特有参数 ============
readonly RETOOL_MAX_TURNS=8
readonly RETOOL_SANDBOX_TIMEOUT=20
# RETOOL_MAX_RESPONSE_PER_TURN: 单轮最大响应 token 数
# NOTE:
# - 这里不再用 4096 去人为限制单轮；让 sglang 自己基于 max_tokens/上下文窗口来处理。
# - 仍保留该参数，但设置为与 --ppo-resp-seq-len 一致（16384）。
readonly RETOOL_MAX_RESPONSE_PER_TURN=$RESP_SEQ_LENGTH

if [ -z "${VLLM_HOST_IP:-}" ]; then
  echo "WARN: VLLM_HOST_IP is empty. Set env VLLM_HOST_IP (or __HOST_IP__) to the host IP if needed by vLLM/sglang."
fi
if [ -z "${SANDBOX_FUSION_URL:-}" ]; then
  echo "WARN: SANDBOX_FUSION_URL is empty. If your ReTool sandbox requires it, export SANDBOX_FUSION_URL before running."
fi

echo "INFO
MPI_RANK $MPI_RANK
PPO_ROLE $PPO_ROLE
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
EP_SIZE $EP_SIZE
CP_SIZE $CP_SIZE
DP_SIZE $DP_SIZE
MICRO_BATCH_SIZE $MICRO_BATCH_SIZE
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
SAMPLER_SVR_IPS $SAMPLER_SVR_IPS
SAMPLER_SVR_PORTS $SAMPLER_SVR_PORTS
SANDBOX_FUSION_URL $SANDBOX_FUSION_URL
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
    --context-parallel-size $CP_SIZE \
    --use-distributed-optimizer \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

if [ "$PPO_ROLE" = "sampler" ]; then
    MP_ARGS="$MP_ARGS
        --use-tp-pp-dp-mapping \
        --no-fused-kernel \
    "
    python tools/auto_place.py --fn init_ray --config-folder $PLACE_CFG_FOLDER
fi


if [ "$PPO_ROLE" = "critic" ]; then
    MP_ARGS="$MP_ARGS
        --no-fused-kernel \
    "
fi

TRAINING_ARGS="
    --use-mcore-models \
    --seq-length $SEQ_LENGTH \
    --seed 1111 \
    --no-check-for-nan-in-loss-and-grad \
    --eod-mask-loss \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr $LR \
    --lr-warmup-iters 0 \
    --lr-decay-style constant \
    --optimizer adam \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend auto \
    --use-flash-attn \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --actor-tokenizer-model ${ACTOR_TOKENIZER_MODEL} \
    --rm-tokenizer-models ${RM_TOKENIZER_MODELS} \
    --dataloader-type external \
    --vocab-file none \
    --merge-file none \
    --num-workers 1 \
    --gdatasetv4-train-metadata-file ${DATA_PATH} \
    --gdatasetv4-eval-metadata-file ${EVAL_DATA_PATH} \
    --px-shuffle-data \
    --px-shuffle-buffer-size ${SHUFFLE_BUFFER_SIZE} \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval -1 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --eval-interval 1 \
    --eval-iters $EVAL_ITERS \
"

if [ "$PPO_ROLE" = "actor" ]; then
    OUTPUT_ARGS="$OUTPUT_ARGS
        --wandb-project $USERNAME \
        --wandb-exp-name $GRPO_RUN-$PPO_ROLE \
        --wandb-save-dir  $WANDB_DIR \
    "
fi


GEN_ARGS="
    --ppo-sort-prompts-across-batches 8 \
    --ppo-rollout-max-prompt-len-diff 128 \
"

EVAL_ARGS="
    --ppo-step-eval-interval $PPO_STEP_EVAL_INTERVAL \
    --ppo-eval-steps $PPO_EVAL_STEPS \
    --ppo-eval-rollout-global-batch-size $PPO_EVAL_ROULLOUT_GLOBAL_BATCH_SIZE \
    --ppo-eval-rollout-micro-batch-size $PPO_EVAL_ROULLOUT_MICRO_BATCH_SIZE \
    --ppo-eval-sampling-repeat $PPO_EVAL_SAMPLING_REPEAT \
"


RL_ARGS="$GEN_ARGS
    --max-concurrency-all-dp $MAX_CONCURRENCY_ALL_DP \
    --ppo-early-swap-model \
    --infer-engine-impl sglang \
    --ppo-auto-calc-args \
    --distributed-timeout-minutes $DIST_TIMEOUT_MIN \
    --ppo-display-rollout-generation \
    --ppo-disable-tqdm \
    --ppo-standalone-sampler \
    --hf-config-json-path $TOKENIZER_MODEL/config.json \
    --ppo-actor-node-ips $ACTOR_NODE_IPS \
    --ppo-actor-data-parallel-size $ACTOR_DP_SIZE \
    --ppo-actor-pipeline-model-parallel-size $ACTOR_PP_SIZE \
    --ppo-critic-ips $CRITIC_SVR_IPS \
    --ppo-critic-ports $CRITIC_SVR_PORTS \
    --ppo-critic-pipeline-model-parallel-size $CRITIC_PP_SIZE \
    --ppo-critic-tensor-model-parallel-size $CRITIC_TP_SIZE \
    --ppo-critic-data-parallel-size $CRITIC_DP_SIZE \
    --ppo-sampler-ips $SAMPLER_SVR_IPS \
    --ppo-sampler-ports $SAMPLER_SVR_PORTS \
    --sampler-dist-init-addrs $SAMPLER_DIST_INIT_ADDRS \
    --ppo-sampler-tensor-model-parallel-size $SAMPLER_TP_SIZE \
    --ppo-sampler-pipeline-model-parallel-size $SAMPLER_PP_SIZE \
    --ppo-sampler-data-parallel-size $SAMPLER_DP_SIZE \
    --ppo-step-update-sampler-interval 1 \
    --ppo-max-epochs 1 \
    --ppo-max-epochs-2 1 \
    --ppo-step-save-interval 100 \
    --ppo-step-per-epoch -1 \
    --ppo-rollout-micro-batch-size $ROLLOUT_MICRO_BATCH_SIZE \
    --ppo-rollout-global-batch-size $ROLLOUT_GLOBAL_BATCH_SIZE \
    --ppo-resp-seq-len $RESP_SEQ_LENGTH \
    --ppo-logps-fwd-micro-batch-size $PPO_LOGPS_FWD_MICRO_BATCH_SIZE \
    --combine-rm-and-critic-server \
    --ppo-rollout-top-p 1.0 \
    --ppo-rollout-top-k -1 \
    --ppo-rollout-temperature 1.0 \
    --ppo-ratio-eps 0.2 \
    --ppo-clip-ratio-low 0.2 \
    --ppo-clip-ratio-high 0.28 \
    --ppo-dual-clip-ratio-c 10.0 \
    --ppo-rm-mask-prompt \
    --rm-output-scalar 1 \
    --rm-output-sequence 0 \
    --ppo-sampling-keeping-strategy all \
    --ppo-sampling-repeat 8 \
    --ppo-sampling-keep 8 \
    --ppo-use-absolute-kl \
    --use-grpo \
    --grpo-advantage-epsilon 1e-4 \
    --grpo-kl-loss-beta 0.0 \
    --rm-head-arch multi_layers \
    --ppo-save-first-rollout-data \
    --ppo-grpo-reward-type rule_only \
    --gen-term-at-nan \
    --ppo-rm-reward-alpha 0.5 \
    --ppo-rule-reward-beta 0.5 \
    --grpo-prefetch-samplings \
    --ppo-dynamic-sampling-max-replay $MAX_SAMPLING_RETRIES \
    --update-weight-max-size-mb 512 \
    --sampler-gpu-memory-utilization 0.6 \
    --gen-rm-gpu-memory-utilization 0.7 \
    --ppo-smart-pad-infer \
    --ppo-smart-pad-train \
    --ppo-train-dynamic-mbs-target-seq 4096 \
    --ppo-train-dynamic-mbs-limit 16 \
    --ppo-update-ref-w-actor-interval 25 \
    --ppo-update-ref-w-actor-coef 1.0 \
    --max-running-requests 64\
    --ppo-rollout-pad-to-multiple-of 2048 \
    --retool-max-turns $RETOOL_MAX_TURNS \
    --retool-sandbox-timeout $RETOOL_SANDBOX_TIMEOUT \
    --retool-max-response-per-turn $RETOOL_MAX_RESPONSE_PER_TURN \
    --no-hook-webapi \
    --ppo-logps-ratio-clamp 20 \
"
#     --ppo-rollout-top-k 20 \

if [[ "$PPO_ROLE" = "sampler" || "$PPO_ROLE" = "gen-rm" || ( "$PPO_ROLE" = "critic" && "$REWARD_TYPE" = "rule_only" ) ]]; then
  RL_ARGS="$RL_ARGS
      --no-fused-kernel \
  "
fi

if [ "$PPO_ROLE" = "actor" ]; then
  RL_ARGS="$RL_ARGS
      --ppo-actor-freeze-ppo-steps 0 \
  "
fi


# # 仅为 sampler 打开 sglang tokenzier 初始化
# if [ "$PPO_ROLE" = "sampler" ]; then
#   RL_ARGS="$RL_ARGS
#       --enable-sglang-openai-api \
#       --tool-call-parser "qwen25" \
#       --sglang-do-tokenizer-init \
#   "
# fi

FINETUNE_ARGS="
    --finetune \
"

ACTOR_NODE_IPS=($ACTOR_NODE_IPS)

# ============ ReTool: 指向 tasks/retool 下的训练脚本 ============
if [ "$PPO_ROLE" = "actor" ]; then
    RUN_PY='./tasks/retool/train_ppo_actor.py'
elif [ "$PPO_ROLE" = "critic" ]; then
    RUN_PY='./tasks/retool/train_ppo_critic.py'
elif [ "$PPO_ROLE" = "sampler" ]; then
    RUN_PY='./tasks/retool/internal_agent/train_ppo_sampler_retool.py'
else
    echo "no ${PPO_ROLE}"
    exit 0
fi

torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    $FINETUNE_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --px-apply-chat-template \
    --px-answer-format "$ANSWER_FORMAT" \
    --px-tools-json "$TOOLS_JSON" \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --save-ref $REF_SAVE_CHECKPOINT_DIR \
    --load-ref $REF_LOAD_CHECKPOINT_DIR 2>&1 &

