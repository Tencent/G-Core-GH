#!/bin/bash
# TODO: align with https://github.com/volcengine/verl/blob/main/recipe/dapo/test_dapo_7b_math.sh
readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

export PYTHONPATH="$PWD:../3rdparty/Megatron-LM-core_v0.13.0:$PYTHONPATH"
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

readonly WANDB_PROJECT="yeazhao_dapo"
readonly WANDB_EXP_NAME="qwen2.5-dapo-gdatasetv4-sglang"
export WANDB_BASE_URL=
export WANDB_API_KEY=

readonly ACTOR_TOKENIZER_MODEL="${PWD}/hf-hub/Qwen/Qwen2.5-Math-1.5B/"
readonly RM_TOKENIZER_MODELS="${PWD}/hf-hub/Qwen/Qwen2.5-Math-RM-72B/"
readonly data_path="tasks/math_rl_v3/qwen/rl_metadata.json"
readonly eval_data_path="tasks/math_rl_v3/qwen/rl_eval_metadata.json"

if [ "$PPO_ROLE" = "actor" ]; then
    readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-math-1.5b.yaml"
    readonly LOAD_CHECKPOINT_DIR="${PWD}/ckpts/qwen_2_5_1_5b_sft"
    readonly REF_LOAD_CHECKPOINT_DIR="${PWD}/ckpts/qwen_2_5_1_5b_ref"
    readonly SAVE_CHECKPOINT_DIR="${PWD}/ckpts/qwen_2_5_1_5b_save"
    readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
    readonly TB_DIR="tb/grpo-actor"
    readonly WANDB_DIR="wandb_local/grpo-actor"
    readonly LR=5e-7
elif [ "$PPO_ROLE" = "critic" ]; then
    readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-math-rm-72b.yaml"
    readonly LOAD_CHECKPOINT_DIR="${PWD}/ckpts/qwen_2_5_rm_72b"
    readonly REF_LOAD_CHECKPOINT_DIR=${LOAD_CHECKPOINT_DIR}
    readonly SAVE_CHECKPOINT_DIR="${PWD}/ckpts/qwen_2_5_rm_72b_save"
    readonly TOKENIZER_MODEL=$RM_TOKENIZER_MODELS
    readonly TB_DIR="tb/grpo-critic"
    readonly WANDB_DIR="wandb_local/grpo-critic"
    readonly LR=5e-7
elif [ "$PPO_ROLE" = "sampler" ]; then
    readonly MODEL_YAML="gpatch/model_yamls/qwen2.5-math-1.5b.yaml"
    # LOAD_CHECKPOINT_DIR 填入 hf ckpt 
    readonly LOAD_CHECKPOINT_DIR="${PWD}/hf-hub/Qwen/Qwen2.5-Math-1.5B/"
    readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR
    readonly SAVE_CHECKPOINT_DIR="none"
    readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
    readonly TB_DIR="tb/dqa-ppo-sampler"
    readonly WANDB_DIR="wandb_local/grpo-sampler"
    readonly LR=5e-7
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

readonly ROLLOUT_GLOBAL_BATCH_SIZE=256
readonly ROLLOUT_MICRO_BATCH_SIZE=1
readonly MICRO_BATCH_SIZE=1
readonly SHUFFLE_BUFFER_SIZE=$((256*$ROLLOUT_GLOBAL_BATCH_SIZE))
if [ "$PPO_ROLE" = "actor" ]; then
    readonly GLOBAL_BATCH_SIZE=256
else
    # sampler 和 rm 不需要训练，没有真正的 GBS，设置成 dp size，方便 scaling。
    readonly GLOBAL_BATCH_SIZE=$DP_SIZE
fi

# eval args
# 每 PPO_STEP_EVAL_INTERVAL 个 ppo_step 进入一次 eval_loop
# 每次 eval_loop 消费 PPO_EVAL_STEPS * PPO_EVAL_ROULLOUT_GLOBAL_BATCH_SIZE 条数据
# metrics 根据所有消费的数据计算 sample-wise mean
readonly PPO_STEP_EVAL_INTERVAL=0
readonly PPO_EVAL_STEPS=1
readonly PPO_EVAL_ROULLOUT_GLOBAL_BATCH_SIZE=256
readonly PPO_EVAL_ROULLOUT_MICRO_BATCH_SIZE=1
readonly PPO_EVAL_SAMPLING_REPEAT=8

readonly PPO_LOGPS_FWD_MICRO_BATCH_SIZE=1
readonly TRAIN_ITERS=-1
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=$((4*1024))
readonly RESP_SEQ_LENGTH=$((1024))
readonly MAX_SAMPLING_RETRIES=8

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
GRADIENT_ACCUMULATE_STEP $GRADIENT_ACCUMULATE_STEP
GLOBAL_BATCH_SIZE $GLOBAL_BATCH_SIZE
SAMPLER_SVR_IPS $SAMPLER_SVR_IPS
SAMPLER_SVR_PORTS $SAMPLER_SVR_PORTS
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
"

if  [ "$PPO_ROLE" = "sampler" ] || [ "$PPO_ROLE" = "gen-rm" ];  then
    MP_ARGS="$MP_ARGS
        --use-tp-pp-dp-mapping \
    "
    python tools/auto_place.py --fn init_ray --config-folder $PLACE_CFG_FOLDER
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
    --min-lr 0. \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend fused \
    --use-flash-attn \
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
    --gdatasetv4-train-metadata-file ${data_path} \
    --gdatasetv4-eval-metadata-file ${eval_data_path} \
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
        --wandb-project $WANDB_PROJECT \
        --wandb-exp-name $WANDB_EXP_NAME \
        --wandb-save-dir wandb \
    "
fi

EVAL_ARGS="
    --ppo-step-eval-interval $PPO_STEP_EVAL_INTERVAL \
    --ppo-eval-steps $PPO_EVAL_STEPS \
    --ppo-eval-rollout-global-batch-size $PPO_EVAL_ROULLOUT_GLOBAL_BATCH_SIZE \
    --ppo-eval-rollout-micro-batch-size $PPO_EVAL_ROULLOUT_MICRO_BATCH_SIZE \
    --ppo-eval-sampling-repeat $PPO_EVAL_SAMPLING_REPEAT \
"

RL_ARGS="
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
    --ppo-max-epochs 2 \
    --ppo-max-epochs-2 2 \
    --ppo-step-save-interval 10 \
    --ppo-step-per-epoch -1 \
    --ppo-rollout-micro-batch-size $ROLLOUT_MICRO_BATCH_SIZE \
    --ppo-rollout-global-batch-size $ROLLOUT_GLOBAL_BATCH_SIZE \
    --ppo-resp-seq-len $RESP_SEQ_LENGTH \
    --ppo-rollout-pad-to-multiple-of 1024 \
    --ppo-logps-fwd-micro-batch-size $PPO_LOGPS_FWD_MICRO_BATCH_SIZE \
    --combine-rm-and-critic-server \
    --ppo-rollout-top-p 0.9 \
    --ppo-rollout-top-k 0 \
    --ppo-rollout-temperature 1.0 \
    --ppo-ratio-eps 0.2 \
    --ppo-clip-ratio-low 0.2 \
    --ppo-clip-ratio-high 0.28 \
    --ppo-rm-mask-prompt \
    --rm-output-scalar 1 \
    --rm-output-sequence 0 \
    --ppo-sampling-keeping-strategy all \
    --ppo-sampling-repeat 32 \
    --ppo-sampling-keep 32 \
    --ppo-use-absolute-kl \
    --use-grpo \
    --grpo-advantage-epsilon 1e-4 \
    --grpo-kl-loss-beta 0 \
    --rm-head-arch multi_layers \
    --ppo-save-first-rollout-data \
    --ppo-grpo-reward-type rm_with_rule \
    --gen-term-at-nan \
    --ppo-rm-reward-alpha 0.5 \
    --ppo-rule-reward-beta 0.5 \
    --grpo-prefetch-samplings \
    --ppo-dynamic-sampling-max-replay $MAX_SAMPLING_RETRIES \
    --update-weight-max-size-mb 512 \
    --sampler-gpu-memory-utilization 0.7 \
    --gen-rm-gpu-memory-utilization 0.7 \
    --ppo-smart-pad-infer \
    --ppo-smart-pad-train \
    --dapo-overlong-penalty \
    --dapo-overlong-buffer-len 256 \
    --dapo-overlong-penalty-factor 1.0 \
    --ppo-dual-clip-ratio-c 10.0
    --ppo-train-dynamic-mbs-target-seq 4096 \
    --ppo-train-dynamic-mbs-limit 16 \
"

if [ "$PPO_ROLE" = "sampler" ]; then
  RL_ARGS="$RL_ARGS
      --no-fused-kernel \
  "
fi

if [ "$PPO_ROLE" = "actor" ]; then
  RL_ARGS="$RL_ARGS
      --ppo-actor-freeze-ppo-steps 0 \
  "
fi

# 热启动的时候去掉 FINETUNE_ARGS args
FINETUNE_ARGS="
    --finetune \
"

ACTOR_NODE_IPS=($ACTOR_NODE_IPS)
MONITOR_ARGS="
  --do-monitor \
  --monitor-server-ip ${ACTOR_NODE_IPS[0]} \
  --monitor-port 62000 \
  --auto-set-finetune-arg \
"

if [ "$PPO_ROLE" = "actor" ]; then
    RUN_PY='./tasks/math_rl_v3/train_ppo_actor.py'
elif [ "$PPO_ROLE" = "critic" ]; then
    RUN_PY='./tasks/math_rl_v3/train_ppo_critic.py'
elif [ "$PPO_ROLE" = "sampler" ]; then
    RUN_PY='./tasks/math_rl_v3/train_ppo_sampler.py'
else
    echo "no ${PPO_ROLE}"
    exit 0
fi

torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $EVAL_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    $FINETUNE_ARGS \
    $MONITOR_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --load-ref $REF_LOAD_CHECKPOINT_DIR