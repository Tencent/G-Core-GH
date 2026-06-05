#!/bin/bash
#
# ```
readonly MPI_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly MPI_SIZE="${OMPI_COMM_WORLD_SIZE:-1}"

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="bond1"
export GLOO_SOCKET_IFNAME="bond1"
export HF_DATASETS_OFFLINE=1
export NVTE_FUSED_ATTN=0

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0

readonly DIST_TIMEOUT_MIN=300

readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly GPUS_PER_NODE=${GPUS_PER_NODE:-8}
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export TOPO_CONFIG_FOLDER=$1
export PPO_ROLE=$2

SAMPLER_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-nnodes`
SAMPLER_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-master-addr`
SAMPLER_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-svr-ips`
SAMPLER_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn sampler-svr-ports`
CRITIC_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-nnodes`
CRITIC_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-master-addr`
CRITIC_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-svr-ips`
CRITIC_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn critic-svr-ports`
ACTOR_NNODES=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-nnodes`
ACTOR_MASTER_ADDR=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-master-addr`
ACTOR_SVR_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-svr-ips`
ACTOR_SVR_PORTS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-svr-ports`
ACTOR_NODE_IPS=`python tools/auto_place.py --fn get --config-folder $TOPO_CONFIG_FOLDER --get-fn actor-node-ips`


# only actor
if [ "$PPO_ROLE" = "actor" ]; then
  export MASTER_ADDR="${ACTOR_MASTER_ADDR:-localhost}"
  readonly MASTER_PORT=65531
elif [ "$PPO_ROLE" = "critic" ]; then
  export MASTER_ADDR="${CRITIC_MASTER_ADDR:-localhost}"
  readonly MASTER_PORT=65534
elif [ "$PPO_ROLE" = "sampler" ]; then
  export MASTER_ADDR="${SAMPLER_MASTER_ADDR:-localhost}"
  readonly MASTER_PORT=65535
else
  echo "$PPO_ROLE no support"
  exit 0
fi

export WANDB_API_KEY=
export WANDB_BASE_URL=
readonly USERNAME="xtl_test"
readonly GRPO_RUN="test_ep4_rm_with_rule_rm_ep8"

readonly DFS_PATH="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly ACTOR_TOKENIZER_MODEL="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-30B-A3B"
readonly RM_TOKENIZER_MODELS="$DFS_PATH/nrwu/hf-hub/Qwen/Qwen3-30B-A3B"
readonly DATA_PATH="tasks/math_rl_v3/qwen/rl_metadata_nj.json"

# 对每个角色设置加载路径
if [ "$PPO_ROLE" = "actor" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen3-30b-a3b-moe.yaml"
  readonly REF_LOAD_CHECKPOINT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm_ref"
  readonly LOAD_CHECKPOINT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm"
  readonly SAVE_CHECKPOINT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm_save"
  readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
  readonly TB_DIR="tb/dqa-ppo-actor"
  readonly WANDB_DIR="wandb_local/dqa-ppo-actor"
  readonly LR=5e-7
elif [ "$PPO_ROLE" = "critic" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen3-30b-a3b-moe-rm.yaml"
  readonly LOAD_CHECKPOINT_DIR="${PWD}/qwen3_30b_a3b_moe_mlm_random_rm_head"
  readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR
  readonly SAVE_CHECKPOINT_DIR="none"
  readonly TOKENIZER_MODEL=$RM_TOKENIZER_MODELS
  readonly TB_DIR="tb/dqa-ppo-critic"
  readonly WANDB_DIR="wandb_local/dqa-ppo-critic"
  readonly LR=5e-7
elif [ "$PPO_ROLE" = "sampler" ]; then
  readonly MODEL_YAML="gpatch/model_yamls/qwen3-30b-a3b-moe.yaml"
  readonly LOAD_CHECKPOINT_DIR="${DFS_PATH}/nrwu/hf-hub/Qwen/Qwen3-30B-A3B/"
  readonly REF_LOAD_CHECKPOINT_DIR=$LOAD_CHECKPOINT_DIR
  readonly SAVE_CHECKPOINT_DIR="none"
  readonly TOKENIZER_MODEL=$ACTOR_TOKENIZER_MODEL
  readonly TB_DIR="tb/dqa-ppo-sampler"
  readonly WANDB_DIR="wandb_local/dqa-ppo-sampler"
  readonly LR=5e-7
else
  echo "$PPO_ROLE no support"
  exit 0
fi

# 确定 topo
readonly CRITIC_TP_SIZE=1
readonly CRITIC_PP_SIZE=4
readonly CRITIC_EP_SIZE=8
readonly CRITIC_DP_SIZE=$(($GPUS_PER_NODE*$CRITIC_NNODES/$CRITIC_TP_SIZE/$CRITIC_PP_SIZE))

readonly SAMPLER_TP_SIZE=4
readonly SAMPLER_PP_SIZE=1
readonly SAMPLER_DP_SIZE=$(($GPUS_PER_NODE*$SAMPLER_NNODES/$SAMPLER_TP_SIZE/$SAMPLER_PP_SIZE))
readonly SAMPLER_MP_SIZE=$(($SAMPLER_TP_SIZE*$SAMPLER_PP_SIZE))

readonly ACTOR_TP_SIZE=1
readonly ACTOR_PP_SIZE=4
readonly ACTOR_CP_SIZE=1
readonly ACTOR_EP_SIZE=4
readonly ACTOR_DP_SIZE=$(($GPUS_PER_NODE*$ACTOR_NNODES/$ACTOR_TP_SIZE/$ACTOR_PP_SIZE/$ACTOR_CP_SIZE))

if [ "$PPO_ROLE" = "actor" ]; then
  readonly TP_SIZE=$ACTOR_TP_SIZE
  readonly PP_SIZE=$ACTOR_PP_SIZE
  readonly EP_SIZE=$ACTOR_EP_SIZE
  readonly CP_SIZE=$ACTOR_CP_SIZE
  readonly DP_SIZE=$ACTOR_DP_SIZE
elif [ "$PPO_ROLE" = "critic" ]; then
  readonly TP_SIZE=$CRITIC_TP_SIZE
  readonly PP_SIZE=$CRITIC_PP_SIZE
  readonly EP_SIZE=$CRITIC_EP_SIZE
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

readonly PPO_LOGPS_FWD_MICRO_BATCH_SIZE=4
readonly TRAIN_ITERS=-1
readonly EVAL_ITERS=0
readonly SEQ_LENGTH=$((4*1024))
readonly RESP_SEQ_LENGTH=$((1024))
readonly MAX_SAMPLING_RETRIES=8
# readonly RM_RESP_SEQ_LENGTH=512

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
    --expert-model-parallel-size $EP_SIZE \
    --use-distributed-optimizer \
"

EXPERT_ARGS="
    --moe-grouped-gemm \
    --moe-token-dispatcher-type allgather \
"

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
    --min-lr 5e-9 \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --optimizer adam \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --attention-backend flash \
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
    --no-save-rng \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
    --wandb-project $USERNAME \
    --wandb-exp-name $GRPO_RUN \
    --wandb-save-dir  $WANDB_DIR \
"
# 上报参数
    # --wandb-project $USERNAME \
    # --wandb-exp-name $GRPO_RUN \
    # --wandb-save-dir  $WANDB_DIR \

GEN_ARGS="
    --ppo-sort-prompts-across-batches 8 \
    --ppo-rollout-max-prompt-len-diff 128 \
"

# 注意 --ppo-rollout-temperature 设置过大会导致产生中文乱码
RL_ARGS="$GEN_ARGS
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
    --ppo-sampler-tensor-model-parallel-size $SAMPLER_TP_SIZE \
    --ppo-sampler-pipeline-model-parallel-size $SAMPLER_PP_SIZE \
    --ppo-sampler-data-parallel-size $SAMPLER_DP_SIZE \
    --sampler-mp-size $SAMPLER_MP_SIZE \
    --ppo-step-update-sampler-interval 2 \
    --ppo-max-epochs 1 \
    --ppo-max-epochs-2 4 \
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
    --ppo-rollout-temperature 0.7 \
    --ppo-ratio-eps 0.2 \
    --ppo-rm-mask-prompt \
    --rm-output-scalar 1 \
    --rm-output-sequence 0 \
    --ppo-sampling-keeping-strategy all \
    --ppo-sampling-repeat 4 \
    --ppo-sampling-keep 4 \
    --ppo-use-absolute-kl \
    --use-grpo \
    --grpo-advantage-epsilon 1e-4 \
    --grpo-kl-loss-beta 1e-2 \
    --ppo-save-first-rollout-data \
    --ppo-grpo-reward-type rm_with_rule \
    --gen-term-at-nan \
    --ppo-rm-reward-alpha 0.5 \
    --ppo-rule-reward-beta 0.5 \
    --grpo-prefetch-samplings \
    --ppo-dynamic-sampling-max-replay $MAX_SAMPLING_RETRIES \
"

if [ "$PPO_ROLE" = "actor" ]; then
  RL_ARGS="$RL_ARGS
      --ppo-actor-freeze-ppo-steps 0 \
  "
fi

# 热启动的时候去掉 FINETUNE_ARGS args
FINETUNE_ARGS="
    --finetune \
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

export VLLM_USE_V1=1

torchrun $DISTRIBUTED_ARGS $RUN_PY \
    $MP_ARGS \
    $EXPERT_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    $FINETUNE_ARGS \
    --cli-arg-yaml-cfgs $MODEL_YAML \
    --distributed-backend nccl \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --load-ref $REF_LOAD_CHECKPOINT_DIR
