readonly MCORE_PATH='../3rdparty//Megatron-LM:../Megatron-LM'
export PYTHONPATH="$PWD:$MCORE_PATH:../mbridge:$PYTHONPATH"

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# readonly DFS_FOLDER="/mnt/gemininjceph2/geminicephfs/mm-base-plt2"
readonly DFS_FOLDER="/mnt/ceph-hz1-csp/mm-base-plt2/"
# readonly DFS_FOLDER='/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7'
readonly TOKENIZER_MODEL="$DFS_FOLDER/nrwu/hf-hub/deepseek-ai/DeepSeek-V3"

readonly TP_SIZE=8
readonly PP_SIZE=4
readonly EP_SIZE=2
readonly CP_SIZE=1

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

readonly convert_way="mlm_to_hf"
# readonly convert_way="hf_to_mlm"

if [ $convert_way == "hf_to_mlm" ]; then
    HF_INPUT_DIR="${DFS_FOLDER}/nrwu/hf-hub/deepseek-ai/DeepSeek-V3"
    MLM_OUTPUT_DIR="$PWD/deepseek-v3-mlm-ckpt"

    CKPT_ARGS="
        --convert_way hf_to_mlm \
        --hf_dir $HF_INPUT_DIR \
        --load_model_path $HF_INPUT_DIR \
        --save_model_path $MLM_OUTPUT_DIR \
    "
elif [ $convert_way == "mlm_to_hf" ]; then
    # MLM_INPUT_DIR="/mnt/ceph-sg1-csp/mmsearch-luban-universal/group_7/user_lesliejiang/cephnj2/RM/short_qa/ds-v3-sft/Megatron-LM/tasks/ai_search/deepseek-v3-sft-bf16/iter_0000060"
    # HF_DIR="${DFS_FOLDER}/nrwu/hf-hub/deepseek-ai/DeepSeek-V3"
    # HF_OUTPUT_DIR="/mnt/ceph-sg1-csp/mmsearch-luban-universal/group_7/user_jeffhong/ai-search/hf_ckpt/lesliejiang-iter_0000060"

    MLM_INPUT_DIR="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_lesliejiang/cephnj2/RM/short_qa/ds-v3-sft/Megatron-LM/tasks/ai_search/deepseek-v3-sft-bf16/iter_0000060"
    HF_DIR="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_jeffhong/hf-hub/deepseek-ai/DeepSeek-V3"
    HF_OUTPUT_DIR="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_jeffhong/ai-search/hf_ckpt/lesliejiang-iter_0000060"

    CKPT_ARGS="
        --convert_way mlm_to_hf \
        --hf_dir $HF_DIR \
        --load_model_path $MLM_INPUT_DIR \
        --save_model_path $HF_OUTPUT_DIR \
        --override_args_from_ckpt \
        --remove_fp8 \
        --remove_mtp \
    "
fi

  

# export NCCL_DEBUG=INFO
torchrun $DISTRIBUTED_ARGS tools/px_ckpt_conv/convert_from_bridge.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --cp $CP_SIZE \
    --ep $EP_SIZE \
    --vpp 1 \
    --num_layers_in_first_pipeline_stage 14 \
    --num_layers_in_last_pipeline_stage 15 \
    --dist-ckpt-format torch_dist \
    $CKPT_ARGS >log/convert2-$convert_way-${NODE_RANK}.log 2>&1 &