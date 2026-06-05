export PYTHONPATH="$PWD:$PYTHONPATH"

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=40000
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

echo "INFO
NODE_RANK $NODE_RANK
NNODES $NNODES
"

readonly MASTER="${MASTER_ADDR}:${MASTER_PORT}"
echo "${MASTER}"

# MODEL_PATH="/mnt/ceph-sh2/user_jeffhong/hf-hub/deepseek-ai/DeepSeek-V3"
# MODEL_PATH="/mnt/ceph-sh2/user_jeffhong/hf-hub/deepseek-ai/DeepSeek-V3-fp16"
MODEL_PATH="/mnt/ceph-sh2/user_jeffhong/workspace/wepsdl-dev/gcore-dev/deepseek-v3-hf-ckpt"

export PATH="/opt/rh/gcc-toolset-12/root/usr/bin:$PATH"
export SGL_ENABLE_JIT_DEEPGEMM=1
# --speculative-algorithm EAGLE \

python -m sglang.launch_server --model-path ${MODEL_PATH} \
    --tp 32 --dist-init-addr ${MASTER} --nnodes ${NNODES} --node-rank ${NODE_RANK} --trust-remote-code >log/sglang-${NODE_RANK}.log 2>&1 &