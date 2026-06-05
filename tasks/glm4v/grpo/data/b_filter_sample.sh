cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile
export _MASTER_ADDR=${__POD_IP__:-localhost}

DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../../../..
CUR_DIR=$(pwd)
MYWD=${CUR_DIR}



cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile


mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pkill -9 -f python
sleep 3


repo_id="zai-org/GLM-4.5V"
HF_MODEL_PATH="$MYWD/hf-hub/${repo_id}"


# just test
repo_id=hiyouga/geometry3k
save_dir=$MYWD/hf-hub/$repo_id


PROCESSOR_PER_NODE=64
RUN_PY="${MYWD}/tools/filter/glm4v_filter.py"
PY_ARGS="
    --seq-length 4096 \
    --use-grpo \
    --hf-model-path $HF_MODEL_PATH \
    --input-jsonl-files $save_dir/gcore-data/*.jsonl \
    --output-jsonl-dir $save_dir/gcore-data/filter_4k_glm \
"

export PYTHONPATH=${MYWD}/../mbridge:${MYWD}/../Megatron-LM:${MYWD}:${PYTHONPATH}

mpirun -v --allow-run-as-root \
    --bind-to none --map-by slot --hostfile /root/hostfile \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR -x PYTHONPATH \
    bash tools/filter/filter_run.sh \
    $PROCESSOR_PER_NODE \
    $RUN_PY \
    $PY_ARGS