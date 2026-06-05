#!/bin/bash

# cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

export _MASTER_ADDR=${__POD_IP__:-localhost}

MYWD=$PWD
readonly TOKENIZER_PATH="$MYWD/hf-hub/Qwen/Qwen2.5-VL-3B-Instruct"

readonly PROCESSOR_PER_NODE=64
readonly RUN_PY="tools/filter/qwen2vl_filter.py"
readonly PY_ARGS="
  --seq-length 4096 \
  --tokenizer-model $TOKENIZER_PATH \
  --processor-path $TOKENIZER_PATH \
  --input-jsonl-files $MYWD/hf-hub/RadGenome/PMC-VQA/gcore-data/*.jsonl \
  --output-jsonl-dir $MYWD/hf-hub/RadGenome/PMC-VQA/gcore-data/filter_4k \
  --use-grpo \
"

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
  bash tools/filter/filter_run.sh \
  $PROCESSOR_PER_NODE \
  $RUN_PY \
  $PY_ARGS \
