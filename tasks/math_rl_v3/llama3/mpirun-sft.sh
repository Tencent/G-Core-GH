#!/bin/bash

head -n 1 /etc/mpi/hostfile >/root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

export _MASTER_ADDR=$__POD_IP__

LOG_DIR=log/sft
mkdir -p $LOG_DIR

# RUN_PATH=tasks/math_rl_v3/llama3/sft.sh
RUN_PATH=tasks/math_rl_v3/llama3/align_llama3_loss.sh

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
  bash $RUN_PATH > $LOG_DIR/$(date '+%Y%m%d-%H%M%S').log 2>&1 &
