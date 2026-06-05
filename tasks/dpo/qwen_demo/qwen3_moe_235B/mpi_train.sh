#!/bin/bash

FILE_PATH=/root/hostfile
head -n $2 /etc/mpi/hostfile > $FILE_PATH
sed -i 's/slots=8/slots=1/g' $FILE_PATH

export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1

if [ -f "$FILE_PATH" ]; then
  export _MASTER_ADDR=$__POD_IP__
  GEMINI_MPI_ARGS="--bind-to none --map-by slot --hostfile $FILE_PATH --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct"
else
  export _MASTER_ADDR="127.0.0.1"
  GEMINI_MPI_ARGS="--bind-to none --map-by slot --np 1"
fi

mpirun -v --allow-run-as-root \
  $GEMINI_MPI_ARGS \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x WANDB_API_KEY -x WANDB_BASE_URL \
  pkill -9 -f python

sleep 5

RUN_SHELL=$1
mpirun -v --allow-run-as-root \
  $GEMINI_MPI_ARGS \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x WANDB_API_KEY -x WANDB_BASE_URL \
  bash $RUN_SHELL


cd ../../gpu_protect
bash occupy.sh
