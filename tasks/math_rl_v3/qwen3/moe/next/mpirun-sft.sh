#!/bin/bash

# head -n 4 /etc/mpi/hostfile >/root/hostfile
cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

export _MASTER_ADDR=$__POD_IP__
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
  pkill -9 -f python 

sleep 2

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
  bash tasks/math_rl_v3/qwen3/moe/next/sft.sh >log/sft.log 2>&1 &

wait

# cd /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_luckytyang/cephnj2/gpu_cal && nohup bash mpirun_cal.sh &
cd /mnt/geminihzceph1/geminicephfs/mmsearch-luban-universal/group_7/user_oriontian/gpu_cal/ & nohup sh mpirun_cal.sh &