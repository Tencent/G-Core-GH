
#!/bin/bash

cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

MLM_INPUT_PATH=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_marcoszhu/code_qwen3_next_sft_gcore/gcore-dev/tasks/math_rl_v3/qwen3/moe/next/output/train_multi_target_sft_from_qwen3_next_lr8e-6_bsz256_epoch2
HF_OUTPUT_PATH=./qwen-next-hf

mpirun -v --allow-run-as-root \
    --bind-to none --map-by slot --hostfile /root/hostfile \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
    pkill -9 -f python

sleep 3

mkdir -p log

mpirun -v --allow-run-as-root \
    --bind-to none --map-by slot --hostfile /root/hostfile \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
    bash tasks/math_rl_v3/qwen3/moe/next/mlm_to_hf.sh $MLM_INPUT_PATH $HF_OUTPUT_PATH >log/mlm_to_hf.log 2>&1 &


wait

cd /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_7/user_luckytyang/cephnj2/gpu_cal && nohup bash mpirun_cal.sh &
