cat /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

export _MASTER_ADDR=$__POD_IP__

MLM_INPUT_PATH=/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/tmp/mpirun_ckpt_test_welm
HF_OUTPUT_PATH=/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/tmp/mpirun_ckpt_test_welm_hf
MBRIDGE_PATH=/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/code/3rdparty/welm-deps/mbridge/ 
MCORE_PATH=/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/code/3rdparty/welm-deps/Megatron-LM/
TOKENIZER_PATH=/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/tmp/hf_test_welm

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pip install -U transformers==4.57.5

sleep 10

mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
      -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
      pkill -9 -f python
sleep 3

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
  bash tasks/welm_v4/scripts/mlm_to_hf.sh $MLM_INPUT_PATH $HF_OUTPUT_PATH $MBRIDGE_PATH $MCORE_PATH $TOKENIZER_PATH > log/convert.log 2>&1

