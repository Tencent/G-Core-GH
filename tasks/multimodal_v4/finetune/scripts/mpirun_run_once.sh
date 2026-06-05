cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

RUN_ONCE_SH=$1

export _MASTER_ADDR=${__POD_IP__:-localhost}
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

sleep 3
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  bash tasks/multimodal_v4/finetune/scripts/prepare_env.sh

sleep 3
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS -x _MASTER_ADDR \
  -x WANDB_BASE_URL -x WANDB_API_KEY -x PYTHONUNBUFFERED -x PYTORCH_CUDA_ALLOC_CONF -x CUDA_DEVICE_MAX_CONNECTIONS \
  bash $1 $__HOST_IP__ ${@:2}
