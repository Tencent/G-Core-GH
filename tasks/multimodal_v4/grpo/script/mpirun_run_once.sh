cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

# mpirun 第一次要不会跑 laucher
mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  hostname

RUN_ONCE_SH=$1

ray stop --force
pkill -9 -f python

export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  -x WANDB_BASE_URL -x WANDB_API_KEY -x PYTHONUNBUFFERED \
  -x PYTORCH_CUDA_ALLOC_CONF -x CUDA_DEVICE_MAX_CONNECTIONS \
  -x SGLANG_EMPTY_CACHE_INTERVAL -x SGLANG_RETURN_ORIGINAL_LOGPROB \
  -x GPATCH_EXTRA_PROPAGATE_ENV \
  bash $1 $__HOST_IP__ ${@:2}
