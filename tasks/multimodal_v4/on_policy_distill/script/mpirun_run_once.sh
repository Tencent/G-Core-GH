cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

RUN_ONCE_SH=$1

export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  -x WANDB_BASE_URL -x WANDB_API_KEY -x PYTHONUNBUFFERED \
  bash $1 $__HOST_IP__ ${@:2}
