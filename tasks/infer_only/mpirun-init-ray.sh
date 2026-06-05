cp /etc/mpi/hostfile /root/hostfile
# head -n 1 /etc/mpi/hostfile > /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile

ray stop --force
pkill -9 -f python

export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export GCORE_SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=1800

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  -x PYTHONUNBUFFERED -x GCORE_SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S \
  bash tasks/infer_only/init-ray.sh $__HOST_IP__
