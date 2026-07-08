
# cp /etc/mpi/hostfile /root/hostfile
if [ -z "${GCORE_GPU}" ]; then
    cp /etc/mpi/hostfile /root/hostfile
else
    head -n $GCORE_GPU /etc/mpi/hostfile > /root/hostfile
fi
sed -i 's/slots=8/slots=1/g' /root/hostfile

ray stop --force
pkill -9 -f python

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  pkill -f -9 EngineCore

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  pkill -f -9 occupy

sleep 3

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  bash tasks/math_dsv4/scripts/prepare_env.sh
sleep 3

export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export VLLM_LOG_STATS_INTERVAL=3

export VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}
if [ "${VLLM_ENABLE_CUDA_COMPATIBILITY}" = "1" ]; then
    export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.8/compat"
fi
export GPATCH_EXTRA_PROPAGATE_ENV=VLLM_LOG_STATS_INTERVAL,VLLM_ENABLE_CUDA_COMPATIBILITY,VLLM_CUDA_COMPATIBILITY_PATH

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  -x WANDB_BASE_URL -x WANDB_API_KEY -x PYTHONUNBUFFERED \
  -x PYTORCH_CUDA_ALLOC_CONF -x CUDA_DEVICE_MAX_CONNECTIONS \
  -x GPATCH_EXTRA_PROPAGATE_ENV \
  bash tasks/math_rl_v4/scripts/init-ray.sh $__HOST_IP__
