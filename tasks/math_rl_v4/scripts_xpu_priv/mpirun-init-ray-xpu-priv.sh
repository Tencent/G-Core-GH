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
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  ray stop --force

sleep 5

export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export VLLM_LOG_STATS_INTERVAL=3
export GPATCH_EXTRA_PROPAGATE_ENV=VLLM_LOG_STATS_INTERVAL

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  hostname

# mpirun -v --allow-run-as-root \
#   --bind-to none --map-by slot --hostfile /root/hostfile \
#   --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
#   -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
#   bash tasks/math_rl_v4/scripts_xpu_priv/setup_rdma_routes_xpu_priv.sh

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS \
  -x WANDB_BASE_URL -x WANDB_API_KEY -x PYTHONUNBUFFERED \
  -x PYTORCH_CUDA_ALLOC_CONF -x CUDA_DEVICE_MAX_CONNECTIONS \
  -x GPATCH_EXTRA_PROPAGATE_ENV \
  bash -c "source tasks/math_rl_v4/scripts_xpu_priv/xpu_env_priv.sh && bash tasks/math_rl_v4/scripts/init-ray.sh $__HOST_IP__"

# # 先测试单机
# pip uninstall click --y && pip install click==8.3.1
# pip uninstall ray --y && pip install ray==2.53.0
# pip install nvidia-modelopt --no-deps
# pip install pulp
# pip install tensorboard --no-deps
# pip install absl-py

# ray start --head --node-ip-address $__HOST_IP__ --port=$6379 --num-gpus=8
