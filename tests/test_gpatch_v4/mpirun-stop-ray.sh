if [ -z "${GCORE_GPU}" ]; then
    cp /etc/mpi/hostfile /root/hostfile
else
    head -n $GCORE_GPU /etc/mpi/hostfile > /root/hostfile
fi
sed -i 's/slots=8/slots=1/g' /root/hostfile

export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1
export _CHECK_PEFT=0

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile /root/hostfile \
  --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
  -x RAY_DEDUP_LOGS -x PYTHONUNBUFFERED -x _CHECK_PEFT \
  bash tests/test_gpatch_v4/stop-ray.sh $__HOST_IP__
