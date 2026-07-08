POD_NAME=$POD_NAME

mkdir -p inspct_memory
nvidia-smi > inspct_memory/$POD_NAME.log

# mpirun -v --allow-run-as-root --bind-to none --map-by slot --hostfile /etc/mpi/hostfile --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x RAY_DEDUP_LOGS bash tasks/math_dsv4/scripts/inspect_cuda_memory.sh