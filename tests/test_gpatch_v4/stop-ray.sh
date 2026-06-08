MIP=$1
PORT=6379

node_rank=$OMPI_COMM_WORLD_RANK
nnodes=$OMPI_COMM_WORLD_SIZE
num_gpus=$((8*nnodes))

ray stop -f

echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
sync
echo 3 >/proc/sys/vm/drop_caches
echo 1 >/proc/sys/vm/compact_memory