# 关闭透明大页，并清空 page cache（tlinux 特殊姿势）。
# echo never > /sys/kernel/mm/transparent_hugepage/enabled
# echo never > /sys/kernel/mm/transparent_hugepage/defrag
# sync
# echo 3 >/proc/sys/vm/drop_caches
# echo 1 >/proc/sys/vm/compact_memory

MIP=$1

PORT=6379

node_rank=$OMPI_COMM_WORLD_RANK
nnodes=$OMPI_COMM_WORLD_SIZE
num_gpus=$((8*nnodes))

if [[ "$node_rank" == '0' ]]; then
  ray start --head --node-ip-address $MIP --port=$PORT
else
  ray start --address="$MIP:$PORT"
fi
