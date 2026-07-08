MIP=$1
PORT=6379

node_rank=$OMPI_COMM_WORLD_RANK
nnodes=$OMPI_COMM_WORLD_SIZE
num_gpus_per_node=8
num_gpus=$((num_gpus_per_node*nnodes))


if [[ "$node_rank" == '0' ]]; then
  ray start --head --node-ip-address $MIP --port=$PORT --num-gpus=$num_gpus_per_node
else
  ray start --address="$MIP:$PORT" --num-gpus=$num_gpus_per_node
fi
