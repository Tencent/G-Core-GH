MIP=$1
PORT=6379

node_rank=$OMPI_COMM_WORLD_RANK
nnodes=$OMPI_COMM_WORLD_SIZE
num_gpus=$((8*nnodes))

pip3 install lipsum pytest-timeout

if [[ "$node_rank" == '0' ]]; then
  ray start --head --node-ip-address $MIP --port=$PORT
else
  ray start --address="$MIP:$PORT"
fi
