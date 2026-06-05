# DeepEP env
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond1
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1

export NCCL_IB_TC=160
export NVSHMEM_IB_TRAFFIC_CLASS=160

# distributed args
readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
export WORLD_SIZE=${NNODES}
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"
export MASTER_PORT=65535
export RANK=${NODE_RANK}

echo "ws ${WORLD_SIZE} ma ${MASTER_ADDR} mp ${MASTER_PORT} r ${RANK}"
# python tasks/deepseek/DeepEP/test_internode.py >log/deepep-${NODE_RANK}.log 2>&1 &

python tasks/deepseek/DeepEP/test_low_latency.py > log/deepep-${NODE_RANK}.log 2>&1