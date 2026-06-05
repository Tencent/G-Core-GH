export PYTHONPATH="$PWD:$PYTHONPATH"

DFS_PATH="/mnt/ceph-sz2-csp/mm-base-plt2"
data_path="$DFS_PATH/user_xiaotaoliu/data/small-dpo-data/"


# mpirun -v --allow-run-as-root \
#     --bind-to none --map-by slot --hostfile /etc/mpi/hostfile \
#     --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
#     -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
#     python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
#     --data_folder $data_path \
#     --data_file_postfix 'jsonl' \
#     --domain_name 'dpo-data' \


data_path="$PWD/data/dpo_data_with_ref_dense/dqa-dpo/"
mpirun -v --allow-run-as-root \
    --bind-to none --map-by slot -np 8 \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
    python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $data_path \
    --data_file_postfix 'jsonl' \
    --domain_name 'dpo-data' \
