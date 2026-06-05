set -ex

MYWD=$PWD


repo_id=BUAADreamer/llava-en-zh-300k
save_dir=$MYWD/hf-hub/$repo_id
# to jsonl and lmdb
# only using the zh dataset
mkdir -p $save_dir/gcore-data/zh
python tools/data_convert/convert_llava.py \
    --input_path $save_dir/zh \
    --output_path $save_dir/gcore-data/zh
# build the data index
mpirun -v --allow-run-as-root \
    --hostfile /etc/mpi/hostfile --oversubscribe -np 32 \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
    python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $save_dir/gcore-data/zh \
    --data_file_postfix 'jsonl' \
    --ensure_each_line_forms_valid_json
#---------------------------------


repo_id=yusuf802/captcha_dataset
save_dir=$MYWD/hf-hub/$repo_id
# to jsonl and lmdb
python tools/data_convert/convert_captcha.py \
    --input_path $save_dir/data \
    --output_path $save_dir/gcore-data
# build the data index
mpirun -v --allow-run-as-root \
    --hostfile /etc/mpi/hostfile --oversubscribe -np 32 \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
    python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder $save_dir/gcore-data \
    --data_file_postfix 'jsonl' \
    --ensure_each_line_forms_valid_json
#---------------------------------


repo_id=llamafactory/RLHF-V
save_dir=$MYWD/hf-hub/$repo_id
if [ ! -d "$save_dir/gcore-data" ]; then
    # to jsonl and lmdb
    python tools/data_convert/convert_rlhf_v.py \
        --input_path $save_dir \
        --output_path $save_dir/gcore-data
    # build the data index
    mpirun -v --allow-run-as-root \
        --hostfile /etc/mpi/hostfile --oversubscribe -np 32 \
        --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 \
        -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
        python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
        --data_folder $save_dir/gcore-data \
        --data_file_postfix 'jsonl' \
        --ensure_each_line_forms_valid_json
fi
