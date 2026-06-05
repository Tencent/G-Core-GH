MYWD=$PWD
repo_id=RadGenome/PMC-VQA
save_dir=$MYWD/new-hf-hub/$repo_id
# convert to gcore support dataset format
unzip $save_dir/images_2.zip -d $save_dir
# to jsonl and lmdb
python tools/data_convert/convert_pmc_vqa.py \
    --csv_input $save_dir/train_2.csv $save_dir/test_2.csv \
    --image_dir $save_dir/figures \
    --output_dir $save_dir/gcore-data

cp /etc/mpi/hostfile /root/hostfile
sed -i 's/slots=8/slots=1/g' /root/hostfile
export _MASTER_ADDR=${__POD_IP__:-localhost}

# filter for gdatasetv4
TOKENIZER_PATH="$MYWD/hf-hub/Qwen/Qwen3-VL-4B-Instruct"

PROCESSOR_PER_NODE=64
RUN_PY="tools/filter/qwen2vl_filter.py"
PY_ARGS="
    --seq-length 4096 \
    --tokenizer-model $TOKENIZER_PATH \
    --processor-path $TOKENIZER_PATH \
    --model-arch qwen3_vl \
    --input-jsonl-files $save_dir/gcore-data/*.jsonl \
    --output-jsonl-dir $save_dir/gcore-data/filter_4k_qwen3vl \
"

mpirun -v --allow-run-as-root \
    --bind-to none --map-by slot --hostfile /root/hostfile \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 --mca routed direct \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
    bash tools/filter/filter_run.sh \
    $PROCESSOR_PER_NODE \
    $RUN_PY \
    $PY_ARGS

# remove unuse file
rm -r $save_dir/figures
