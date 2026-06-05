MYWD=$PWD
readonly LMDB_PORT=8312
readonly DATASET_ROOT="${MYWD}/hf-hub/RadGenome/PMC-VQA/gcore-data"
readonly LMDB_PATH="${DATASET_ROOT}/img_file.lmdb"
readonly DATASET_META="/tmp/filter_4k_pmc_vqa_gdataset_v4.json"

readonly IP=$1
readonly DATASET_NAME=$2
echo $DATASET_NAME

python3 tools/data_convert/build_dataset_v4_meta.py \
    --name "PMC-VQA" \
    --description "the dataset from RadGenome/PMC-VQA" \
    --lmdb_port $LMDB_PORT \
    --output_fullpath $DATASET_META \
    --json_inputs $DATASET_ROOT/$DATASET_NAME/train_2.csv.jsonl \
    --rebuild

readonly EVAL_DATASET_META="/tmp/filter_4k_pmc_vqa_gdataset_v4_eval.json"
python3 tools/data_convert/build_dataset_v4_meta.py \
    --name "PMC-VQA" \
    --description "the dataset from RadGenome/PMC-VQA" \
    --lmdb_port $LMDB_PORT \
    --output_fullpath $EVAL_DATASET_META \
    --json_inputs $DATASET_ROOT/$DATASET_NAME/test_2.csv.jsonl \
    --rebuild

pkill -f -9 lmdb_read_svr.py
nohup python3 megatron_datasets/tools/lmdb_read_svr.py \
    --lmdb-path $LMDB_PATH \
    --lmdb-map-size 500 \
    --lmdb-port $LMDB_PORT > svr.log 2>&1 &
