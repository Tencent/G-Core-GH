ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1
DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../..
CUR_DIR=$(pwd)

path=$1

export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM==1
readonly MLM_PATH=$CUR_DIR/../Megatron-LM
# readonly MBRIDGE_PATH=$CUR_DIR/../mbridge
# readonly LLaMAFactory_Path=$CUR_DIR/../LLaMA-Factory/src
readonly EnergonPath=$CUR_DIR/../Megatron-Energon/src
export PYTHONPATH="$MLM_PATH:$PYTHONPATH:${EnergonPath}:${CUR_DIR}"

python gdataset/data/tools/prepare_energon.py  --path $path