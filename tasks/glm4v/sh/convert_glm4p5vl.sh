DIR="$(cd "$( dirname "$0" )" && pwd)"
cd ${DIR}/../../..
CUR_DIR=$(pwd)


readonly MCORE_PATH='../Megatron-LM'
export PYTHONPATH="${CUR_DIR}:$MCORE_PATH:../mbridge:$PYTHONPATH"


HF_HUB=${CUR_DIR}/hf-hub/zai-org/GLM-4.5V
MLM_PATH=${CUR_DIR}/ckpt_glm4p5vl_save_sft/iter_0000500
OUTPUT_PATH=${CUR_DIR}/ckpt_glm4p5vl_save_sft_hf
mkdir ${OUTPUT_PATH}

bash ${CUR_DIR}/tasks/glm4v/sh/convert_from_bridge.sh mlm_to_hf $MLM_PATH $OUTPUT_PATH $HF_HUB

