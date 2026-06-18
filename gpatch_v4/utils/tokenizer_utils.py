from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.models.deepseek_v4.chat_template import DSV4_CHAT_TEMPLATE

TOKENIZER_TEMPLATE = {
    MODEL_ARCH.DEEPSEEK_V4: DSV4_CHAT_TEMPLATE,
}


def get_tokenizer_template(model_arch):
    if model_arch not in TOKENIZER_TEMPLATE:
        return None
    return TOKENIZER_TEMPLATE[model_arch]
