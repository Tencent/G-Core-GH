from functools import partial

from gpatch_v4.core.constants import MODEL_ARCH

from .base import BaseRopeIndexHelper


def bind_vision_position_ids_if_available(helper, hf_class):
    import transformers

    helper.transformers_version = transformers.__version__
    helper.has_vision_position_ids = hasattr(hf_class, "get_vision_position_ids")
    if helper.has_vision_position_ids:
        helper.get_vision_position_ids = partial(hf_class.get_vision_position_ids, helper)


class Glm4VRopeIndexHelper(BaseRopeIndexHelper):
    def __init__(self, config):
        from transformers import Glm4vModel
        super().__init__(config, Glm4vModel)


class Qwen3VlRopeIndexHelper(BaseRopeIndexHelper):
    def __init__(self, config):
        from transformers import Qwen3VLModel

        super().__init__(config, Qwen3VLModel)
        bind_vision_position_ids_if_available(self, Qwen3VLModel)


class Qwen3_5RopeIndexHelper(BaseRopeIndexHelper):
    def __init__(self, config):
        from transformers import Qwen3_5Model

        super().__init__(config, Qwen3_5Model)
        bind_vision_position_ids_if_available(self, Qwen3_5Model)


class Qwen3_5_Moe_RopeIndexHelper(BaseRopeIndexHelper):
    def __init__(self, config):
        from transformers import Qwen3_5MoeModel

        super().__init__(config, Qwen3_5MoeModel)
        bind_vision_position_ids_if_available(self, Qwen3_5MoeModel)


class Qwen3_Omni_Moe_RopeIndexHelper(BaseRopeIndexHelper):
    def __init__(self, config):
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
        super().__init__(config.thinker_config, Qwen3OmniMoeThinkerForConditionalGeneration)
        self.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.get_llm_pos_ids_for_vision = partial(self.hf_class.get_llm_pos_ids_for_vision, self)


ROPE_INDEX_HELPERS = {}


def register_index_helper(arch, cls):
    global ROPE_INDEX_HELPERS
    ROPE_INDEX_HELPERS[arch] = cls


def get_index_helper(arch, config_path):
    global ROPE_INDEX_HELPERS
    assert arch in ROPE_INDEX_HELPERS
    cls = ROPE_INDEX_HELPERS[arch]
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(config_path)
    return cls(config)


register_index_helper(MODEL_ARCH.GLM4V, Glm4VRopeIndexHelper)
register_index_helper(MODEL_ARCH.QWEN3_VL, Qwen3VlRopeIndexHelper)
register_index_helper(MODEL_ARCH.QWEN3_VL_MOE, Qwen3VlRopeIndexHelper)
register_index_helper(MODEL_ARCH.QWEN3_5, Qwen3_5RopeIndexHelper)
register_index_helper(MODEL_ARCH.QWEN3_5_MOE, Qwen3_5_Moe_RopeIndexHelper)
register_index_helper(MODEL_ARCH.QWEN3_OMNI_MOE, Qwen3_Omni_Moe_RopeIndexHelper)
register_index_helper(MODEL_ARCH.WEMM3_EMBEDDING, Qwen3VlRopeIndexHelper)
