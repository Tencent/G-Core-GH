from gpatch_v4.configs.config import (
    AgenticRlConfig,
    DpoConfig,
    FinetuneConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RewardConfig,
    RlConfig,
)
from gpatch_v4.core.constants import MODEL_ARCH

try:
    from gpatch_v4.extended_model.deepseek_v4 import (
        DeepseekV4DpoPrepareDataForwardLLM,
        DeepseekV4PrepareDataForwardLLM,
    )
except ImportError:
    DeepseekV4DpoPrepareDataForwardLLM = None
    DeepseekV4PrepareDataForwardLLM = None
from gpatch_v4.extended_model.gemma4 import Gemma4PrepareDataForward
from gpatch_v4.extended_model.llm import (
    DpoPrepareDataForwardLLM,
    OffPoilicyDistillPrepareDataForwardLLM,
    PrepareDataForwardLLM,
    SamplerGenerateFuncLLM,
)
from gpatch_v4.extended_model.multi_modal import (
    ApplySamplingRolloutAttrMultiModal,
    ApplySamplingRolloutAttrQwen3_5,
    ApplySamplingRolloutAttrQwen3_5_MOE,
    SamplerGenerateFuncMultiModal,
)
from gpatch_v4.extended_model.qwen3_vl import (
    Qwen3VLDpoPrepareDataForward,
    Qwen3VLOffPoilicyDistillPrepareDataForward,
    Qwen3VLPrepareDataForward,
)
from gpatch_v4.extended_model.rollout_attr_hook import ApplySamplingRolloutAttrLLM
from gpatch_v4.extended_model.wemm3_embedding import Wemm3EmbeddingPrepareDataForward

try:
    from gpatch_v4.extended_model.welm_v4 import (
        WelmV4DpoPrepareDataForwardLLM,
        WelmV4PrepareDataForwardLLM,
    )
except ImportError:
    WelmV4PrepareDataForwardLLM = None
    WelmV4DpoPrepareDataForwardLLM = None

try:
    from gpatch_v4.extended_model.welm_omni_v4_5 import WelmOmniV45PrepareDataForward
except ImportError:
    WelmOmniV45PrepareDataForward = None

DEFAULT = "default"

REGISTER_APPLY_SAMPLING_ROLLOUT_ATTR = {
    DEFAULT: ApplySamplingRolloutAttrLLM,
    MODEL_ARCH.QWEN3_VL: ApplySamplingRolloutAttrMultiModal,
    MODEL_ARCH.QWEN3_VL_MOE: ApplySamplingRolloutAttrMultiModal,
    MODEL_ARCH.QWEN3_5: ApplySamplingRolloutAttrQwen3_5,
    MODEL_ARCH.QWEN3_5_MOE: ApplySamplingRolloutAttrQwen3_5_MOE,
    MODEL_ARCH.QWEN3_OMNI_MOE: ApplySamplingRolloutAttrMultiModal,
    MODEL_ARCH.QWEN3_5_WEMM: ApplySamplingRolloutAttrMultiModal,
    MODEL_ARCH.QWEN3_5_MOE_WEMM: ApplySamplingRolloutAttrMultiModal,
    MODEL_ARCH.QWEN3_VL_WEMM: ApplySamplingRolloutAttrMultiModal,
}

REGISTER_SAMPLER_GENERATE_FUNC = {
    DEFAULT: SamplerGenerateFuncLLM,
    MODEL_ARCH.QWEN3_VL: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_VL_MOE: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_5: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_5_MOE: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_OMNI_MOE: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_5_WEMM: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_5_MOE_WEMM: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.QWEN3_VL_WEMM: SamplerGenerateFuncMultiModal,
    MODEL_ARCH.GEMMA4: SamplerGenerateFuncMultiModal,
}

REGISTER_RL_PREPARE_DATA_FORWARD = {
    DEFAULT:
        PrepareDataForwardLLM,
    **({
        MODEL_ARCH.WELMV4_MOE: WelmV4PrepareDataForwardLLM
    } if WelmV4PrepareDataForwardLLM else {}),
    **(
        {
            MODEL_ARCH.DEEPSEEK_V4: DeepseekV4PrepareDataForwardLLM
        } if DeepseekV4PrepareDataForwardLLM else {}
    ),
    MODEL_ARCH.QWEN3_VL:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_VL_MOE:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5_MOE:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_OMNI_MOE:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5_WEMM:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5_MOE_WEMM:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_VL_WEMM:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.GEMMA4:
        Gemma4PrepareDataForward,
}

REGISTER_SFT_PREPARE_DATA_FORWARD = {
    DEFAULT:
        PrepareDataForwardLLM,
    **({
        MODEL_ARCH.WELMV4_MOE: WelmV4PrepareDataForwardLLM
    } if WelmV4PrepareDataForwardLLM else {}),
    **(
        {
            MODEL_ARCH.DEEPSEEK_V4: DeepseekV4PrepareDataForwardLLM
        } if DeepseekV4PrepareDataForwardLLM else {}
    ),
    MODEL_ARCH.QWEN3_VL:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_VL_MOE:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5_MOE:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_OMNI_MOE:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5_WEMM:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_5_MOE_WEMM:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.QWEN3_VL_WEMM:
        Qwen3VLPrepareDataForward,
    MODEL_ARCH.GEMMA4:
        Gemma4PrepareDataForward,
    MODEL_ARCH.WEMM3_EMBEDDING:
        Wemm3EmbeddingPrepareDataForward,
    MODEL_ARCH.WELM_OMNI_V4_5:
        WelmOmniV45PrepareDataForward,
}

REGISTER_OFF_POLICY_DISTILL_PREPARE_DATA_FORWARD = {
    DEFAULT:
        OffPoilicyDistillPrepareDataForwardLLM,
    **({
        MODEL_ARCH.WELMV4_MOE: WelmV4PrepareDataForwardLLM
    } if WelmV4PrepareDataForwardLLM else {}),
    MODEL_ARCH.QWEN3_VL:
        Qwen3VLOffPoilicyDistillPrepareDataForward,
    MODEL_ARCH.QWEN3_VL_MOE:
        Qwen3VLOffPoilicyDistillPrepareDataForward,
    MODEL_ARCH.QWEN3_5:
        Qwen3VLOffPoilicyDistillPrepareDataForward,
    MODEL_ARCH.QWEN3_5_MOE:
        Qwen3VLOffPoilicyDistillPrepareDataForward,
}

REGISTER_DPO_PREPARE_DATA_FORWARD = {
    DEFAULT:
        DpoPrepareDataForwardLLM,
    **(
        {
            MODEL_ARCH.WELMV4_MOE: WelmV4DpoPrepareDataForwardLLM
        } if WelmV4DpoPrepareDataForwardLLM else {}
    ),
    **(
        {
            MODEL_ARCH.DEEPSEEK_V4: DeepseekV4DpoPrepareDataForwardLLM
        } if DeepseekV4DpoPrepareDataForwardLLM else {}
    ),
    MODEL_ARCH.QWEN3_VL:
        Qwen3VLDpoPrepareDataForward,
    MODEL_ARCH.QWEN3_VL_MOE:
        Qwen3VLDpoPrepareDataForward,
    MODEL_ARCH.QWEN3_5:
        Qwen3VLDpoPrepareDataForward,
    MODEL_ARCH.QWEN3_5_MOE:
        Qwen3VLDpoPrepareDataForward,
}


class ApplySamplingRolloutAttrFactory:
    """Factory for rollout attribute handlers."""
    @staticmethod
    def get_apply_sampling(config):
        """Return the appropriate ``ApplySamplingRolloutAttr`` for the model arch.

        Parameters
        ----------
        config : object
            Configuration with ``policy.model_arch``.

        Returns
        -------
        ApplySamplingRolloutAttrBase
        """
        if config.policy.model_arch in REGISTER_APPLY_SAMPLING_ROLLOUT_ATTR:
            return REGISTER_APPLY_SAMPLING_ROLLOUT_ATTR[config.policy.model_arch](config)

        return REGISTER_APPLY_SAMPLING_ROLLOUT_ATTR[DEFAULT](config)


class SamplerGenerateFuncFactory:
    """Factory for sampler generation functions."""
    @staticmethod
    def get_gen_func(config, idx):
        """Return the appropriate sampler generation function.

        Parameters
        ----------
        config : object
            Configuration with sampler model info.
        idx : int
            Sampler index.

        Returns
        -------
        SamplerGenerateFunc
        """
        model_info = config.sampler.model_info[idx]
        if model_info.model_arch in REGISTER_SAMPLER_GENERATE_FUNC:
            return REGISTER_SAMPLER_GENERATE_FUNC[model_info.model_arch]()

        return REGISTER_SAMPLER_GENERATE_FUNC[DEFAULT]()


class PrepareDataForwardFactory:
    """Factory for data-preparation-and-forward handlers."""
    @staticmethod
    def get_prepare_data_fwd(config):
        """Return the appropriate data preparation handler.

        Parameters
        ----------
        config : object

        Returns
        -------
        PrepareDataForward
        """
        register_clss = None
        if isinstance(config, (RlConfig, OnPolicyDistillConfig)):
            register_clss = REGISTER_RL_PREPARE_DATA_FORWARD
        elif isinstance(config, OffPolicyDistillConfig):
            register_clss = REGISTER_OFF_POLICY_DISTILL_PREPARE_DATA_FORWARD
        elif isinstance(config, FinetuneConfig):
            register_clss = REGISTER_SFT_PREPARE_DATA_FORWARD
        elif isinstance(config, RewardConfig):
            # reward model reuses the SFT prepare-data classes (they carry rm_train)
            register_clss = REGISTER_SFT_PREPARE_DATA_FORWARD
        elif isinstance(config, DpoConfig):
            register_clss = REGISTER_DPO_PREPARE_DATA_FORWARD
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

        if config.policy.model_arch in register_clss:
            return register_clss[config.policy.model_arch](config)

        return register_clss[DEFAULT](config)
