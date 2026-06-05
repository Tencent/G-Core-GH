from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.extended_pipeline.pipeline_bagel import FSDP2EngineBagel
from gpatch_v4.extended_pipeline.pipeline_flux import FluxPipeline
from gpatch_v4.extended_pipeline.pipeline_qwen_image_edit import QwenImageEditPipeline

try:
    from gpatch_v4.extended_pipeline.pipeline_oteam4_3 import Oteam43Pipeline
    from gpatch_v4.extended_pipeline.pipeline_oteam4_4 import Oteam44Pipeline
    _OTEAM_PIPELINES = {
        MODEL_ARCH.OTEAM4_3: Oteam43Pipeline,
        MODEL_ARCH.OTEAM4_4: Oteam44Pipeline,
    }
except ImportError:
    _OTEAM_PIPELINES = {}

REGISTER_PIPELINE = {
    MODEL_ARCH.FLUX: FluxPipeline,
    MODEL_ARCH.BAGEL: FSDP2EngineBagel,
    MODEL_ARCH.QWEN_IMAGE_EDIT: QwenImageEditPipeline,
    **_OTEAM_PIPELINES,
}


class ExtendPipelineFactory:
    """Factory that returns the appropriate pipeline for a given model arch."""
    @staticmethod
    def get_pipeline(config):
        """Instantiate the extended pipeline based on ``config.policy.model_arch``.

        Parameters
        ----------
        config : object
            Top-level configuration with ``policy.model_arch``.

        Returns
        -------
        ExtendedPipelineAbc
        """
        return REGISTER_PIPELINE[config.policy.model_arch](config)
