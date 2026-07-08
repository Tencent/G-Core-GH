import traceback
from contextlib import suppress

from gpatch_v4.configs.config import (
    DpoConfig,
    FinetuneConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
    RewardConfig,
    RlConfig,
    T2iDpoConfig,
    T2iEditSftConfig,
    T2iRlConfig,
)

try:
    # Fsdp2EngineLm / Fsdp2EngineT2i require megatron_datasets; McoreEngine requires megatron.core.
    # Guard both so that Bagel/WGOv3 pipelines (which use FSDP2EngineBase directly) are unaffected.
    from gpatch_v4.training_backend.fsdp2_backend import Fsdp2EngineLm, Fsdp2EngineT2i
    from gpatch_v4.training_backend.loss_factory import (
        LOSS_FUNC_REGISTRY,
        register_custom_loss_fn,
    )
    from gpatch_v4.training_backend.megatron_backend import McoreEngine
except ImportError:
    traceback.print_exc()
    Fsdp2EngineLm = None  # type: ignore[assignment,misc]
    Fsdp2EngineT2i = None  # type: ignore[assignment,misc]
    LOSS_FUNC_REGISTRY = None  # type: ignore[assignment,misc]
    register_custom_loss_fn = None  # type: ignore[assignment,misc]
    McoreEngine = None  # type: ignore[assignment,misc]


class TrainingEngineFactory:
    """Factory that returns the appropriate training engine."""
    @staticmethod
    def get_training_engine(config, **kwargs):
        """Instantiate a training engine based on the backend type.

        Parameters
        ----------
        config : object
            Training configuration with ``training.training_backend``.
        **kwargs : dict

        Returns
        -------
        BaseEngine

        Raises
        ------
        NotImplementedError
            If the backend is not supported for the config type.
        """
        assert config.training.training_backend in ["fsdp2", "mcore"]
        if isinstance(config, (
            T2iRlConfig,
            T2iDpoConfig,
            T2iEditSftConfig,
        )):
            if config.training.training_backend == "fsdp2":
                return Fsdp2EngineT2i(config, **kwargs)
            else:
                raise NotImplementedError(
                    f"backend {config.training.training_backend} not implemented"
                )
        elif isinstance(
            config, (
                FinetuneConfig,
                RlConfig,
                OnPolicyDistillConfig,
                OffPolicyDistillConfig,
                DpoConfig,
                RewardConfig,
            )
        ):
            if config.training.training_backend == "fsdp2":
                return Fsdp2EngineLm(config, **kwargs)
            else:
                return McoreEngine(config, **kwargs)
        else:
            raise NotImplementedError(f"config {type(config)} not implemented")


BUILDIN_LOSS_FUNC = list(LOSS_FUNC_REGISTRY.keys())
