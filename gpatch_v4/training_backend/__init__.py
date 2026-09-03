import traceback
from contextlib import suppress

from gpatch_v4.compat import ensure_typing_self

ensure_typing_self()

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

# Fsdp2EngineLm / Fsdp2EngineT2i require megatron_datasets; McoreEngine requires megatron.core.
# Guard independently so Bagel/WGOv3 (FSDP2EngineBase) stays importable, and an FSDP2
# import failure does not wipe LOSS_FUNC_REGISTRY used by MCore.
try:
    from gpatch_v4.training_backend.fsdp2_backend import Fsdp2EngineLm, Fsdp2EngineT2i
except ImportError:
    traceback.print_exc()
    Fsdp2EngineLm = None  # type: ignore[assignment,misc]
    Fsdp2EngineT2i = None  # type: ignore[assignment,misc]

try:
    from gpatch_v4.training_backend.loss_factory import (
        LOSS_FUNC_REGISTRY,
        register_custom_loss_fn,
    )
except ImportError:
    traceback.print_exc()
    LOSS_FUNC_REGISTRY = None  # type: ignore[assignment,misc]
    register_custom_loss_fn = None  # type: ignore[assignment,misc]

try:
    from gpatch_v4.training_backend.megatron_backend import (
        DynamicBatchMcoreEngine,
        McoreEngine,
    )
except ImportError:
    traceback.print_exc()
    McoreEngine = None  # type: ignore[assignment,misc]
    DynamicBatchMcoreEngine = None  # type: ignore[assignment,misc]

try:
    from gpatch_v4.training_backend.mlite_backend import MliteEngine
    _MLITE_IMPORT_ERROR = None
except ImportError as exc:
    MliteEngine = None  # type: ignore[assignment,misc]
    _MLITE_IMPORT_ERROR = exc


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
        assert config.training.training_backend in ["fsdp2", "mcore", "mlite"]
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
            elif isinstance(config, RlConfig) and config.training.dynamic_batch_train:
                return DynamicBatchMcoreEngine(config, **kwargs)
            elif config.training.training_backend == "mlite":
                if not isinstance(config, FinetuneConfig):
                    raise NotImplementedError("mlite currently supports FinetuneConfig only")
                if MliteEngine is None:
                    raise ImportError(
                        "mlite backend requires megatron.lite; put "
                        "mlite/experimental/lite before Megatron-LM in PYTHONPATH"
                    ) from _MLITE_IMPORT_ERROR
                return MliteEngine(config, **kwargs)
            else:
                return McoreEngine(config, **kwargs)
        else:
            raise NotImplementedError(f"config {type(config)} not implemented")


BUILDIN_LOSS_FUNC = (list(LOSS_FUNC_REGISTRY.keys()) if LOSS_FUNC_REGISTRY is not None else [])
