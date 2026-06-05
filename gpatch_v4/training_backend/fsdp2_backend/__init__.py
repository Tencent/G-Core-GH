# Fsdp2EngineLm / Fsdp2EngineT2i depend on megatron_datasets which is only available
# in certain environments. Guard the import so that Bagel/WGOv3 pipelines that do
# not use these classes can still import fsdp2_backend sub-modules (e.g. fsdp2_utils_bagel).
import traceback

try:
    from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import Fsdp2EngineLm
    from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_t2i import Fsdp2EngineT2i
except ImportError:
    traceback.print_exc()
    Fsdp2EngineLm = None  # type: ignore[assignment,misc]
    Fsdp2EngineT2i = None  # type: ignore[assignment,misc]
