from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model

try:
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
        FullyShardedDataParallel as McoreFullyShardedDataParallel,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import (
        MegatronFSDP,
    )

    ALL_MODULE_WRAPPER_CLASSNAMES = (
        DDP,
        Float16Module,
        McoreFullyShardedDataParallel,
        MegatronFSDP,
    )
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)
