import logging
import os
import random
from contextlib import contextmanager
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist

from megatron.core import mpu, tensor_parallel
from megatron.core.parallel_state import (
    RankGenerator,
    create_group,
    default_embedding_ranks,
    default_position_embedding_ranks,
)

from gpatch_v4.configs.dist_config import DistConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_MODEL_PARALLEL_GROUP_GLOO = None
_MODEL_PARALLEL_GLOBAL_RANKS_GLOO = None
_GROUP_GLOO = None

# model parallel with cp
_MODEL_AND_CONTEXT_PARALLEL_GROUP = None
_MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO = None
_MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP = None
_MODEL_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = None
_MODEL_EXPERT_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = None


def set_random_seed(config, data_parallel_random_init: bool = False):
    """Set random seeds for reproducibility across parallel ranks.

    Parameters
    ----------
    config : object
        Training configuration (must have ``training.seed`` if available).
    data_parallel_random_init : bool, optional
    """
    if not hasattr(config, 'training'):
        seed = 100 * mpu.get_pipeline_model_parallel_rank() + 5
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    seed = config.training.seed
    if seed is not None and seed > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if config.training.training_backend == "mcore" and torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def is_rtx_pro_5000() -> bool:
    """Return True when GPU 0 is an NVIDIA RTX PRO 5000."""
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    normalized = name.upper().replace("-", " ")
    return "PRO 5000" in normalized or "PRO5000" in normalized


def disable_flash_attn_3():
    """Disable Flash Attention 3 in TransformerEngine.

    FA3's deterministic backward is broken (dQ accumulation uses
    non-deterministic global atomics on Hopper). Monkey-patch TE's
    FlashAttentionUtils to prevent FA3 selection at runtime, forcing
    fallback to FA2 which has correct deterministic backward (>= 2.4.1).

    Must be called AFTER TE is imported (module-level imports already
    triggered by megatron.core), but BEFORE the first forward pass.

    注意：升级 TE 后，这里不一定能用
    """
    try:
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            FlashAttentionUtils,
        )
        was_installed = FlashAttentionUtils.v3_is_installed
        FlashAttentionUtils.v3_is_installed = False
        FlashAttentionUtils.set_flash_attention_3_params = staticmethod(lambda: None)
        if was_installed:
            logging.info(
                "Disabled FA3 (v%s) for deterministic mode, falling back to FA2 (v%s)",
                FlashAttentionUtils.fa3_version,
                FlashAttentionUtils.version,
            )
    except ImportError:
        pass


def disable_flash_attn_4():
    """Disable Flash Attention 4 in TransformerEngine.

    FA4 on RTX PRO 5000 (sm_120) is unstable for packed-THD vision
    training (async CUDA IMA). Monkey-patch TE so FA4 is not selected
    and TE falls back to FA2.

    Must be called AFTER TE is imported, but BEFORE the first DPA forward.

    注意：升级 TE 后，这里不一定能用
    """
    try:
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            FlashAttentionUtils,
        )
        was_installed = FlashAttentionUtils.v4_is_installed
        FlashAttentionUtils.v4_is_installed = False
        if was_installed:
            logging.info(
                "Disabled FA4 (v%s), falling back to FA2 (v%s)",
                FlashAttentionUtils.fa4_version,
                FlashAttentionUtils.version,
            )
    except ImportError:
        pass


def apply_flash_attn_disables(training) -> None:
    """Apply ``training.disable_flash_attn_{3,4}`` to the local TE runtime.

    RTX PRO 5000 forces ``disable_flash_attn_4=True`` even if the yaml
    left it false.
    """
    if training is None:
        return
    # TODO(guanyouhe): 后续 fa4 支持 pro5000 后加上
    if is_rtx_pro_5000() and not training.disable_flash_attn_4:
        logging.info(
            "RTX PRO 5000 detected (%s): set disable_flash_attn_4=True",
            torch.cuda.get_device_name(0),
        )
        training.disable_flash_attn_4 = True
    if training.disable_flash_attn_3:
        disable_flash_attn_3()
    if training.disable_flash_attn_4:
        disable_flash_attn_4()


def enable_deterministic_mode():
    """Enable deterministic mode for training.

    IMPORTANT: NCCL env vars (NCCL_DETERMINISTIC, NCCL_ALGO) must be set
    **before** any NCCL communicators are created. Call
    ``enable_deterministic_mode_env()`` early (before
    ``torch.distributed.init_process_group``), then call this function
    afterward to set the remaining torch-level flags.

    FA3/FA4 TE monkey-patches are applied separately via
    ``apply_flash_attn_disables`` from ``training.disable_flash_attn_*``.
    """
    enable_deterministic_mode_env()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(
        "Deterministic mode enabled: CUBLAS_WORKSPACE_CONFIG=%s, NCCL_ALGO=%s",
        os.environ["CUBLAS_WORKSPACE_CONFIG"],
        os.environ.get("NCCL_ALGO", "unset"),
    )


def enable_deterministic_mode_env():
    """Set environment variables required for deterministic execution.

    Must be called **before** ``torch.distributed.init_process_group()`` so
    that NCCL communicators are created with deterministic settings.
    Idempotent — safe to call more than once.
    """
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@contextmanager
def preserve_rng_state():
    """Context manager that saves and restores all RNG states.

    Useful when an operation (e.g. rebuilding a dataloader) may consume
    or reset the global RNG, but the caller needs the RNG to stay
    exactly where it was (e.g. after restoring from a checkpoint).
    """
    state = {
        "random": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
        "rng_tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }
    try:
        yield
    finally:
        random.setstate(state["random"])
        np.random.set_state(state["np"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None:
            torch.cuda.set_rng_state(state["torch_cuda"])
        tensor_parallel.get_cuda_rng_tracker().set_states(state["rng_tracker"])


def initlize_parallel_state(config, dist_config):
    """Initialize Megatron-Core model parallelism and set random seeds.

    Parameters
    ----------
    config : object
    dist_config : DistConfig
    """

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=dist_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=dist_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=dist_config.virtual_pipeline_model_parallel_size,
        use_sharp=False,
        context_parallel_size=dist_config.context_parallel_size,
        expert_model_parallel_size=dist_config.expert_model_parallel_size,
        expert_tensor_parallel_size=dist_config.expert_tensor_parallel_size,
        nccl_communicator_config_path=None,
        distributed_timeout_minutes=dist_config.torch_dist_timeout_minutes,
        order='tp-cp-ep-dp-pp' if not dist_config.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
        dynamic_context_parallel=dist_config.dynamic_context_parallel,
        min_dynamic_context_parallel_size=dist_config.min_dynamic_context_parallel_size,
    )
    if hasattr(config, 'training'):
        set_random_seed(config, data_parallel_random_init=config.training.data_parallel_random_init)
        if config.training.apply_deterministic_mode:
            enable_deterministic_mode()
        apply_flash_attn_disables(config.training)
    else:
        set_random_seed(config, False)


# TODO drop it later
# def init_distributed(meta_info):
#     role_master_addr = meta_info["role_master_addr"]
#     role_master_port = meta_info["role_master_port"]
#     rank = meta_info["rank"]
#     role = meta_info["role"]
#     world_size = meta_info["world_size"]
#     torch_dist_timeout_minutes = meta_info["torch_dist_timeout_minutes"]
#     init_method = f'tcp://{role_master_addr}:{role_master_port}'
#
#     init_process_group_kwargs = {
#         'backend': "nccl",
#         'world_size': world_size,
#         'init_method': init_method,
#         'rank': rank,
#         'timeout': timedelta(minutes=torch_dist_timeout_minutes),
#     }
#     torch.cuda.set_device(rank % torch.cuda.device_count())
#     # logging.info(f"{role} worker {rank} initializing with init_method={init_method}")
#     dist.init_process_group(**init_process_group_kwargs)
#     logging.info(f"{role} worker {rank} initialized with init_method={init_method}")


def destroy_process_group():
    """Destroy the default torch distributed process group."""
    dist.destroy_process_group()
    logging.info(f"torch.dist destroyed process group")


def init_pg(dist_config: DistConfig):
    """Initialize gloo process groups for model-parallel, context-parallel, and expert-parallel.

    Parameters
    ----------
    dist_config : DistConfig
    """
    distributed_timeout_minutes = dist_config.torch_dist_timeout_minutes
    timeout = timedelta(minutes=distributed_timeout_minutes)

    global _GROUP_GLOO
    world_size = torch.distributed.get_world_size()
    ranks = np.arange(world_size)
    _GROUP_GLOO = torch.distributed.new_group(ranks=ranks, timeout=timeout, backend='gloo')

    global _MODEL_PARALLEL_GROUP_GLOO
    global _MODEL_PARALLEL_GLOBAL_RANKS_GLOO
    assert _MODEL_PARALLEL_GROUP_GLOO is None, 'model parallel group is already initialized'

    # partial src from megatron/core/parallel_state.py

    rank = torch.distributed.get_rank()

    #TODO: encoder_tensor_model_parallel_size 和 encoder_tensor_model_parallel_size 暂无定义
    # 使用 initialize_model_parallel 的默认值
    encoder_pipeline_model_parallel_size = 0
    encoder_tensor_model_parallel_size = 0
    get_embedding_ranks = None
    get_position_embedding_ranks = None

    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if encoder_tensor_model_parallel_size == 0 and encoder_pipeline_model_parallel_size > 0:
        encoder_tensor_model_parallel_size = dist_config.tensor_model_parallel_size

    if get_embedding_ranks is None:
        get_embedding_ranks = partial(
            default_embedding_ranks, split_rank=dist_config.pipeline_model_parallel_split_rank
        )

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = partial(
            default_position_embedding_ranks,
            split_rank=dist_config.pipeline_model_parallel_split_rank
        )

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    encoder_model_size = (
        encoder_tensor_model_parallel_size * encoder_pipeline_model_parallel_size *
        dist_config.context_parallel_size
    )
    decoder_model_size = (
        dist_config.tensor_model_parallel_size * dist_config.pipeline_model_parallel_size *
        dist_config.context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size
    data_parallel_size: int = world_size // total_model_size
    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    expert_tensor_model_pipeline_parallel_size = (
        dist_config.expert_tensor_parallel_size * dist_config.expert_model_parallel_size *
        dist_config.pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        order = 'tp-cp-ep-dp-pp' if not dist_config.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp'
        if is_expert:
            d_ranks = RankGenerator(
                tp=dist_config.expert_tensor_parallel_size,
                ep=dist_config.expert_model_parallel_size,
                dp=expert_data_parallel_size,
                pp=dist_config.pipeline_model_parallel_size,
                cp=1,
                order=order,
                rank_offset=encoder_world_size,
            ).get_ranks(group_type, **kwargs)
        else:
            d_ranks = RankGenerator(
                tp=dist_config.tensor_model_parallel_size,
                ep=1,
                dp=data_parallel_size,
                pp=dist_config.pipeline_model_parallel_size,
                cp=dist_config.context_parallel_size,
                order=order,
                rank_offset=encoder_world_size,
            ).get_ranks(group_type, **kwargs)

        if encoder_world_size > 0:
            encoder_rank_generator = RankGenerator(
                tp=encoder_tensor_model_parallel_size,
                ep=1,
                dp=data_parallel_size,
                pp=encoder_pipeline_model_parallel_size,
                cp=dist_config.context_parallel_size,
                order=order,
                rank_offset=0,
            )
        else:
            encoder_rank_generator = None

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == 'tp-pp':
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y

    for ranks in generator_wrapper('tp-pp'):
        group = create_group(
            ranks,
            timeout=timeout,
            backend="gloo",
            group_desc='_MODEL_PARALLEL_GROUP_GLOO',
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP_GLOO = group
            _MODEL_PARALLEL_GLOBAL_RANKS_GLOO = ranks

        # Build the model-parallel groups with cp

    global _MODEL_AND_CONTEXT_PARALLEL_GROUP
    global _MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO
    global _MODEL_AND_CONTEXT_PARALLEL_GLOBAL_RANKS
    global _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP
    global _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _MODEL_AND_CONTEXT_PARALLEL_GROUP is None, 'model and context parallel group is already initialized'
    assert _MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO is None, 'model and context parallel group is already initialized'
    for ranks in generator_wrapper('tp-cp-pp'):
        group = create_group(
            ranks, timeout=timeout, backend="nccl", group_desc='_MODEL_AND_CONTEXT_PARALLEL_GROUP'
        )
        group_gloo = create_group(
            ranks,
            timeout=timeout,
            backend="gloo",
            group_desc='_MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO'
        )
        if rank in ranks:
            _MODEL_AND_CONTEXT_PARALLEL_GROUP = group
            _MODEL_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
            _MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO = group_gloo

    # Build the model-parallel groups expert parallel and cp
    # TODO(@nrwu): check if is_expert is correct
    assert (
        _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP is None
    ), 'model, context and expert parallel group is already initialized'
    for ranks in generator_wrapper('tp-cp-ep-pp', is_expert=True):
        group = create_group(
            ranks,
            timeout=timeout,
            backend="nccl",
            group_desc='_MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP'
        )
        if rank in ranks:
            _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP = group
            _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = ranks


def get_model_parallel_group_gloo():
    """Get the model-parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP_GLOO is not None, 'model parallel gloo group is not initialized'
    return _MODEL_PARALLEL_GROUP_GLOO


def get_model_parallel_src_rank_gloo():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    assert _MODEL_PARALLEL_GLOBAL_RANKS_GLOO is not None, "Model parallel group is not initialized"
    return _MODEL_PARALLEL_GLOBAL_RANKS_GLOO[0]


def cpu_barrier(pg=None):
    """Synchronize all ranks using the gloo process group.

    Parameters
    ----------
    pg : ProcessGroup, optional
    """
    if pg is None:
        pg = _GROUP_GLOO
    torch.distributed.barrier(group=pg)


def is_mp_head():
    """Check if this rank is the model-parallel head (TP rank 0 and PP first stage).

    Returns
    -------
    bool
    """
    return mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0


def is_mp_and_cp_head():
    """Check if this rank is the model+context parallel head.

    Returns
    -------
    bool
    """
    return mpu.is_pipeline_first_stage() \
            and mpu.get_tensor_model_parallel_rank() == 0 \
            and mpu.get_context_parallel_rank() == 0


def is_tp_and_cp_head():
    """Check if this rank is TP rank 0 and CP rank 0.

    Returns
    -------
    bool
    """
    return mpu.get_tensor_model_parallel_rank() == 0 \
            and mpu.get_context_parallel_rank() == 0


def is_last_rank():
    return torch.distributed.get_rank() == torch.distributed.get_world_size() - 1


def is_first_rank():
    return torch.distributed.get_rank() == 0


def get_last_rank(pg=None):
    return torch.distributed.get_world_size(group=pg) - 1


def get_model_and_context_parallel_group(with_expert_parallel=False):
    """Get the model parallel group the caller rank belongs to."""
    if with_expert_parallel:
        assert (
            _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP is not None
        ), 'model, exeprt and context parallel group is not initialized'
        return _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GROUP
    assert _MODEL_AND_CONTEXT_PARALLEL_GROUP is not None, \
        'model and context parallel group is not initialized'
    return _MODEL_AND_CONTEXT_PARALLEL_GROUP


def get_model_and_context_parallel_group_gloo(with_expert_parallel=False):
    """Get the model parallel group the caller rank belongs to."""
    if with_expert_parallel:
        raise NotImplementedError("Expert parallel is not supported")
    assert _MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO is not None, \
        'model and context parallel group is not initialized'
    return _MODEL_AND_CONTEXT_PARALLEL_GROUP_GLOO


def get_model_and_context_parallel_src_rank(with_expert_parallel=False):
    """Get the model parallel group the caller rank belongs to."""
    if with_expert_parallel:
        assert (
            _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'model, expert and context parallel src rank is not initialized'
        return _MODEL_EXPERT_AND_CONTEXT_PARALLEL_GLOBAL_RANKS[0]
    assert _MODEL_AND_CONTEXT_PARALLEL_GLOBAL_RANKS is not None, \
        'model and context parallel src rank is not initialized'
    return _MODEL_AND_CONTEXT_PARALLEL_GLOBAL_RANKS[0]


def cpu_group():
    """Return the global gloo process group.

    Returns
    -------
    ProcessGroup
    """
    return _GROUP_GLOO
