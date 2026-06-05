import copy
import gc
import inspect
import json
import os
import random
import re
import shutil
import threading
import traceback
import warnings
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed
from packaging.version import Version

from megatron.core import dist_checkpointing, mpu, package_info, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    HAVE_NVRX,
    TorchDistSaveShardedStrategy,
)
from megatron.core.optimizer import MegatronOptimizer

from gpatch_v4.configs.checkpoint_config import CheckpointConfig
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model
from gpatch_v4.utils.common_utils import (
    assert_hf_metadata_cache_exists,
    copy_cached_hf_metadata_files,
    log,
    logging_rank0,
    save_args_json,
    sync_cuda_and_get_time,
)


def _get_hf_save_path(checkpoint_config, iteration):
    if checkpoint_config.export_hf_save_path is not None:
        export_dir = Path(checkpoint_config.export_hf_save_path) / f"{iteration}"
    else:
        export_dir = Path(checkpoint_config.save_ckpt_path) / f"hf/{iteration}"
    return export_dir


def _mbridge_save_hf(config, iteration, model, bridge):
    cpu_barrier()
    logging_rank0(f"begin to mbridge_save_hf")
    checkpoint_config = config.checkpoint
    start_eport_hf = sync_cuda_and_get_time()

    export_dir = _get_hf_save_path(checkpoint_config, iteration)

    if os.path.exists(export_dir):
        logging_rank0(f"WARNING: may override the huggingface checkpoint at {export_dir}")
    else:
        os.makedirs(export_dir, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        assert_hf_metadata_cache_exists(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
        )
    cpu_barrier()
    unwrapped_model = unwrap_model(model)

    assert hasattr(bridge, "safetensor_io") and bridge.safetensor_io is not None, (
        "bridge.safetensor_io must be initialized when building from mbridge"
    )

    # 兼容不同版本的 mbridge
    save_func_sig = inspect.signature(bridge.save_weights)
    save_weights_kwargs = {}
    if "distributed_filesystem" in save_func_sig.parameters:
        save_weights_kwargs["distributed_filesystem"
                           ] = checkpoint_config.mbridge_distributed_filesystem
    if "strict" in save_func_sig.parameters:
        save_weights_kwargs["strict"] = checkpoint_config.strict_export
    bridge.save_weights(unwrapped_model, export_dir, memory_efficient=True, **save_weights_kwargs)
    cpu_barrier()
    if torch.distributed.get_rank() == 0:
        copy_cached_hf_metadata_files(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
            export_dir,
        )
    save_args_json(config, export_dir)
    cpu_barrier()
    end_eport_hf = sync_cuda_and_get_time()
    logging_rank0(
        f"Finish to mbridge_save_hf {export_dir}, time cost:{end_eport_hf - start_eport_hf}"
    )
    return export_dir


def _megatron_bridge_save_hf(config, iteration, model, bridge):
    cpu_barrier()
    logging_rank0(f"begin to megatron_bridge_save_hf")
    checkpoint_config = config.checkpoint
    start_eport_hf = sync_cuda_and_get_time()

    export_dir = _get_hf_save_path(checkpoint_config, iteration)

    if os.path.exists(export_dir):
        logging_rank0(f"WARNING: may override the huggingface checkpoint at {export_dir}")
    else:
        os.makedirs(export_dir, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        assert_hf_metadata_cache_exists(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
        )
    cpu_barrier()

    save_func_sig = inspect.signature(bridge.save_hf_weights)
    extra_kwargs = {}
    if "distributed_save" in save_func_sig.parameters:
        extra_kwargs["distributed_save"] = checkpoint_config.mbridge_distributed_filesystem
        extra_kwargs["save_every_n_ranks"] = checkpoint_config.mbridge_save_every_n_ranks
    if "strict" in save_func_sig.parameters:
        extra_kwargs["strict"] = checkpoint_config.strict_export
    bridge.save_hf_weights(model, export_dir, **extra_kwargs)
    cpu_barrier()
    if torch.distributed.get_rank() == 0:
        copy_cached_hf_metadata_files(
            config.policy.hf_model_path,
            checkpoint_config.save_ckpt_path,
            export_dir,
        )

    save_args_json(config, export_dir)
    cpu_barrier()
    end_eport_hf = sync_cuda_and_get_time()
    logging_rank0(
        f"Finish to megatron_bridge_save_hf {export_dir}, time cost:{end_eport_hf - start_eport_hf}"
    )
    return export_dir


def _update_tokenizer_configs(
    ckpt_dir: str, override_tokenizer_special_token: dict, debug_dry_run: bool = False
) -> dict:
    """Update config.json and tokenizer_config.json in `ckpt_dir`.

    Args:
        ckpt_dir:
        override_tokenizer_special_token: e.g. ``{"eos_token": [id, str]}``.
        debug_dry_run: If True, print changes without writing.

    Returns:
        Dict of changes applied.
    """
    changes = {"config.json": {}, "tokenizer_config.json": {}}

    def load_json_file(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    for k, v in override_tokenizer_special_token.items():
        assert len(v) == 2, f"len(v) must be 2, but got {len(v)} items {k=} {v=}"

    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        config = load_json_file(config_path)

        for k, v in override_tokenizer_special_token.items():
            _k = f"{k}_id"
            if _k in config.keys():
                original_token_id = config.get(_k)
                if original_token_id == v[0]:
                    continue

                config[_k] = v[0]
                changes["config.json"][_k] = {"old": original_token_id, "new": v[0]}

        if not debug_dry_run:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            log(f"update_tokenizer_configs {config_path}", rank=0)
        else:
            log(f"[DRY RUN] Would update {config_path}", rank=0)
    else:
        log(f"warning {config_path} not found", rank=0)

    tokenizer_config_path = os.path.join(ckpt_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        tokenizer_config = load_json_file(tokenizer_config_path)
        target_tokens = ["eos_token"]

        for target_token in target_tokens:
            if target_token in override_tokenizer_special_token.keys():
                original_token = tokenizer_config.get(target_token)
                new_token = override_tokenizer_special_token[target_token][1]
                if new_token == original_token:
                    continue

                tokenizer_config[target_token] = new_token
                changes["tokenizer_config.json"][target_token] = {
                    "old": original_token,
                    "new": new_token
                }

        if not debug_dry_run:
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            log(f"update_tokenizer_configs {tokenizer_config_path}", rank=0)
        else:
            log(f"[DRY RUN] Would update {tokenizer_config_path}", rank=0)
    else:
        log(f"warning {tokenizer_config_path} not found", rank=0)

    log(f"total update changes {changes}", rank=0)
    return changes


def bridge_save_hf(config, iteration, model, bridge, override_tokenizer_special_token: dict = None):
    try:
        if config.training.build_from_mbridge:
            save_path = _mbridge_save_hf(config, iteration, model, bridge)
        else:
            save_path = _megatron_bridge_save_hf(config, iteration, model, bridge)

        if torch.distributed.get_rank() == 0:
            if override_tokenizer_special_token is not None:
                assert isinstance(
                    override_tokenizer_special_token, dict
                ), f"{type(override_tokenizer_special_token)}"
                _update_tokenizer_configs(save_path, override_tokenizer_special_token)

    except Exception as e:
        log(f"bridge_save_hf failed: {e}")
        traceback.print_exc()
        raise e


def get_latest_checkpoint_folder(path):
    if path is None:
        return None

    if torch.distributed.get_rank() == 0:
        latest_checkpoint_file = os.path.join(path, "latest_checkpointed_iteration.txt")

        if os.path.exists(latest_checkpoint_file):
            with open(latest_checkpoint_file, "r") as f:
                step = int(f.read().strip())
        else:
            step = -1
    else:
        step = 0
    step_tensor = torch.tensor(step, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.SUM)
    latest_ckpt_step = step_tensor.item()
    if latest_ckpt_step == -1:
        return None
    return latest_ckpt_step


def get_rng_state(use_dist_ckpt: bool = True, data_parallel_random_init: bool = False):
    """collect rng state across data parallel ranks"""
    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }

    if torch.cuda.is_available():
        rng_state["cuda_rng_state"] = torch.cuda.get_rng_state()

    rng_state_list = None
    if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size(
    ) > 1 and data_parallel_random_init:
        rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list, rng_state, group=mpu.get_data_parallel_group()
        )
    else:
        rng_state_list = [rng_state]

    if use_dist_ckpt:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        rng_state_list = ShardedObject(
            "rng_state",
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
        )

    return rng_state_list


def load_rng_states(rng_states, data_parallel_random_init=False, use_dist_ckpt=True):
    # access rng_state for data parallel rank
    if data_parallel_random_init:
        rng_states = rng_states[mpu.get_data_parallel_rank()]
    else:
        rng_states = rng_states[0]
    random.setstate(rng_states["random_rng_state"])
    np.random.set_state(rng_states["np_rng_state"])
    torch.set_rng_state(rng_states["torch_rng_state"])

    if torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_states["cuda_rng_state"])

    # Check for empty states array
    if not rng_states["rng_tracker_states"]:
        raise KeyError
    tensor_parallel.get_cuda_rng_tracker().set_states(rng_states["rng_tracker_states"])


def _write_no_fork(transform_list, use_msc, rank, write_buckets, global_results_queue):
    """In-process checkpoint writer, drop-in for
    ``FileSystemWriterAsync.write_preloaded_data_multiproc``.

    Why
    ---
    Megatron 的 ``write_preloaded_data_multiproc`` 通过
    ``mp.get_context("fork")`` 写 checkpoint。POSIX 下多线程进程 fork 是
    UB：子进程只保留调用线程，其他线程持有的锁/资源残留，导致内存分配器、
    CRC 库访问到损坏状态，最终在 ``PyTorchStreamWriter::writeRecord →
    crc32_16bytes`` segfault。Megatron-LM 独立训练 (torchrun) 后台线程少，
    撞概率低；Ray Actor 进程额外有 gRPC / object store / uvloop /
    health-check 等线程，fork 时几乎必然撞锁，因此稳定复现。

    本函数在调用者进程内直接做相同的 I/O，避免 fork。
    """
    from torch.distributed.checkpoint.filesystem import _write_item

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        write_results = {}
        extra_kwargs = {}
        if "serialization_format" in inspect.signature(_write_item).parameters:
            from torch.distributed.checkpoint.filesystem import SerializationFormat
            extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE

        open_fn = open
        if use_msc:
            import multistorageclient as msc
            open_fn = msc.open

        for i, write_bucket in enumerate(write_buckets):
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            local_results = []
            with open_fn(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(
                        _write_item(
                            *transform_list,
                            stream,
                            data,
                            write_item,
                            storage_key,
                            **extra_kwargs,
                        )
                    )
                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(
                        _write_item(
                            *transform_list,
                            stream,
                            tensor,
                            write_item,
                            storage_key,
                            **extra_kwargs,
                        )
                    )
                if use_msc:
                    stream.fsync()
                else:
                    os.fsync(stream.fileno())
            write_results[i] = local_results

        global_results_queue.put(write_results)
    except Exception as e:
        global_results_queue.put(RuntimeError(f"In-process write failure: {e}"))
    finally:
        if gc_was_enabled:
            gc.enable()


def save_dist_checkpointing(sharded_state_dict, ckpt_path, async_save=False):
    """Save sharded checkpoint via Megatron dist_checkpointing.

    通过 ``async_sharded_save=True`` 取 ``AsyncRequest`` 并把 ``async_fn``
    （默认 fork 子进程的 ``write_preloaded_data_multiproc``）替换为
    ``_write_no_fork``，避免 Ray Actor 多线程 fork segfault。详见
    ``_write_no_fork``。
    """
    validate_sharding_integrity = True
    save_strategy = TorchDistSaveShardedStrategy("torch_dist", 1, thread_count=1)
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    async_request = dist_checkpointing.save(
        sharded_state_dict,
        ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=True,
        validate_access_integrity=validate_sharding_integrity,
        async_strategy="nvrx" if HAVE_NVRX else "mcore",
    )

    if async_request.async_fn is not None:
        orig = async_request.async_fn
        transform_list = orig.args[0] if hasattr(orig, "args") and len(orig.args) >= 1 else []
        use_msc = orig.args[1] if hasattr(orig, "args") and len(orig.args) >= 2 else False
        async_request = async_request._replace(
            async_fn=partial(_write_no_fork, transform_list, use_msc)
        )

    if async_save:
        return async_request

    async_request.execute_sync()
    return None


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(
        sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy
    )

    return state_dict


def _build_sharded_state_dict_metadata(
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None
) -> dict:
    """Builds metadata used for sharded_state_dict versioning.

    The metadata is passed into model/optimizer ``sharded_state_dict``
    methods. Keep it minimal, semantically named (e.g.
    ``distrib_optim_sharding_type``); a plain integer/SemVer version
    flag is discouraged because metadata is shared by all
    models/optimizers and a single linear version can't cover them.

    Args:
        args: Arguments namespace
        dp_cp_group: DP+CP group; falls back to mpu API if None.
    """
    metadata = {}
    if Version(package_info.__version__) <= Version("0.13.1"):
        return metadata

    metadata['distrib_optim_sharding_type'] = 'dp_reshardable'

    metadata['singleton_local_shards'] = False
    metadata['chained_optim_avoid_prefix'] = True
    # Add dp_cp_group to metadata. If not provided, fallback to global parallel state.
    if dp_cp_group is None:
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    metadata['dp_cp_group'] = dp_cp_group
    return metadata


def generate_state_dict(
    models,
    optimizer,
    lr_scheduler,
    generate_optimizer: bool = True,
    generate_model: bool = True,
    is_loading: bool = False,
):
    torch.cuda.synchronize()
    state_dict = {}

    sharded_sd_metadata = dict(metadata=_build_sharded_state_dict_metadata())

    # Should always generate model state dict
    # All ranks Save Model to reduce memory pressure
    # Get sharded state dict, notice that state_dict will collect among dp groups, causing memory pressure
    if generate_model:
        for vpp_rank, model in enumerate(models):
            if len(models) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                key = f"model{vpp_rank}" if len(models) > 1 else "model"
            else:
                key = "model"
            if hasattr(model, "module"):
                model = model.module
            state_dict[key] = model.sharded_state_dict(**sharded_sd_metadata)

    # Optimizer State Dict
    if generate_optimizer:
        torch.distributed.barrier()
        optimizer_sharded_states = optimizer.sharded_state_dict(
            state_dict, is_loading=is_loading, **sharded_sd_metadata
        )
        state_dict["optimizer"] = optimizer_sharded_states

        if lr_scheduler is not None:
            lr_state_dict = lr_scheduler.state_dict()
            state_dict["lr_scheduler"] = lr_state_dict

        # RNG States State Dict
        torch.distributed.barrier()
        rng_state = get_rng_state()
        state_dict["rng_state"] = rng_state

    return state_dict


def get_dist_checkpoint_path(checkpoint_path, global_step):
    dist_checkpoint_path = os.path.join(checkpoint_path, 'iter_{:07d}'.format(global_step))
    return dist_checkpoint_path


def save_checkpoint(
    config,
    models,
    optimizer: MegatronOptimizer = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    global_step: int = 0,
    bridge=None,
    override_tokenizer_special_token: dict = None
):
    tracker_filename = "latest_checkpointed_iteration.txt"
    checkpoint_config = config.checkpoint
    # record the previous global step
    dist_checkpoint_path = get_dist_checkpoint_path(checkpoint_config.save_ckpt_path, global_step)
    os.makedirs(dist_checkpoint_path, exist_ok=True)
    log(f"saving checkpoint to {dist_checkpoint_path}", rank=0)
    if not isinstance(models, list):
        models = [models]
    if checkpoint_config.convert_mcore_to_hf_online and checkpoint_config.save_ckpt_path is not None:
        assert bridge is not None
        bridge_save_hf(
            config,
            global_step,
            models,
            bridge,
            override_tokenizer_special_token=override_tokenizer_special_token
        )

    # Note that model weights, optimizer states, and extra states are generated
    # together in a state dict, we save them in one time
    if checkpoint_config.use_dist_checkpointing:
        # Generate state dict for saving
        state_dict = generate_state_dict(
            models,
            optimizer,
            lr_scheduler,
            generate_optimizer=not checkpoint_config.no_save_optim,
            generate_model=not checkpoint_config.skip_save_mcore_model,
        )
        log(f"Generated state dict for saving: {state_dict.keys()}", rank=0)
        if not checkpoint_config.skip_save_mcore_model:
            for vpp_rank, _model in enumerate(models):
                if len(models) > 1:
                    model_i_keys = state_dict[f"model{vpp_rank}"].keys()
                    log(f"Generated state dict for model saving: {model_i_keys}", rank=0)
                else:
                    log(
                        f"Generated state dict for model saving: {state_dict['model'].keys()}",
                        rank=0
                    )
        else:
            log(f"NOT Generated state dict for model saving", rank=0)
        # Start Async save if enabled
        async_save_request = save_dist_checkpointing(
            sharded_state_dict=state_dict,
            ckpt_path=dist_checkpoint_path,
            async_save=checkpoint_config.async_save,
        )

        # Synchronize all async save requests
        if not checkpoint_config.async_save:
            assert async_save_request is None, "Async save request should be None when not using async save."
            torch.distributed.barrier()
    else:
        assert checkpoint_config.use_hf_checkpoint, "When not using distributed checkpointing, use_hf_checkpoint should be True."
        raise NotImplementedError("hf format checkpoint has not been implemented")
        # TODO (@yeazhao) hf format checkpoint

    prev_iteration = 0
    save_retain_interval = checkpoint_config.save_retain_interval
    if save_retain_interval is not None:
        tracker_path = os.path.join(checkpoint_config.save_ckpt_path, tracker_filename)
        if os.path.exists(tracker_path):
            with open(tracker_path, 'r') as f:
                content = f.read().strip()
                if content:
                    prev_iteration = int(content)
                else:
                    prev_iteration = 0
        else:
            prev_iteration = 0

    if 0 == torch.distributed.get_rank():
        latest_checkpoint_file = os.path.join(checkpoint_config.save_ckpt_path, tracker_filename)
        with open(latest_checkpoint_file, "w") as f:
            f.write(str(global_step))
    cpu_barrier()
    log(f"successfully saved checkpoint to {dist_checkpoint_path}", rank=0)

    def delete_checkpoint(iteration_to_delete):
        directory = 'iter_{:07d}'.format(iteration_to_delete)
        checkpoint_name = os.path.join(checkpoint_config.save_ckpt_path, directory)
        hf_dir = format(iteration_to_delete)
        hf_checkpoint_name = os.path.join(checkpoint_config.save_ckpt_path, "hf", hf_dir)
        try:
            shutil.rmtree(checkpoint_name)
            log(
                f"successfully deleted checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
            )

        except Exception as e:
            log(
                f'encountered exception "{e}" when trying to delete checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}'
            )
            # Any exception encountered in checkpoint deletion can be ignored and is not fatal.
            pass
        try:
            shutil.rmtree(hf_checkpoint_name)
            log(
                f"successfully deleted hf checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}"
            )
        except Exception as e:
            log(
                f'encountered exception "{e}" when trying to delete hf checkpoint from iteration {iteration_to_delete:7d} at {checkpoint_config.save_ckpt_path}'
            )
            # Any exception encountered in checkpoint deletion can be ignored and is not fatal.
            pass

    def sorted_checkpoints(output_dir=None, checkpoint_prefix="iter_", use_mtime=True):
        ordering_and_checkpoint_path = []
        glob_checkpoints = [
            str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}*") if os.path.isdir(x)
        ]
        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        # mtime is not reliable on all filesystems, especially on some fuse fs in cloud environments
        # so we check if the mtime is fake and fallback to numerical ordering if needed
        if use_mtime and len(ordering_and_checkpoint_path) > 1:
            mtime_diff = checkpoints_sorted[-1][0] - checkpoints_sorted[0][0]
            if mtime_diff < 1.0:  # less than 1 second, which is almost impossible when mtime works fine
                warnings.warn(
                    "mtime may not be reliable on this filesystem, falling back to numerical ordering"
                )
                return sorted_checkpoints(
                    use_mtime=False, output_dir=output_dir, checkpoint_prefix=checkpoint_prefix
                )
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def rotate_checkpoints(output_dir=None):
        save_total_limit = checkpoint_config.save_total_limit
        if save_total_limit is None or save_total_limit <= 0:
            return
        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = sorted_checkpoints(
            output_dir=output_dir, checkpoint_prefix="iter_", use_mtime=True
        )
        if len(checkpoints_sorted) <= save_total_limit:
            return

        num_of_ckpt_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:num_of_ckpt_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            log(
                f"deleting older checkpoint [{checkpoint}] due to --gcore-save-total-limit={save_total_limit}."
            )
            shutil.rmtree(checkpoint, ignore_errors=True)

        # delete hf ckpts
        hf_ckpt_sorted = sorted_checkpoints(
            output_dir=os.path.join(output_dir, "hf"), checkpoint_prefix="", use_mtime=True
        )
        if len(hf_ckpt_sorted) <= save_total_limit:
            return

        num_of_ckpt_to_delete = max(0, len(hf_ckpt_sorted) - save_total_limit)
        hf_ckpt_to_be_deleted = hf_ckpt_sorted[:num_of_ckpt_to_delete]
        for ckpt in hf_ckpt_to_be_deleted:
            log(
                f"deleting older hf checkpoint [{ckpt}] due to --gcore-save-total-limit={save_total_limit}."
            )
            shutil.rmtree(ckpt, ignore_errors=True)

    def async_cleanup(prev_iteration):
        delete_checkpoint(prev_iteration)
        rotate_checkpoints(checkpoint_config.save_ckpt_path)

    if 0 == torch.distributed.get_rank():
        if save_retain_interval is not None:
            if prev_iteration > 0 and prev_iteration != global_step and prev_iteration % save_retain_interval != 0:
                directory = 'iter_{:07d}'.format(prev_iteration)
                checkpoint_name = os.path.join(checkpoint_config.save_ckpt_path, directory)

                # Don't delete if `checkpoint_name` is a symbolic link.
                if os.path.islink(checkpoint_name):
                    log(
                        f'skipping deleting checkpoint from iteration {prev_iteration:7d} at {checkpoint_config.save_ckpt_path} since it is a symbolic link'
                    )
                else:
                    # Asynchronous version of async_cleanup(prev_iteration).
                    threading.Thread(target=async_cleanup, args=(prev_iteration, )).start()
        else:
            threading.Thread(target=rotate_checkpoints,
                             args=(checkpoint_config.save_ckpt_path, )).start()


def load_checkpoint(
    config,
    models,
    optimizer: MegatronOptimizer = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    global_step: int = 0,
    bridge=None,
):
    checkpoint_config = config.checkpoint
    assert global_step is not None, \
        f"fail to read training step from dir {checkpoint_config.load_ckpt_path}/latest_checkpointed_iteration.txt"
    dist_checkpoint_path = get_dist_checkpoint_path(checkpoint_config.load_ckpt_path, global_step)
    assert os.path.exists(
        dist_checkpoint_path
    ), f"Checkpoint path {dist_checkpoint_path} does not exist."
    log(f"loading checkpoint from {dist_checkpoint_path}", rank=0)

    if not isinstance(models, list):
        models = [models]

    # 当 skip_save_mcore_model 为 True 时，模型权重需要从 HF 格式通过 bridge 加载
    load_model_from_hf = checkpoint_config.skip_save_mcore_model

    # Get State Dict for loading
    sharded_state_dict = generate_state_dict(
        models,
        optimizer,
        lr_scheduler,
        generate_optimizer=not checkpoint_config.no_load_optim,
        generate_model=not load_model_from_hf,
        is_loading=True,
    )
    log(f"Generated state dict for loading: {sharded_state_dict.keys()}", rank=0)

    # Load Dist Checkpointing
    state_dict = load_dist_checkpointing(
        sharded_state_dict=sharded_state_dict,
        ckpt_dir=dist_checkpoint_path,
    )

    if load_model_from_hf:
        # 从 HF 格式通过 bridge 加载模型权重
        assert bridge is not None, (
            "bridge must be provided when skip_save_mcore_model=True, "
            "so that model weights can be loaded from HF format."
        )
        hf_model_path = _get_hf_save_path(checkpoint_config, global_step)
        log(f"Loading model weights from HF format via bridge: {hf_model_path}", rank=0)
        if config.training.build_from_mbridge:
            bridge.load_weights(models, hf_model_path, memory_efficient=True)
        else:
            bridge.load_hf_weights(models, hf_model_path, allowed_mismatched_params=[])
        log(f"Loaded model weights from HF format: {hf_model_path}", rank=0)
    elif checkpoint_config.use_dist_checkpointing:
        assert "model" in state_dict or any(
            f"model{vpp_rank}" in state_dict for vpp_rank in range(len(models))
        ), f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {checkpoint_config.load_ckpt_path}."
        for vpp_rank, _model in enumerate(models):
            if len(models) == 1:
                model_state_dict = state_dict["model"]
            else:
                assert f"model{vpp_rank}" in state_dict, f"model{vpp_rank} not found in state_dict"
                model_state_dict = state_dict[f"model{vpp_rank}"]
            mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            models[vpp_rank].load_state_dict(model_state_dict)
        log(f"Loaded sharded model checkpoint from {checkpoint_config.load_ckpt_path}", rank=0)
    elif checkpoint_config.use_hf_checkpoint:
        raise NotImplementedError("hf format checkpoint has not been implemented")

    if not checkpoint_config.no_load_optim:
        assert "optimizer" in state_dict, (
            f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {checkpoint_config.load_ckpt_path}."
        )
        optimizer_state_dict = state_dict["optimizer"]
        optimizer.load_state_dict(optimizer_state_dict)
        log(f"Loaded optimizer checkpoint from {checkpoint_config.load_ckpt_path}", rank=0)
        if "lr_scheduler" in state_dict:
            lr_scheduler_state_dict = state_dict["lr_scheduler"]
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log(
                    f"Loaded LR scheduler checkpoint from {checkpoint_config.load_ckpt_path}",
                    rank=0
                )

        assert "rng_state" in state_dict, (
            f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {checkpoint_config.load_ckpt_path}."
        )
        rng_state = state_dict["rng_state"]
        load_rng_states(rng_state)
        log(f"Loaded RNG states from {checkpoint_config.load_ckpt_path}", rank=0)
    cpu_barrier()
    log(f"successfully loaded checkpoint from {dist_checkpoint_path}", rank=0)
    return global_step
