import json
import os
import time
from typing import Any, Dict

import torch
import torch.distributed as dist

from gpatch_v4.utils import copy_cached_hf_metadata_files, log

_LR_SCHEDULER_STATE = "lr_scheduler.pt"
_LATEST_CHECKPOINT_MARKER = "latest_checkpointed_iteration.txt"
_RANK_LOCAL_TOPOLOGY = "rank_local_topology.json"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _hf_save_path(checkpoint_config: Any, global_step: int) -> str:
    export_root = checkpoint_config.export_hf_save_path
    if export_root is None:
        export_root = os.path.join(checkpoint_config.save_ckpt_path, "hf")
    return os.path.join(export_root, str(global_step))


def _copy_hf_metadata(engine: Any, hf_path: str) -> None:
    if not dist.is_initialized():
        copy_cached_hf_metadata_files(
            engine.policy_config.hf_model_path,
            engine.checkpoint_config.save_ckpt_path,
            hf_path,
        )
        return

    error = None
    cause = None
    if _rank() == 0:
        try:
            copy_cached_hf_metadata_files(
                engine.policy_config.hf_model_path,
                engine.checkpoint_config.save_ckpt_path,
                hf_path,
            )
        except Exception as exc:
            cause = exc
            error = f"{type(exc).__name__}: {exc}"
    error_payload = [error]
    dist.broadcast_object_list(error_payload, src=0)
    if error_payload[0] is not None:
        raise RuntimeError(
            f"mlite online HF export failed to copy metadata: {error_payload[0]}"
        ) from cause


def _save_hf_checkpoint(engine: Any, global_step: int) -> None:
    protocol = engine.handle._extras.get("protocol")
    save_hf_weights = getattr(protocol, "save_hf_weights", None)
    if not callable(save_hf_weights):
        raise RuntimeError(
            "mlite online HF export requires the model protocol to expose save_hf_weights"
        )
    model_chunks = engine.handle._extras.get("model_chunks")
    if not model_chunks:
        raise RuntimeError("mlite online HF export requires runtime model_chunks")
    model_config = engine.handle._extras.get("model_cfg")
    if model_config is None:
        raise RuntimeError("mlite online HF export requires runtime model_cfg")
    parallel_state = engine.handle._parallel_state
    if parallel_state is None:
        raise RuntimeError("mlite online HF export requires runtime parallel state")

    hf_path = _hf_save_path(engine.checkpoint_config, global_step)
    log(f"exporting HuggingFace checkpoint to {hf_path}", rank=0)
    start_time = time.monotonic()
    save_hf_weights(model_chunks, hf_path, model_config, parallel_state)
    _copy_hf_metadata(engine, hf_path)
    log(
        f"HuggingFace checkpoint saved to {hf_path} "
        f"({time.monotonic() - start_time:.1f}s)",
        rank=0,
    )


def _uses_rank_local_checkpoint(engine: Any) -> bool:
    return engine.policy_config.model_arch == "welmv4_moe"


def _write_latest_marker(save_path: str, global_step: int) -> None:
    if _rank() != 0:
        return
    os.makedirs(save_path, exist_ok=True)
    marker = os.path.join(save_path, _LATEST_CHECKPOINT_MARKER)
    marker_tmp = f"{marker}.tmp"
    with open(marker_tmp, "w", encoding="utf-8") as stream:
        stream.write(str(global_step))
    os.replace(marker_tmp, marker)


def _current_rank_local_topology(engine: Any) -> Dict[str, int]:
    dist_config = engine.dist_config
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        if dist_config.nnodes <= 0:
            raise RuntimeError(
                "rank-local checkpoint topology requires initialized distributed "
                "state or a positive dist_config.nnodes"
            )
        world_size = dist_config.num_gpus_per_node * dist_config.nnodes
    return {
        "world_size": world_size,
        "tp": dist_config.tensor_model_parallel_size,
        "ep": dist_config.expert_model_parallel_size,
        "etp": dist_config.expert_tensor_parallel_size,
        "pp": dist_config.pipeline_model_parallel_size,
        "cp": dist_config.context_parallel_size,
    }


def _save_rank_local_topology(engine: Any, step_path: str) -> None:
    if _rank() != 0:
        return
    topology_path = os.path.join(step_path, _RANK_LOCAL_TOPOLOGY)
    with open(topology_path, "w", encoding="utf-8") as stream:
        json.dump(_current_rank_local_topology(engine), stream, sort_keys=True)


def _validate_rank_local_topology(engine: Any, step_path: str) -> None:
    topology_path = os.path.join(step_path, _RANK_LOCAL_TOPOLOGY)
    payload: list[Any] = [None, None]
    if not dist.is_initialized() or _rank() == 0:
        try:
            with open(topology_path, encoding="utf-8") as stream:
                payload[0] = json.load(stream)
        except Exception as exc:
            payload[1] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise RuntimeError(f"failed to load mlite rank-local topology: {payload[1]}")
    current = _current_rank_local_topology(engine)
    if payload[0] != current:
        raise RuntimeError(
            "mlite rank-local checkpoint topology mismatch: "
            f"saved={payload[0]}, current={current}"
        )


def _save_rank_local_checkpoint(engine: Any, save_path: str, global_step: int) -> None:
    step_path = os.path.join(save_path, f"step_{global_step}")
    engine.runtime.save_checkpoint(
        engine.handle,
        step_path,
        step=global_step,
        use_dcp=False,
    )
    if dist.is_initialized():
        dist.barrier()
    if _rank() == 0:
        torch.save(
            engine.handle._lr_scheduler.state_dict(),
            os.path.join(step_path, _LR_SCHEDULER_STATE),
        )
    _save_rank_local_topology(engine, step_path)
    if engine.checkpoint_config.convert_mcore_to_hf_online:
        _save_hf_checkpoint(engine, global_step)
    if dist.is_initialized():
        dist.barrier()
    _write_latest_marker(save_path, global_step)
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(engine, global_step: int, dataloader=None) -> None:
    del dataloader
    save_path = engine.checkpoint_config.save_ckpt_path
    if not save_path:
        raise ValueError("checkpoint.save_ckpt_path is required")
    if _uses_rank_local_checkpoint(engine):
        _save_rank_local_checkpoint(engine, save_path, global_step)
        return
    engine.runtime.save_checkpoint(
        engine.handle,
        save_path,
        step=global_step,
        use_dcp=True,
        save_optimizer=not engine.checkpoint_config.no_save_optim,
    )
    step_path = os.path.join(save_path, f"step_{global_step}")
    if _rank() == 0:
        if not engine.checkpoint_config.no_save_optim:
            torch.save(
                engine.handle._lr_scheduler.state_dict(),
                os.path.join(step_path, _LR_SCHEDULER_STATE),
            )
    if engine.checkpoint_config.convert_mcore_to_hf_online:
        _save_hf_checkpoint(engine, global_step)
    _write_latest_marker(save_path, global_step)
    if dist.is_initialized():
        dist.barrier()


def _resolve_rank_local_resume_path(path: str) -> tuple[int, str]:
    basename = os.path.basename(os.path.normpath(path))
    if basename.startswith("step_"):
        return int(basename.removeprefix("step_")), path
    marker = os.path.join(path, _LATEST_CHECKPOINT_MARKER)
    payload: list[Any] = [None, None]
    if not dist.is_initialized() or _rank() == 0:
        try:
            if not os.path.isfile(marker):
                raise FileNotFoundError(f"mlite checkpoint marker does not exist: {marker}")
            with open(marker, encoding="utf-8") as stream:
                step = int(stream.read().strip())
            step_path = os.path.join(path, f"step_{step}")
            if not os.path.isdir(step_path):
                raise FileNotFoundError(
                    "mlite checkpoint marker points to a missing directory: "
                    f"{step_path}"
                )
            payload[0] = step
        except Exception as exc:
            payload[1] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        if str(payload[1]).startswith("FileNotFoundError:"):
            raise FileNotFoundError(payload[1])
        raise RuntimeError(f"failed to resolve mlite checkpoint marker: {payload[1]}")
    step = int(payload[0])
    return step, os.path.join(path, f"step_{step}")


def _load_rank_local_checkpoint(engine: Any, path: str) -> int:
    expected_step, step_path = _resolve_rank_local_resume_path(path)
    _validate_rank_local_topology(engine, step_path)
    step = int(engine.runtime.load_checkpoint(
        engine.handle,
        step_path,
        use_dcp=False,
    ))
    if step != expected_step:
        raise RuntimeError(
            f"mlite checkpoint step mismatch: marker={expected_step}, payload={step}"
        )
    engine.optimizer.reload_model_params()
    scheduler_path = os.path.join(step_path, _LR_SCHEDULER_STATE)
    if not os.path.exists(scheduler_path):
        raise FileNotFoundError(f"mlite checkpoint is missing scheduler state: {scheduler_path}")
    scheduler_state = torch.load(
        scheduler_path,
        map_location="cpu",
        weights_only=False,
    )
    engine.handle._lr_scheduler.load_state_dict(scheduler_state)
    if dist.is_initialized():
        dist.barrier()
    return step


def load_checkpoint(engine, path: str) -> int:
    if _uses_rank_local_checkpoint(engine):
        return _load_rank_local_checkpoint(engine, path)
    step = int(
        engine.runtime.load_checkpoint(
            engine.handle,
            path,
            use_dcp=True,
            load_optimizer=not engine.checkpoint_config.no_load_optim,
        )
    )
    if engine.checkpoint_config.no_load_optim:
        return step
    step_path = (
        path if os.path.basename(os.path.normpath(path)) == f"step_{step}" else
        os.path.join(path, f"step_{step}")
    )
    scheduler_path = os.path.join(step_path, _LR_SCHEDULER_STATE)
    if not os.path.exists(scheduler_path):
        raise FileNotFoundError(f"mlite checkpoint is missing scheduler state: {scheduler_path}")
    scheduler_state = torch.load(
        scheduler_path,
        map_location="cpu",
        weights_only=False,
    )
    engine.handle._lr_scheduler.load_state_dict(scheduler_state)
    return step
