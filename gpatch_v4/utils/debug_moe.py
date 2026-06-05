"""Debug hooks for MoE expert token distribution observation.

Installs opt-in monkey patches that, together, accumulate per-layer per-expert
token routing counts during RL or finetune training and dump them to disk:

1. ``ForwardStepMixin._update_policy``: wrapped to advance a per-process PPO step
   counter at the start of every PPO update and to flush the step's collected
   counts at the end.
2. ``ForwardStepMixin.rl_forward_step``: wrapped to advance a per-process micro
   batch counter at every micro batch forward.
3. ``ForwardStepMixin._finetune_step``: wrapped to advance the step counter at
   the start of every finetune train step and to flush the collected counts at
   the end.
4. ``ForwardStepMixin._finetune_func``: wrapped to advance the micro batch
   counter at every finetune micro batch forward.
5. ``megatron.core.transformer.moe.router.TopKRouter.forward``: wrapped to
   accumulate ``routing_map.sum(dim=0)`` per ``(step, mbs, layer)`` after
   reducing across the TP/CP token dimension and the expert-data-parallel group.

Notes
-----
- Under ``recompute_granularity=full`` the router forward is executed twice per
  micro batch (forward + backward re-execute), so the recorded counts are 2x the
  true value. We accumulate (``+=``) so the recompute factor stays predictable
  and emit ``recompute_factor`` in the dumped JSONL meta for downstream
  normalization.
- Per-step JSONL dump is written by every PP stage's representative rank
  (``tp == cp == 0`` and ``dp == 0``) into a per-PP file
  ``{dump_dir}/step_{step}/pp{pp_rank}.jsonl``; concatenating these files across
  PP ranks recovers the full layer set. Per-PP-stage stdout summary is emitted
  by the representative rank of each PP stage (``tp == cp == 0``).
- Only the alltoall dispatcher path is supported (which matches the current
  default ``moe_token_dispatcher_type="alltoall"``). Other dispatchers may
  produce a differently shaped ``routing_map`` and the reduce groups would
  need to change accordingly.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist

_INSTALLED = False
_INSTALL_LOCK = threading.Lock()

# Module-level config set by ``install`` and consumed by the patched callables.
_DUMP_DIR: Path = Path("debug-tmp/moe_dist")
# -1 means dump every step; N>0 means dump only the first N PPO steps.
# 0 is reserved for "disabled" and ``install`` is not expected to be called
# in that case (the caller in ``BaseActor.init`` gates on ``!= 0``).
_MAX_STEPS: int = -1

# ----- shared mutable state -----------------------------------------------------


class _DebugState:
    """Container for per-process debug counters and the MoE collector."""
    def __init__(self) -> None:
        self.step_id: int = 0
        # Incremented at every call to the rl_forward_step inner func
        # (= once per micro batch forward). Reset on every _update_policy entry.
        self.mbs_id: int = -1
        # collector[(step_id, mbs_id, layer_number)] = LongTensor[num_experts]
        self.collector: Dict[Tuple[int, int, int], torch.Tensor] = {}
        # How many times each (step, mbs, layer) bucket has been observed -
        # useful to confirm the recompute factor at analysis time.
        self.observe_count: Dict[Tuple[int, int, int], int] = defaultdict(int)


_STATE = _DebugState()


def _should_dump(step_id: int) -> bool:
    """Whether the given PPO step should be observed and dumped."""
    return _MAX_STEPS == -1 or step_id <= _MAX_STEPS


# ----- MoE token distribution hook ---------------------------------------------


def _install_step_counter_hook() -> None:
    """Wrap RL and finetune forward paths to track PPO/train steps and micro-batches.

    RL path:
      ``_update_policy`` advances the per-process PPO step counter and triggers
      the per-step JSONL flush at the end.
      ``rl_forward_step`` advances the micro-batch counter.

    Finetune path:
      ``_finetune_step`` advances the step counter and triggers flush.
      ``_finetune_func`` advances the micro-batch counter.
    """
    from gpatch_v4.training_backend.megatron_backend import mixin as _mixin
    from gpatch_v4.utils.common_utils import log

    # ---- RL path hooks ----

    orig_update = _mixin.ForwardStepMixin._update_policy

    def patched_update_policy(self, *args, **kwargs):
        _STATE.step_id += 1
        _STATE.mbs_id = -1

        try:
            return orig_update(self, *args, **kwargs)
        finally:
            if _should_dump(_STATE.step_id):
                try:
                    _flush_moe_collector_for_step(_STATE.step_id)
                except Exception as exc:  # noqa: BLE001 - debug code must not crash training
                    log(f"[MOE_DIST] flush failed for step={_STATE.step_id}: {exc}", rank=0)

    _mixin.ForwardStepMixin._update_policy = patched_update_policy

    orig_rl = _mixin.ForwardStepMixin.rl_forward_step

    def patched_rl_forward_step(self, *args, **kwargs):
        inner = orig_rl(self, *args, **kwargs)

        def wrapped(*inner_args, **inner_kwargs):
            _STATE.mbs_id += 1
            return inner(*inner_args, **inner_kwargs)

        return wrapped

    _mixin.ForwardStepMixin.rl_forward_step = patched_rl_forward_step

    # ---- Finetune path hooks ----

    orig_finetune_step = _mixin.ForwardStepMixin._finetune_step

    def patched_finetune_step(self, *args, **kwargs):
        _STATE.step_id += 1
        _STATE.mbs_id = -1

        try:
            return orig_finetune_step(self, *args, **kwargs)
        finally:
            if _should_dump(_STATE.step_id):
                try:
                    _flush_moe_collector_for_step(_STATE.step_id)
                except Exception as exc:  # noqa: BLE001 - debug code must not crash training
                    log(f"[MOE_DIST] flush failed for step={_STATE.step_id}: {exc}", rank=0)

    _mixin.ForwardStepMixin._finetune_step = patched_finetune_step

    orig_finetune_func = _mixin.ForwardStepMixin._finetune_func

    def patched_finetune_func(self, *args, **kwargs):
        inner = orig_finetune_func(self, *args, **kwargs)

        def wrapped(*inner_args, **inner_kwargs):
            _STATE.mbs_id += 1
            return inner(*inner_args, **inner_kwargs)

        return wrapped

    _mixin.ForwardStepMixin._finetune_func = patched_finetune_func


def _install_moe_router_hook() -> None:
    """Wrap ``TopKRouter.forward`` to accumulate per-expert token counts.

    The routed token counts are reduced across the TP/CP token dimension via the
    router's own ``tp_cp_group``, and across data parallel via the expert data
    parallel group, yielding the full per-expert distribution for that PP stage.
    Different PP stages hold disjoint layers, so no PP reduction is required.

    Only the alltoall dispatcher path is supported (which matches the current
    default ``moe_token_dispatcher_type="alltoall"``). Other dispatchers may
    produce a differently shaped ``routing_map`` and the reduce groups would
    need to change accordingly.
    """
    from megatron.core.transformer.moe import router as _router

    orig_forward = _router.TopKRouter.forward

    # Forward arbitrary args/kwargs to stay compatible with downstream forks
    # whose ``TopKRouter.forward`` signature differs from upstream (e.g.
    # ``wxdev`` Megatron-LM adds a ``padding_mask`` kwarg).
    def patched_forward(self, *args, **kwargs):
        scores, routing_map = orig_forward(self, *args, **kwargs)

        # Fast path: skip when the current PPO step is past the dump window,
        # and skip eval-only forwards. Recompute reruns also enter here so the
        # observation count doubles under recompute_granularity=full (the
        # per-step flush normalizes by the recorded observe_count factor).
        if not _should_dump(_STATE.step_id) or not torch.is_grad_enabled():
            return scores, routing_map

        try:
            with torch.no_grad():
                # routing_map: [num_local_tokens, num_experts] (global expert dim)
                local = routing_map.sum(dim=0).long()

                if getattr(self, "tp_cp_group", None) is not None and self.tp_cp_group.size() > 1:
                    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=self.tp_cp_group)

                from megatron.core import mpu
                try:
                    edp_group = mpu.get_expert_data_parallel_group()
                    if edp_group is not None and edp_group.size() > 1:
                        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=edp_group)
                except Exception:
                    # Older mcore versions only expose get_data_modulo_expert_parallel_group.
                    edp_group = mpu.get_data_modulo_expert_parallel_group()
                    if edp_group is not None and edp_group.size() > 1:
                        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=edp_group)

                key = (_STATE.step_id, _STATE.mbs_id, int(self.layer_number))
                cpu_local = local.detach().cpu()
                if key in _STATE.collector:
                    _STATE.collector[key] = _STATE.collector[key] + cpu_local
                else:
                    _STATE.collector[key] = cpu_local
                _STATE.observe_count[key] += 1
        except Exception:
            # Never break training because of debug instrumentation.
            pass

        return scores, routing_map

    _router.TopKRouter.forward = patched_forward


# ----- per-step dump ------------------------------------------------------------


def _flush_moe_collector_for_step(step_id: int) -> None:
    """Persist (and summarize) the MoE collector for a single PPO step.

    Persistence happens once per PP stage on its representative rank
    (``tp == cp == 0`` and ``dp == 0``); the file name carries ``pp{rank}``
    so writes from different PP stages do not clobber each other and the
    full layer set (PP0 + PP1 + ...) is recoverable by concatenation.
    Stdout summary is emitted by every PP-stage representative rank
    (``tp == cp == 0``) so all layers stay visible in the log.
    """
    from gpatch_v4.utils.common_utils import log

    keys = [k for k in list(_STATE.collector.keys()) if k[0] == step_id]
    if not keys:
        return

    by_layer: Dict[int, List[Tuple[int, torch.Tensor, int]]] = defaultdict(list)
    for key in keys:
        _, mbs_idx, layer_num = key
        by_layer[layer_num].append((mbs_idx, _STATE.collector[key], _STATE.observe_count[key]))

    try:
        from megatron.core import mpu
        tp_rank = mpu.get_tensor_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        try:
            dp_rank = mpu.get_data_parallel_rank()
        except Exception:
            dp_rank = 0
    except Exception:
        tp_rank = cp_rank = pp_rank = dp_rank = 0
        pp_size = 1

    is_pp_repr = (tp_rank == 0 and cp_rank == 0)
    # All DP replicas of a PP stage hold an identical post-allreduce copy of
    # the routing counts, so picking dp==0 is sufficient and avoids each DP
    # replica writing the same payload.
    is_writer = is_pp_repr and dp_rank == 0
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    if is_pp_repr:
        for layer_num in sorted(by_layer.keys()):
            entries = by_layer[layer_num]
            stacked = torch.stack([t for _, t, _ in entries], dim=0).float()
            obs_factor = max(1, max(c for _, _, c in entries))
            per_step = stacked.sum(dim=0) / obs_factor
            num_experts = per_step.numel()
            max_tok = per_step.max().item()
            min_tok = per_step.min().item()
            std = per_step.std(unbiased=False).item()
            active = int((per_step > 0).sum().item())
            log(
                f"[MOE_DIST] step={step_id} pp={pp_rank}/{pp_size} layer={layer_num} "
                f"max_tok={max_tok:.0f} min_tok={min_tok:.0f} std={std:.2f} "
                f"active_experts={active}/{num_experts} recompute_factor={obs_factor}",
                rank=global_rank,
            )

    if is_writer:
        out_dir = _DUMP_DIR / f"step_{step_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pp{pp_rank}.jsonl"
        with out_path.open("w") as f:
            for key in sorted(keys):
                step, mbs_idx, layer_num = key
                t = _STATE.collector[key]
                obs = _STATE.observe_count[key]
                record = {
                    "step": step,
                    "mbs": mbs_idx,
                    "layer": layer_num,
                    "tokens_per_expert": t.tolist(),
                    "recompute_factor": obs,
                    "pp_rank": pp_rank,
                }
                f.write(json.dumps(record) + "\n")

    for key in keys:
        _STATE.collector.pop(key, None)
        _STATE.observe_count.pop(key, None)


# ----- public entry point -------------------------------------------------------


def install(config) -> None:
    """Install the MoE token distribution hooks based on the actor config.

    No-op when the config has no ``debug`` block or when
    ``debug.debug_dump_first_n_ppo_step_moe_token_dist`` is ``0``.
    When enabled, requires the megatron training backend.

    Works for both RL (``_update_policy`` / ``rl_forward_step``) and finetune
    (``_finetune_step`` / ``_finetune_func``) training paths.

    Parameters
    ----------
    config : object
        Inspected fields:
        ``config.debug.debug_dump_first_n_ppo_step_moe_token_dist`` (int),
        ``config.debug.debug_moe_dist_dump_dir`` (str),
        ``config.training.training_backend`` (str, must be ``"mcore"``).

    Raises
    ------
    NotImplementedError
        When requested under a non-megatron training backend; the patches
        target ``ForwardStepMixin`` and ``megatron.core...TopKRouter`` and
        have no FSDP2 counterpart.
    """
    if not hasattr(config, "debug"):
        return
    max_steps = config.debug.debug_dump_first_n_ppo_step_moe_token_dist
    if max_steps == 0:
        return
    backend = config.training.training_backend
    if backend != "mcore":
        raise NotImplementedError(
            f"debug_dump_first_n_ppo_step_moe_token_dist is only supported on "
            f"the megatron ('mcore') training backend (got training_backend={backend!r})."
        )
    _install_hooks(config.debug.debug_moe_dist_dump_dir, max_steps)


def _install_hooks(dump_dir: Union[str, Path], max_steps: int) -> None:
    """Idempotently install the step-counter and MoE router patches."""
    global _INSTALLED, _DUMP_DIR, _MAX_STEPS

    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        _DUMP_DIR = Path(dump_dir)
        _MAX_STEPS = max_steps

        _install_step_counter_hook()
        _install_moe_router_hook()
        _INSTALLED = True

        try:
            from gpatch_v4.utils.common_utils import log
            log(
                f"[MOE_DIST] installed MoE token distribution hook "
                f"(dump_dir={_DUMP_DIR}, max_steps={_MAX_STEPS})",
                rank=0,
            )
        except Exception:
            print(
                f"[MOE_DIST] installed MoE token distribution hook "
                f"(dump_dir={_DUMP_DIR}, max_steps={_MAX_STEPS})"
            )
