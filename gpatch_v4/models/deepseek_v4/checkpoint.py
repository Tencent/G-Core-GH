# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""Checkpoint load / save for DeepSeek-V4 EP+FSDP2 models.

Provides ``_load_checkpoint_hp`` and ``_save_checkpoint_hp``, bound to the
model by :func:`.hp.apply_hp` via ``types.MethodType``.
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import gc
import json
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from typing import Literal, Optional

import safetensors
import safetensors.torch as st_torch
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.tensor import DTensor, Shard

from gpatch_v4.utils import log_debug

from .fp_quantize import quant_fp4_e2m1_scale_e8m0_packed, quant_fp8_e4m3_scale_e8m0

# ---------------------------------------------------------------------------
# Meta-buffer materialization (RoPE only; other persistent buffers come from ckpt)
# ---------------------------------------------------------------------------


def _materialize_meta_buffers(model: nn.Module) -> None:
    """Re-compute non-persistent buffers left as meta tensors after load.

    Currently only handles :class:`DeepseekV4RotaryEmbedding`, which registers
    per-rope-type ``{label}_inv_freq`` / ``{label}_original_inv_freq`` buffers
    with ``persistent=False``.     Persistent buffers (``e_score_correction_bias``,
    ``tid2eid``) come from the checkpoint via :func:`_load_checkpoint_hp`.

    Fails fast on any unhandled meta buffer rather than silently zero-init.
    """
    target_device: Optional[torch.device] = None
    for p in model.parameters():
        if not p.is_meta:
            target_device = p.device
            break
    assert target_device is not None, (
        "_materialize_meta_buffers must be called after parameters are loaded"
    )

    for module in model.modules():
        meta_bufs = {n: b for n, b in module.named_buffers(recurse=False) if b.is_meta}
        if not meta_bufs:
            continue

        cls_name = type(module).__name__

        if cls_name == "DeepseekV4RotaryEmbedding":
            # `module.layer_types` is e.g. ["main", "compress"] (rope-type labels,
            # not architecture layer types). Each contributes 2 buffers.
            for label in module.layer_types:
                inv_name = f"{label}_inv_freq"
                orig_name = f"{label}_original_inv_freq"
                if inv_name not in meta_bufs and orig_name not in meta_bufs:
                    continue
                rope_type = module.rope_type[label]
                rope_init_fn = (
                    type(module).compute_default_rope_parameters
                    if rope_type == "default" else _ROPE_INIT_FUNCTIONS[rope_type]
                )
                inv_freq, _ = rope_init_fn(module.config, layer_type=label)
                inv_freq = inv_freq.to(target_device)
                if inv_name in meta_bufs:
                    module.register_buffer(inv_name, inv_freq, persistent=False)
                    meta_bufs.pop(inv_name)
                if orig_name in meta_bufs:
                    module.register_buffer(orig_name, inv_freq.clone(), persistent=False)
                    meta_bufs.pop(orig_name)

        if meta_bufs:
            raise RuntimeError(
                f"Unhandled meta buffers in {cls_name}: {list(meta_bufs.keys())}. "
                f"Add a branch to _materialize_meta_buffers() — silent zero-init is forbidden."
            )


# Lazy import to avoid pulling in transformers when only hp.py / shard helpers are needed.
def _lazy_init_rope_table() -> None:
    global _ROPE_INIT_FUNCTIONS
    if _ROPE_INIT_FUNCTIONS is None:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        _ROPE_INIT_FUNCTIONS = ROPE_INIT_FUNCTIONS


_ROPE_INIT_FUNCTIONS = None  # populated by _lazy_init_rope_table on first need

# ---------------------------------------------------------------------------
# Checkpoint loader: per-rank streaming dequant via HF's conversion machinery
# ---------------------------------------------------------------------------


def _load_checkpoint_hp(
    self: nn.Module,
    hf_path: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Stream a DSV4-Flash FP8/FP4 checkpoint into an EP-FSDP meta model.

    Each rank reads raw safetensors lazily (mmap) and drives HF's conversion
    pipeline per output FQN, then EP-slices expert weights and distributes
    onto the FSDP/EP mesh. Mirrors :func:`qwen3_5_moe.parallelize._load_checkpoint_hp`'s
    streaming pattern; the DSV4-specific complexity (key remapping, expert
    merge, FP8/FP4 dequant via ``.weight + .scale``) is delegated to HF's
    ``conversion_mapping["deepseek_v4"]`` + ``Fp8Dequantize`` + ``WeightConverter``
    machinery — we only own the streaming driver and the EP slicing / DTensor
    distribution.

    Strategy
    --------
    1. Build the loading pipeline by calling
       :meth:`FineGrainedFP8HfQuantizer.update_weight_conversions` on the
       upstream DSV4 conversion mapping. This prepends the
       ``.scale → .weight_scale_inv`` rename and injects an :class:`Fp8Dequantize`
       op into every :class:`WeightConverter` so per-block scales are folded
       in before any merge/concat collapses the per-expert structure.
    2. Open every safetensors shard lazily via ``safe_open``. For each raw
       key, compute its target FQN via :func:`rename_source_key` and append a
       lazy reader (``lambda: f.get_tensor(...)``) to the matching converter's
       ``collected_tensors``. One :class:`WeightConverter` instance is cloned
       per target FQN so collections don't bleed across layers.
    3. For each target FQN, call ``mapping.convert(...)`` to drive the chain
       (Fp8Dequantize → MergeModulelist → Concatenate, etc.). The result is
       a bf16/fp32 tensor at the model's expected layout.
    4. Dispatch on ``isinstance(target, DTensor)``:
       * DTensor → EP-slice along dim-0 for ``.experts.gate_up_proj`` /
         ``.experts.down_proj``, then ``distribute_tensor`` onto the param's
         mesh and wrap in ``nn.Parameter``.
       * Plain tensor (persistent buffer ``e_score_correction_bias`` fp32 /
         ``tid2eid`` int64) → assign directly without cast / wrap / distribute.
    5. ``self.load_state_dict(sharded_sd, strict=False, assign=True)`` and
       :func:`_materialize_meta_buffers` for non-persistent RoPE buffers.

    Memory
    ------
    Peak per rank ≈ the size of one expert-merge target tensor (~8 GB bf16
    for one layer's ``gate_up_proj`` across 256 experts). Far below the
    rank0-broadcast variant we used previously.

    Parameters
    ----------
    hf_path : str
        Path to HF checkpoint directory (``model.safetensors[.index.json]``).
    dtype : torch.dtype, default ``torch.float32``
        Target dtype for floating-point parameters. The Fp8Dequantize output
        is bf16 (E8M0 scale → hardcoded bf16); we cast to ``dtype`` after.
        Persistent integer buffers (``tid2eid``) are unaffected.
    """
    _lazy_init_rope_table()

    import time as _time
    _t0 = _time.time()
    torch.cuda.reset_peak_memory_stats()

    # Late imports: keep this module importable when transformers is absent
    # for the rest of the EP / FSDP helpers.
    from safetensors import safe_open
    from transformers import FineGrainedFP8Config
    from transformers.conversion_mapping import _build_checkpoint_conversion_mapping
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )
    from transformers.quantizers.quantizer_finegrained_fp8 import (
        FineGrainedFP8HfQuantizer,
    )

    rank = dist.get_rank()
    ep_rank = self._ep_rank
    meta_sd = self.state_dict()  # FSDP2 meta DTensors + persistent buffers
    prefix = type(self).base_model_prefix  # "model"

    # 1) Build the FP8-aware conversion pipeline (= DSV4 base mapping
    # + scale rename + Fp8Dequantize injected per WeightConverter + generic
    # fallback). HF's quantizer does exactly the right thing here.
    base_convs = list(_build_checkpoint_conversion_mapping().get("deepseek_v4", []))
    fp8_quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(dequantize=True))
    weight_mapping = fp8_quantizer.update_weight_conversions(base_convs)
    renamings = [m for m in weight_mapping if isinstance(m, WeightRenaming)]
    converters = [m for m in weight_mapping if isinstance(m, WeightConverter)]
    pattern_to_converter = {k: c for c in converters for k in c.source_patterns}

    # Append local renamings that map post-HF model state_dict names to our
    # ``_Fp32ParamHolder``-wrapped paths. HF rename runs each renaming in
    # order and chains the results (see ``rename_source_key`` in
    # transformers/core_model_loading.py), so these apply AFTER all HF base
    # renamings produced ``...sinks`` / ``...position_bias`` from the disk
    # form. Without these the rewritten key would no longer match the model's
    # current ``state_dict()`` (which now exposes ``_sink_holder.weight`` /
    # ``_position_bias_holder.weight`` instead).
    renamings.extend(
        [
            WeightRenaming(
                source_patterns=r"^(.*\.)?self_attn\.sinks$",
                target_patterns=r"\1self_attn._sink_holder.weight",
            ),
            WeightRenaming(
                source_patterns=r"^(.*\.)?self_attn\.compressor\.indexer\.position_bias$",
                target_patterns=r"\1self_attn.compressor.indexer._position_bias_holder.weight",
            ),
            WeightRenaming(
                source_patterns=r"^(.*\.)?self_attn\.compressor\.position_bias$",
                target_patterns=r"\1self_attn.compressor._position_bias_holder.weight",
            ),
        ]
    )

    _t1 = _time.time()
    if rank == 0:
        print(f"[load_checkpoint_hp] phase 1 (build pipeline): {_t1 - _t0:.2f}s", flush=True)

    # 2) Discover shards and group raw keys by target FQN.
    '''
    FQN = Fully Qualified Name，即模块的完整路径名。
    比如 model.layers.0.self_attn.q_a_proj.weight 就是一个 FQN —— 从模型根到具体参数的完整点分路径。
    相对应的是 HF checkpoint 里的 "disk key"（如 layers.0.attn.wq_a.weight），
    两者之间有 prefix 差异和结构重命名（self_attn ↔ attn、q_a_proj ↔ wq_a 等）。
    '''
    weight_map, _ = _discover_safetensor_shards(hf_path)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for disk_key, shard_file in weight_map.items():
        by_shard[shard_file].append(disk_key)

    # ``fqn_to_mapping[target_fqn] = mapping`` whose ``collected_tensors`` holds
    # lazy readers (each re-opens its shard on call — safetensors mmap is cheap
    # and the OS dedupes shared pages across reads).
    fqn_to_mapping: dict[str, WeightConverter | WeightRenaming] = {}
    skipped_keys: list[str] = []
    sharded_sd: dict[str, object] = {}
    num_base_layers = int(self.config.num_hidden_layers)
    num_mtp_layers = int(self.config.num_nextn_predict_layers)

    def _make_reader(shard_path: str, disk_key: str):
        def _read() -> torch.Tensor:
            with safe_open(shard_path, framework="pt") as f:
                return f.get_tensor(disk_key)

        return _read

    # 3) Single pass: route each raw key into the right converter bucket with a
    # lazy reader. We do NOT keep safe_open contexts open across the loop —
    # each reader re-opens its shard at materialization time.
    for shard_file in sorted(by_shard):
        shard_path = os.path.join(hf_path, shard_file)
        for disk_key in by_shard[shard_file]:
            logical_disk_key = disk_key
            _is_mtp_key = (num_mtp_layers > 0 and disk_key.startswith("mtp."))
            if _is_mtp_key:
                mtp_m = re.match(r"^mtp\.(\d+)\.(.+)$", disk_key)
                if mtp_m is not None:
                    mtp_depth = int(mtp_m.group(1))
                    if mtp_depth < num_mtp_layers:
                        logical_disk_key = (
                            f"layers.{num_base_layers + mtp_depth}.{mtp_m.group(2)}"
                        )
                        # MTP-only top-level attributes that HF conversion
                        # mapping doesn't handle (no renaming/converter rule):
                        # disk "hc_head_fn" → model "hc_head.hc_fn", etc.
                        # These exist as direct layer attrs on disk but as
                        # nested submodule attrs in the model (DeepseekV4HyperHead).
                        logical_disk_key = re.sub(
                            r"\.hc_head_(fn|base|scale)$",
                            r".hc_head.hc_\1",
                            logical_disk_key,
                        )
                    else:
                        _is_mtp_key = False  # out-of-range depth, treat as non-MTP

            # For MTP disk keys, pass prefix=None to skip rename_source_key's
            # step 3 "meta_sd prefix heuristic" — that logic fails when the
            # model is truncated (meta_sd has no model.layers.{base+d}.*
            # entries). We manually add the prefix + do MTP remap below.
            renamed_key, source_pattern = rename_source_key(
                logical_disk_key,
                renamings,
                converters,
                None if _is_mtp_key else prefix,
                None if _is_mtp_key else meta_sd,
            )

            if _is_mtp_key:
                # Manual prefix + MTP remap: renamed_key is e.g.
                # "layers.4.self_attn.kv_proj.weight" (no "model." prefix).
                # Strip any leading "layers." (the converter may or may not
                # output it), normalise to "layers.{N}.X", then remap.
                _mtp_check = renamed_key
                if _mtp_check.startswith("model."):
                    _mtp_check = _mtp_check[len("model."):]
                layer_m = re.match(r"^layers\.(\d+)\.(.+)$", _mtp_check)
                if layer_m is not None:
                    layer_idx = int(layer_m.group(1))
                    if num_base_layers <= layer_idx < (num_base_layers + num_mtp_layers):
                        renamed_key = (
                            f"mtp.layers.{layer_idx - num_base_layers}.{layer_m.group(2)}"
                        )

            if renamed_key not in meta_sd:
                skipped_keys.append(disk_key)
                continue

            if source_pattern is not None:
                # Clone the converter per target so collected_tensors stays scoped.
                mapping = fqn_to_mapping.setdefault(
                    renamed_key,
                    deepcopy(pattern_to_converter[source_pattern]),
                )
            else:
                mapping = fqn_to_mapping.setdefault(
                    renamed_key,
                    WeightRenaming(logical_disk_key, renamed_key),
                )
                source_pattern = logical_disk_key

            mapping.add_tensor(
                renamed_key,
                disk_key,
                source_pattern,
                _make_reader(shard_path, disk_key),
            )
    '''
    [load_checkpoint_hp] phase 1 (build pipeline): 0.02s
    [load_checkpoint_hp] phase 2-3 (discover shards + route keys): 5.00s (69187 raw keys → 29 targets, 67616 skipped)
    [load_checkpoint_hp] phase 4 (convert + EP slice + distribute): 107.64s (29 tensors materialized)
    [load_checkpoint_hp] WARN: skipped 67616 keys not in model: layers.1.attn.attn_sink, layers.1.attn.kv_norm.weight, layers.1.attn.q_norm.weight, layers.1.attn.wkv.scale, layers.1.attn.wkv.weight, layers.1.attn.wo_a.scale, layers.1.attn.wo_a.weight, layers.1.attn.wo_b.scale... (+67608 more)
    [load_checkpoint_hp] phase 5 (load_state_dict assign): 0.01s
    [load_checkpoint_hp] phase 6 (materialize buffers): 0.00s
    [load_checkpoint_hp] TOTAL: 112.67s
    '''
    _t2 = _time.time()
    if rank == 0:
        print(
            f"[load_checkpoint_hp] phase 2-3 (discover shards + route keys): {_t2 - _t1:.2f}s "
            f"({len(weight_map)} raw keys → {len(fqn_to_mapping)} targets, "
            f"{len(skipped_keys)} skipped)",
            flush=True,
        )

    # --- MTP routing audit (phase 2-3) -----------------------------------
    # When ``num_mtp_layers > 0`` the model has an active MTP submodule
    # (see DeepseekV4ForCausalLM.__init__: it clears
    # ``_keys_to_ignore_on_load_unexpected`` so mtp.* keys are NOT silently
    # dropped). We must guarantee:
    #   1. every disk key ``mtp.{d}.*`` with ``d < num_mtp_layers`` was
    #      routed (not in skipped_keys);
    #   2. routing produced at least one target FQN under ``mtp.layers.``.
    # Both branches fail loudly — silent MTP-not-loaded is the BUG we are
    # guarding against.
    if num_mtp_layers > 0:
        _disk_mtp_re = re.compile(r"^mtp\.(\d+)\.")
        disk_mtp_keys = [
            k for k in weight_map
            if (m := _disk_mtp_re.match(k)) is not None and int(m.group(1)) < num_mtp_layers
        ]
        skipped_mtp = [
            k for k in skipped_keys
            if (m := _disk_mtp_re.match(k)) is not None and int(m.group(1)) < num_mtp_layers
        ]
        routed_mtp_targets = [t for t in fqn_to_mapping if t.startswith("mtp.layers.")]
        assert not skipped_mtp, (
            f"MTP enabled (num_mtp_layers={num_mtp_layers}) but {len(skipped_mtp)} "
            f"in-range mtp.* disk keys were not routed to a model FQN — remap is "
            f"broken. First few: {sorted(skipped_mtp)[:8]}"
        )
        assert routed_mtp_targets, (
            f"MTP enabled (num_mtp_layers={num_mtp_layers}) but 0 target FQNs under "
            f"mtp.layers.* were created from {len(disk_mtp_keys)} disk mtp.* keys — "
            f"remap is broken."
        )
        if rank == 0:
            print(
                f"[load_checkpoint_hp] MTP routing: {len(disk_mtp_keys)} disk mtp.* keys "
                f"→ {len(routed_mtp_targets)} mtp.layers.* targets "
                f"(num_mtp_layers={num_mtp_layers})",
                flush=True,
            )

    # 4) Distributed convert via wave-based fp32 zero buffer + all_reduce.
    #
    # Targets split two ways:
    #   - Parameters (FSDP DTensor target): always float (nn.Parameter requires
    #     it). Go through wave path with fp32 buffer + fp32 all_reduce SUM,
    #     cast to caller `dtype` AFTER reduce, then EP slice + distribute_tensor.
    #     fp32 reduce mirrors FSDP `MixedPrecisionPolicy(reduce_dtype=fp32)` +
    #     `_assert_fp32_master` (see hp.py) — the canonical contract.
    #   - Persistent buffers (plain tensor target, e_score_correction_bias fp32
    #     + tid2eid int64): tiny (<10 KB per layer), every rank just runs
    #     convert independently — no comm, no distribution.
    world_size = dist.get_world_size()
    FP32_BYTES = 4
    # NCCL all_reduce working set is roughly 2× payload size. Cap the per-call
    # payload so a single oversized param can't blow GPU memory: any param
    # larger than this gets its all_reduce split into multiple chunks, each
    # ≤ MAX_ALLREDUCE_BYTES. DSV4-Flash's `gate_up_proj` is exactly 16 GiB fp32
    # (256 expert × 7168 × 4096) — it fits in one chunk with this cap. Future
    # bigger experts get chunked automatically.
    MAX_ALLREDUCE_BYTES = 16 * 1024**3
    CHUNK_NUMEL = MAX_ALLREDUCE_BYTES // FP32_BYTES

    param_tfqns: list[str] = []
    buffer_tfqns: list[str] = []
    for tfqn in fqn_to_mapping:
        (param_tfqns if isinstance(meta_sd[tfqn], DTensor) else buffer_tfqns).append(tfqn)

    # 4a) Pre-compute full (pre-EP-slice, pre-FSDP-shard) shape for each param.
    param_shapes: dict[str, tuple[int, ...]] = {}
    for tfqn in param_tfqns:
        target = meta_sd[tfqn]
        if (".experts.gate_up_proj" in tfqn) or (".experts.down_proj" in tfqn):
            param_shapes[tfqn] = (target.shape[0] * self._ep_size, *target.shape[1:])
        else:
            param_shapes[tfqn] = tuple(target.shape)

    # 4b) Sort params by (-numel, name) so that within each wave the per-rank
    # convert workload is roughly balanced (same-size params cluster) and big
    # tensors get spread evenly across `world_size` ranks via round-robin
    # ownership. Name as secondary key ensures determinism across ranks.
    def _numel(tfqn: str) -> int:
        n = 1
        for d in param_shapes[tfqn]:
            n *= d
        return n

    sorted_params = sorted(param_tfqns, key=lambda t: (-_numel(t), t))

    # Wave assembly: count-only chunking — each wave holds up to `world_size`
    # params (last wave may be partial). What matters is that every rank has at
    # most one item to convert per wave, exposing `world_size`-way Phase A
    # parallelism for the heavy expert tensors. Phase B does per-param all_reduce
    # (chunked at `MAX_ALLREDUCE_BYTES` for oversized params), so we do NOT need
    # a per-wave size cap.
    waves: list[list[str]] = [
        sorted_params[i:i + world_size] for i in range(0, len(sorted_params), world_size)
    ]

    if rank == 0:
        print(
            f"[load_checkpoint_hp] phase 4 plan: {len(param_tfqns)} params in "
            f"{len(waves)} fp32 waves (≤{world_size} tfqns each; per-param all_reduce "
            f"chunked at ≤{MAX_ALLREDUCE_BYTES >> 30} GiB), "
            f"{len(buffer_tfqns)} buffers; world_size={world_size}",
            flush=True,
        )

    # 4c) Buffer path: every rank runs convert independently (tiny + cheap).
    # Preserve on-disk dtype (e_score_correction_bias fp32, tid2eid int64) —
    # buffers are not parameters and shouldn't be down-cast by the caller's
    # `dtype` arg.
    for tfqn in buffer_tfqns:
        realized = fqn_to_mapping[tfqn].convert(
            tfqn,
            model=self,
            config=self.config,
            hf_quantizer=None,
        )
        assert set(realized.keys()) == {
            tfqn
        }, (f"expected 1:1 converter output for {tfqn}, got {list(realized.keys())}")
        t = realized[tfqn]
        if isinstance(t, list):
            t = t[0]
        sharded_sd[tfqn] = t.contiguous().cuda()

    # 4d) Param wave loop in two phases per wave:
    #
    #   Phase A (parallel convert): each rank converts ITS owned params to fp32
    #     CPU tensors. With at most `world_size` params per wave, every rank
    #     does at most one convert per wave — so all `world_size` CPUs are busy
    #     simultaneously, which is the only way to amortize the per-tensor
    #     convert cost (one expert tensor's FP4→fp32 dequant is the long pole).
    #
    #   Phase B (per-param chunked all_reduce, GPU-peak-bounded): allocate a
    #     fixed-size `CHUNK_NUMEL` fp32 GPU buffer reused across all params; for
    #     each param, owner streams its CPU-staged data through the buffer in
    #     chunks of `CHUNK_NUMEL`, all_reduce per chunk (NCCL working set
    #     bounded by ~MAX_ALLREDUCE_BYTES), and every rank copies just the
    #     part of each chunk that falls in its own EP+FSDP local row range
    #     into a small per-param `local_dst_gpu`. After the chunk loop,
    #     `local_dst_gpu` holds exactly this rank's shard, which we wrap via
    #     `DTensor.from_local` — no further sub-mesh comm.
    #
    #     GPU peak per rank ≈ CHUNK + max(local shard size) — *not* the full
    #     param; safe even when a single param > 16 GiB (e.g. DSV4-Pro
    #     gate_up_proj fp32 ≈ 67.5 GiB would have blown 96 GiB H20 otherwise).
    gpu_chunk_buf = torch.empty(CHUNK_NUMEL, dtype=torch.float32, device="cuda")

    def _local_outer_layout(target_t: torch.Tensor, is_expert: bool) -> tuple[int, int, int]:
        """Return ``(real_lo, real_hi, local_outer)`` for this rank's local shard,
        mirroring ``torch.chunk(t, mesh_size, dim=0)`` + ``new_empty(0)`` padding
        (FSDP2 ``_chunk_with_empty``). Ranks past ``ceil(d0 / chunk_outer)`` get
        ``local_outer == 0``; without this, ``optimizer.step()`` later fails as
        ``output with shape [k] doesn't match the broadcast shape [0]``.
        """
        mesh = target_t.device_mesh
        in_mesh = mesh.get_local_rank()
        mesh_size = mesh.size()
        if is_expert:
            d0 = target_t.shape[0]  # num_experts / ep_size
            base_offset = self._ep_rank * d0
        else:
            d0 = target_t.shape[0]
            base_offset = 0

        chunk_outer = (d0 + mesh_size - 1) // mesh_size if d0 > 0 else 0
        if chunk_outer == 0:
            return base_offset, base_offset, 0
        n_full_chunks = (d0 + chunk_outer - 1) // chunk_outer
        if in_mesh < n_full_chunks:
            lo = in_mesh * chunk_outer
            hi = min(lo + chunk_outer, d0)
            return base_offset + lo, base_offset + hi, hi - lo
        return base_offset + d0, base_offset + d0, 0

    _t_convert_total = 0.0
    _t_allreduce_total = 0.0
    for wave_idx, wave in enumerate(waves):
        # ---------- Phase A: per-rank parallel convert to CPU ----------
        staged_cpu: dict[str, torch.Tensor] = {}
        _tw0 = _time.time()
        assert len(wave) <= world_size
        for i, tfqn in enumerate(wave):
            if i % world_size != rank:
                continue
            realized = fqn_to_mapping[tfqn].convert(
                tfqn,
                model=self,
                config=self.config,
                hf_quantizer=None,  # dequant already injected
            )
            assert set(realized.keys()) == {
                tfqn
            }, (f"expected 1:1 converter output for {tfqn}, got {list(realized.keys())}")
            t = realized[tfqn]
            if isinstance(t, list):
                t = t[0]
            # Cast to fp32 on CPU; stays on CPU until Phase B streams it to GPU.
            # Fp8Dequantize emits bf16 → fp32 is no precision loss (bf16 ⊂ fp32).
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            assert tuple(t.shape) == param_shapes[tfqn], (
                f"shape mismatch on {tfqn}: convert {tuple(t.shape)} vs expected {param_shapes[tfqn]}"
            )
            staged_cpu[tfqn] = t.contiguous()
        _tw1 = _time.time()
        _t_convert_total += _tw1 - _tw0

        # ---------- Phase B: chunked all_reduce + per-rank overlap copy ----------
        max_n_w = max(_numel(t) for t in wave)
        max_chunks = (max_n_w + CHUNK_NUMEL - 1) // CHUNK_NUMEL
        for i, tfqn in enumerate(wave):
            owner = i % world_size
            target = meta_sd[tfqn]
            n = _numel(tfqn)
            is_expert = (".experts.gate_up_proj" in tfqn or ".experts.down_proj" in tfqn)

            # Snapshot the meta DTensor layout; any drift in the synthesized
            # replacement is caught by the post-build assert. `to_local().shape`
            # routes through PyTorch's own chunking, so it matches FSDP2 exactly.
            target_global_shape = tuple(target.shape)
            target_global_stride = target.stride()
            target_local_shape = tuple(target.to_local().shape)
            target_placements = target.placements
            target_device_mesh = target.device_mesh

            real_lo, real_hi, local_outer = _local_outer_layout(target, is_expert)
            trailing_numel = 1
            for d in target.shape[1:]:
                trailing_numel *= int(d)
            real_lo_flat = real_lo * trailing_numel
            real_hi_flat = real_hi * trailing_numel

            local_shape = (local_outer, ) + tuple(int(d) for d in target.shape[1:])
            local_dst = torch.zeros(local_shape, dtype=torch.float32, device="cuda")
            local_dst_flat = local_dst.view(-1)

            # Owner streams from its CPU-staged tensor; non-owners contribute zeros.
            src_cpu_flat = (staged_cpu.pop(tfqn).view(-1) if rank == owner else None)

            for c0 in range(0, n, CHUNK_NUMEL):
                c1 = min(c0 + CHUNK_NUMEL, n)
                chunk_size = c1 - c0
                sub = gpu_chunk_buf[:chunk_size]
                if rank == owner:
                    sub.copy_(src_cpu_flat[c0:c1])  # H2D for owner only
                else:
                    sub.zero_()
                dist.all_reduce(sub, op=dist.ReduceOp.SUM)  # everyone now has owner's chunk

                # Padding rows (beyond real_hi) stay zero. When local_outer=0
                # the if-guard never fires; local_dst remains `(0, ...)`.
                overlap_lo = max(c0, real_lo_flat)
                overlap_hi = min(c1, real_hi_flat)
                if overlap_hi > overlap_lo:
                    local_dst_flat[overlap_lo - real_lo_flat:overlap_hi - real_lo_flat].copy_(
                        sub[overlap_lo - c0:overlap_hi - c0]
                    )

            # Pass global shape/stride explicitly so from_local doesn't infer
            # them from the uneven per-rank sizes.
            if local_dst.dtype != dtype:
                local_dst = local_dst.to(dtype)
            sharded = DTensor.from_local(
                local_dst,
                target_device_mesh,
                target_placements,
                shape=torch.Size(target_global_shape),
                stride=target_global_stride,
            )

            new_local_shape = tuple(sharded.to_local().shape)
            assert new_local_shape == target_local_shape, (
                f"load layout drift on {tfqn}: local shape "
                f"{new_local_shape} != target {target_local_shape} "
                f"(rank={rank}, global={target_global_shape})"
            )
            assert tuple(sharded.shape) == target_global_shape, (
                f"load layout drift on {tfqn}: global shape "
                f"{tuple(sharded.shape)} != {target_global_shape}"
            )
            assert sharded.placements == target_placements, (
                f"load layout drift on {tfqn}: placements "
                f"{sharded.placements} != {target_placements}"
            )
            assert sharded.device_mesh is target_device_mesh, (
                f"load layout drift on {tfqn}: device_mesh changed"
            )

            sharded_sd[tfqn] = nn.Parameter(sharded, requires_grad=target.requires_grad)
            del local_dst, local_dst_flat, src_cpu_flat

        _tw2 = _time.time()
        _t_allreduce_total += _tw2 - _tw1

        if rank == 0:
            n_owned_rank0 = sum(1 for i in range(len(wave)) if i % world_size == 0)
            payload_bytes = sum(_numel(t) for t in wave) * FP32_BYTES
            print(
                f"[load_checkpoint_hp]   wave {wave_idx + 1}/{len(waves)} {len(wave)} params "
                f"({n_owned_rank0} owned by rank0, max param {(max_n_w * FP32_BYTES) >> 20} MiB "
                f"in {max_chunks} chunk(s), {payload_bytes >> 20} MiB total payload): "
                f"convert={_tw1 - _tw0:.2f}s allreduce+distribute={_tw2 - _tw1:.2f}s "
                f"gpu_peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
                flush=True,
            )

        assert not staged_cpu, f"wave {wave_idx} leaked staged tensors: {list(staged_cpu)}"

    del gpu_chunk_buf

    # Single cleanup after all waves: release any stranded convert intermediates
    # (HF mapping internals, lazy reader closures, etc.). gc.collect() first so
    # cycles drop their CUDA refs before empty_cache() returns memory to the OS.
    gc.collect()
    torch.cuda.empty_cache()

    _t3 = _time.time()
    if rank == 0:
        _peak4 = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"[load_checkpoint_hp] phase 4 (parallel convert + all_reduce + EP slice + distribute): "
            f"{_t3 - _t2:.2f}s ({len(fqn_to_mapping)} tensors materialized; "
            f"convert={_t_convert_total:.2f}s, all_reduce={_t_allreduce_total:.2f}s, "
            f"gpu_peak={_peak4:.2f} GiB)",
            flush=True,
        )

    if skipped_keys and rank == 0:
        # Expected: ``mtp.*`` heads from the upstream MTP class, filtered by
        # ``_keys_to_ignore_on_load_unexpected``. Print to surface anything else.
        head = ", ".join(sorted(skipped_keys)[:8])
        tail = f"... (+{len(skipped_keys) - 8} more)" if len(skipped_keys) > 8 else ""
        print(
            f"[load_checkpoint_hp] WARN: skipped {len(skipped_keys)} keys not in model: {head}{tail}"
        )

    self.load_state_dict(sharded_sd, strict=False, assign=True)

    _t4 = _time.time()
    if rank == 0:
        print(
            f"[load_checkpoint_hp] phase 5 (load_state_dict assign): {_t4 - _t3:.2f}s", flush=True
        )

    leftover_params = [n for n, p in self.named_parameters() if p.is_meta]
    assert not leftover_params, f"meta params not materialized after load: {leftover_params}"

    # --- MTP materialization audit (phase 5) -----------------------------
    # The global ``leftover_params`` check above already catches a missed
    # MTP slot (any unfilled meta param fails the assert). This block adds
    # observability: print per-depth param/buffer counts so a regression in
    # the disk → FQN remap surfaces as "MTP: 0 params" instead of being
    # buried in totals.
    if num_mtp_layers > 0 and rank == 0:
        mtp_params = [n for n, _ in self.named_parameters() if n.startswith("mtp.")]
        mtp_bufs = [n for n, _ in self.named_buffers() if n.startswith("mtp.")]
        per_depth_params: dict[int, int] = defaultdict(int)
        for n in mtp_params:
            m = re.match(r"^mtp\.layers\.(\d+)\.", n)
            if m is not None:
                per_depth_params[int(m.group(1))] += 1
        depths_seen = sorted(per_depth_params)
        assert depths_seen == list(range(num_mtp_layers)), (
            f"MTP depth coverage mismatch: expected depths "
            f"{list(range(num_mtp_layers))}, got {depths_seen} "
            f"(per-depth param counts: {dict(per_depth_params)})"
        )
        print(
            f"[load_checkpoint_hp] MTP materialized: {len(mtp_params)} params "
            f"+ {len(mtp_bufs)} buffers across {num_mtp_layers} depth(s) "
            f"(per-depth params: {dict(per_depth_params)})",
            flush=True,
        )

    _materialize_meta_buffers(self)

    _t5 = _time.time()
    if rank == 0:
        _peak_total = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"[load_checkpoint_hp] phase 6 (materialize buffers): {_t5 - _t4:.2f}s\n"
            f"[load_checkpoint_hp] TOTAL: {_t5 - _t0:.2f}s, gpu_peak={_peak_total:.2f} GiB",
            flush=True,
        )
    torch.cuda.reset_peak_memory_stats()


def _discover_safetensor_shards(hf_path: str, ) -> tuple[dict[str, str], list[str]]:
    """Return ``(weight_map, shard_files)`` from an HF checkpoint directory.

    Falls back to a single ``model.safetensors`` when no index file exists.
    """
    from safetensors import safe_open

    index_path = os.path.join(hf_path, "model.safetensors.index.json")
    single_path = os.path.join(hf_path, "model.safetensors")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        return weight_map, shard_files

    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt") as f:
            keys = list(f.keys())
        weight_map = {k: "model.safetensors" for k in keys}
        return weight_map, ["model.safetensors"]

    raise FileNotFoundError(
        f"No safetensors checkpoint found in {hf_path}. "
        f"Expected model.safetensors.index.json or model.safetensors."
    )


# ---------------------------------------------------------------------------
# Checkpoint saver: streaming gather + FP4/FP8 re-quantize → DSV4-Flash format
# ---------------------------------------------------------------------------

# DSV4-Flash on-disk dtype layout (derived from upstream safetensors shard survey).
#
# FP8 (float8_e4m3fn + E8M0 scale, 128×128 block):
#   - attn.{wq_a,wq_b,wkv,wo_a,wo_b}  directly under self_attn.*
#     (NOT under compressor.* or indexer.compressor.* — those are BF16)
#   - attn.indexer.wq_b                (the indexer's only FP8 leaf)
#   - ffn.shared_experts.w{1,2,3}
#
# FP4 (e2m1 packed int8 + E8M0 scale, 1×32 block):
#   - experts.{i}.w{1,2,3}.weight      (routed MoE experts only)
#
# BF16: everything else (embed/head, all compressor/indexer linears, ffn.gate.weight)
# FP32: norms, sinks, hc_*, position_bias, gate_bias (see _F32_DISK_KEY_PATTERNS)
#
# We use an explicit FP8 whitelist rather than a shape heuristic because the
# heuristic over-quantizes embed.weight, head.weight, and all compressor.* linears —
# vanilla DSV4 expects those as BF16 and fails to load FP8 versions.
#
# All patterns operate on the final DISK-FORM key (after reverse WeightRenaming pass).
_FP8_DISK_KEY_PATTERNS: tuple[str, ...] = (
    # Outer attention dense linears (NOT under compressor./indexer.compressor.).
    # Use a negative lookahead to exclude the inner blocks: inner ``compressor.wkv``
    # ships as BF16 and we must not catch it with the same fp8 rule.
    r"^(?:.*\.)?layers\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.weight$",
    # Indexer's own ``wq_b`` (only fp8 leaf the indexer carries; its inner
    # ``compressor.{wkv,wgate}`` are bf16 and not matched here).
    r"^(?:.*\.)?layers\.\d+\.attn\.indexer\.wq_b\.weight$",
    # Shared experts (dense FFN, not the per-token-routed MoE experts).
    r"^(?:.*\.)?layers\.\d+\.ffn\.shared_experts\.w[123]\.weight$",
    # MTP (multi-token-predict) heads ship the same dense-linear set as the
    # main blocks. They are skipped on load via ``_keys_to_ignore_on_load_unexpected``,
    # but to keep the saver round-trip-correct (and the checkpoint vanilla-loadable
    # when MTP is enabled), classify them identically.
    r"^mtp\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.weight$",
    r"^mtp\.\d+\.attn\.indexer\.wq_b\.weight$",
    r"^mtp\.\d+\.ffn\.shared_experts\.w[123]\.weight$",
    r"^mtp\.\d+\.(?:e_proj|h_proj)\.weight$",
)
_FP8_DISK_KEY_RE = re.compile("|".join(_FP8_DISK_KEY_PATTERNS))

# Disk-form keys that must be saved as FP32 even if currently BF16 in memory
# (load_checkpoint_hp may have downcast them via the ``dtype`` arg). Upstream
# DSV4-Flash ships these in FP32; we must match exactly:
#   - hc_attn_* / hc_ffn_* / hc_head_*   Hyper-Connection params
#   - attn_sink                            1D attention sink
#   - compressor.ape                       position bias (up to 2D fp32)
#   - ffn.gate.bias                        e_score_correction_bias on disk
_F32_DISK_KEY_PATTERNS: tuple[str, ...] = (
    r"^(?:.*\.)?hc_(?:attn|ffn|head)_(?:fn|base|scale)$",
    r"^(?:.*\.)?layers\.\d+\.attn\.attn_sink$",
    r"^(?:.*\.)?layers\.\d+\.attn\.(?:.*\.)?compressor\.ape$",
    r"^(?:.*\.)?layers\.\d+\.ffn\.gate\.bias$",
    r"^mtp\.\d+\.attn\.attn_sink$",
    r"^mtp\.\d+\.attn\.(?:.*\.)?compressor\.ape$",
    r"^mtp\.\d+\.ffn\.gate\.bias$",
)
_F32_DISK_KEY_RE = re.compile("|".join(_F32_DISK_KEY_PATTERNS))


def _classify_for_save(name: str, t: "torch.Tensor") -> str:
    """Decide which save path to take for ``t`` based on its disk-form name.

    Operates on the DISK-FORM key (post reverse-mapping, matching the upstream
    DSV4-Flash naming such as ``layers.0.ffn.experts.0.w1.weight`` and
    ``layers.0.attn.wq_a.weight``). The output checkpoint must be loadable by
    BOTH the patched ``model.load_checkpoint_hp`` *and* a vanilla
    ``DeepseekV4ForCausalLM.from_pretrained`` — so the per-key dtype layout
    has to match upstream exactly (modulo quantization noise on the FP4 / FP8
    paths). See ``_FP8_DISK_KEY_PATTERNS`` for the explicit FP8 whitelist and
    ``_F32_DISK_KEY_PATTERNS`` for the explicit F32-passthrough whitelist
    (which forces fp32 output regardless of in-memory dtype).

    Returns one of:

    * ``'fp4_expert'`` — MoE expert weights (``experts.{i}.w{1,2,3}.weight``),
      output ``int8`` packed e2m1 with E8M0 (``float8_e8m0fnu``) per-block
      scales, block ``(1, 32)``.
    * ``'fp8_e4m3'`` — dense linear weights matching ``_FP8_DISK_KEY_PATTERNS``,
      output ``float8_e4m3fn`` with E8M0 scales, block ``(128, 128)``.
    * ``'f32_passthrough'`` — keys matching ``_F32_DISK_KEY_PATTERNS``; written
      as fp32 even if currently bf16 in memory.
    * ``'bf16_passthrough'`` — everything else with floating-point dtype:
      ``embed.weight``, ``head.weight``, all norms, all
      ``compressor.``/``indexer.compressor.`` linears, ``ffn.gate.weight``.
    * ``'int_passthrough'`` — integer buffers (``tid2eid``).
    """
    if not t.dtype.is_floating_point:
        return "int_passthrough"
    # MoE expert weights live under ``.experts.{i}.w{1,2,3}.weight`` on disk
    # (but ``.shared_experts.w*`` are *dense* linears, NOT experts — caught
    # below by the FP8 whitelist instead). The ``.experts.\d+.`` infix segment
    # is the discriminator.
    if re.search(r"\.experts\.\d+\.w[123]\.weight$", name) is not None:
        return "fp4_expert"
    if _FP8_DISK_KEY_RE.search(name) is not None:
        return "fp8_e4m3"
    if _F32_DISK_KEY_RE.search(name) is not None:
        return "f32_passthrough"
    return "bf16_passthrough"


def _scale_key(weight_disk_key: str) -> str:
    """Return the DSV4-Flash on-disk scale companion key for a quantized weight.

    DSV4-Flash stores per-block FP4 / FP8 scales as a sibling tensor whose name
    replaces the trailing ``.weight`` with ``.scale`` (e.g.
    ``layers.0.attn.wq_a.weight`` ↔ ``layers.0.attn.wq_a.scale``). HF's
    :class:`FineGrainedFP8HfQuantizer` then applies a ``.scale → .weight_scale_inv``
    rename inside its load pipeline; the on-disk convention itself is ``.scale``.
    """
    assert weight_disk_key.endswith(".weight"), weight_disk_key
    return weight_disk_key[:-len(".weight")] + ".scale"


# ---------------------------------------------------------------------------
# Model FQN → disk-form key: explicit rename table (decoupled from HF reverse)
# ---------------------------------------------------------------------------
#
# Why this replaces HF's ``WeightConverter.reverse_transform()`` machinery:
#
#   The previous saver drove its reverse mapping off
#   ``_build_checkpoint_conversion_mapping()["deepseek_v4"]`` and
#   ``c.reverse_transform()`` for each forward converter. That approach hit
#   two HF BLOCKER bugs requiring ~290 lines of workaround:
#     1. Forward rules with TWO capture groups (e.g. ``self_attn.(.*?).wkv.``
#        for HCA / CSA / indexer leaves) don't reverse cleanly —
#        ``process_target_pattern`` only handles one group, leaving a literal
#        ``\2`` in the reversed source that never matches. We patched this
#        with ``_MANUAL_REV_LEAF_PATCHES``.
#     2. Rules 178 and 182 reverse to the same source pattern
#        (``self_attn.compressor.indexer.*``) with different targets — HF's
#        ``rename_source_key`` walks the list and picks the first match
#        silently, so we had to ``sort`` by ``(-src_len, -tgt_len)`` and
#        ``dedup`` to keep the more-specific rule.
#     Plus extra fixes for ``..`` → ``.`` regex-escape leftover, manual
#     prefix/suffix re-application on converter outputs, etc.
#
# This table is the strict inverse of ``conversion_mapping["deepseek_v4"]``
# forward rules. Each entry is a (regex, replacement) pair anchored on the
# full model FQN (with ``model.`` prefix already stripped). Order matters:
#   1. specific leaf rules (full-path anchored, e.g.
#      ``^layers\.X\.self_attn\.compressor\.indexer\.kv_proj\.``) run first.
#   2. structural prefix rules at the END (``self_attn.``→``attn.``,
#      ``mlp.``→``ffn.``) — they catch everything not handled by a leaf rule.
#
# Save path is now fully decoupled from HF's reverse machinery. HF can change
# its forward conversion_mapping freely; we only need to keep this table in
# sync. ``tests/test_gfused/test_dsv4_rename_table.py`` is a < 5s pure-CPU
# unit test that round-trips every representative disk-form leaf through HF
# forward → our reverse; run it after every transformers version bump.
_MODEL_TO_DISK_RENAMES: tuple[tuple[str, str], ...] = (
    # ---- top-level (no ``layers.`` prefix) ----
    (r"^embed_tokens\.weight$", "embed.weight"),
    (r"^lm_head\.weight$", "head.weight"),
    (r"^hc_head\.hc_fn$", "hc_head_fn"),
    (r"^hc_head\.hc_base$", "hc_head_base"),
    (r"^hc_head\.hc_scale$", "hc_head_scale"),
    # ---- per-layer specific leaves (run BEFORE structural prefix) ----
    # Hyper-Connection params (forward: hc_attn_*/hc_ffn_* → attn_hc.* / ffn_hc.*)
    (r"^layers\.(\d+)\.attn_hc\.fn$", r"layers.\1.hc_attn_fn"),
    (r"^layers\.(\d+)\.attn_hc\.base$", r"layers.\1.hc_attn_base"),
    (r"^layers\.(\d+)\.attn_hc\.scale$", r"layers.\1.hc_attn_scale"),
    (r"^layers\.(\d+)\.ffn_hc\.fn$", r"layers.\1.hc_ffn_fn"),
    (r"^layers\.(\d+)\.ffn_hc\.base$", r"layers.\1.hc_ffn_base"),
    (r"^layers\.(\d+)\.ffn_hc\.scale$", r"layers.\1.hc_ffn_scale"),
    (r"^layers\.(\d+)\.hc_head\.hc_(fn|base|scale)$", r"layers.\1.hc_head_\2"),
    # Norms
    (r"^layers\.(\d+)\.input_layernorm\.", r"layers.\1.attn_norm."),
    (r"^layers\.(\d+)\.post_attention_layernorm\.", r"layers.\1.ffn_norm."),
    # Inner indexer un-flatten: model flattens indexer into compressor.indexer.*,
    # but disk splits it across two levels (indexer.compressor.* and indexer.*).
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_proj\.",
        r"layers.\1.self_attn.indexer.compressor.wkv."
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.gate_proj\.",
        r"layers.\1.self_attn.indexer.compressor.wgate."
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_norm\.",
        r"layers.\1.self_attn.indexer.compressor.norm."
    ),
    # Strip fp32-keep _position_bias_holder wrapper BEFORE the position_bias→ape rule below.
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\._position_bias_holder\.weight$",
        r"layers.\1.self_attn.compressor.indexer.position_bias"
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.position_bias$",
        r"layers.\1.self_attn.indexer.compressor.ape"
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.q_b_proj\.",
        r"layers.\1.self_attn.indexer.wq_b."
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.weights_proj\.",
        r"layers.\1.self_attn.indexer.weights_proj."
    ),
    # Outer compressor leaves (HCA + CSA outer)
    (r"^layers\.(\d+)\.self_attn\.compressor\.kv_proj\.", r"layers.\1.self_attn.compressor.wkv."),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.gate_proj\.",
        r"layers.\1.self_attn.compressor.wgate."
    ),
    (r"^layers\.(\d+)\.self_attn\.compressor\.kv_norm\.", r"layers.\1.self_attn.compressor.norm."),
    # Strip fp32-keep _position_bias_holder wrapper BEFORE the position_bias→ape rule below.
    (
        r"^layers\.(\d+)\.self_attn\.compressor\._position_bias_holder\.weight$",
        r"layers.\1.self_attn.compressor.position_bias"
    ),
    (
        r"^layers\.(\d+)\.self_attn\.compressor\.position_bias$",
        r"layers.\1.self_attn.compressor.ape"
    ),
    # Attn outer leaves (wq_a/wq_b/wkv/wo_a/wo_b) + q_a_norm + sinks
    (r"^layers\.(\d+)\.self_attn\.q_a_proj\.", r"layers.\1.self_attn.wq_a."),
    (r"^layers\.(\d+)\.self_attn\.q_b_proj\.", r"layers.\1.self_attn.wq_b."),
    (r"^layers\.(\d+)\.self_attn\.kv_proj\.", r"layers.\1.self_attn.wkv."),
    (r"^layers\.(\d+)\.self_attn\.o_a_proj\.", r"layers.\1.self_attn.wo_a."),
    (r"^layers\.(\d+)\.self_attn\.o_b_proj\.", r"layers.\1.self_attn.wo_b."),
    (r"^layers\.(\d+)\.self_attn\.q_a_norm\.", r"layers.\1.self_attn.q_norm."),
    # Strip fp32-keep _sink_holder wrapper BEFORE the sinks→attn_sink rule below.
    (r"^layers\.(\d+)\.self_attn\._sink_holder\.weight$", r"layers.\1.self_attn.sinks"),
    (r"^layers\.(\d+)\.self_attn\.sinks$", r"layers.\1.self_attn.attn_sink"),
    # MoE gate bias + shared_experts gate/down/up → w1/w2/w3
    (r"^layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$", r"layers.\1.mlp.gate.bias"),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.", r"layers.\1.mlp.shared_experts.w1."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.down_proj\.", r"layers.\1.mlp.shared_experts.w2."),
    (r"^layers\.(\d+)\.mlp\.shared_experts\.up_proj\.", r"layers.\1.mlp.shared_experts.w3."),
    # ---- structural prefix renames (MUST run LAST, after all specific leaves) ----
    (r"^layers\.(\d+)\.self_attn\.", r"layers.\1.attn."),
    (r"^layers\.(\d+)\.mlp\.", r"layers.\1.ffn."),
)
_MODEL_TO_DISK_RENAMES_COMPILED: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(src), tgt) for src, tgt in _MODEL_TO_DISK_RENAMES
)

# Module-level so the compile cost is paid once at import.
_EXPERT_STACKED_KEY_RE = re.compile(
    r"^(layers|mtp\.layers)\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)


def _model_key_to_disk_key(model_key: str) -> str:
    """Apply ``_MODEL_TO_DISK_RENAMES`` cascade to convert a model FQN
    (without ``model.`` prefix) to the upstream DSV4-Flash disk-form key.

    Iterative single-rule-per-step rewrite: each pass finds the first matching
    rule, applies it, then restarts the search. Terminates when no rule
    fires. ``max_iter`` (set to 8) guards against accidental rule loops;
    empirically converges in ≤2 steps for current DSV4 keys (one specific
    leaf + one structural prefix).

    Parameters
    ----------
    model_key : str
        Model-side FQN, e.g. ``layers.0.self_attn.q_a_proj.weight`` (already
        stripped of the optional ``model.`` prefix).

    Returns
    -------
    str
        Disk-form key, e.g. ``layers.0.attn.wq_a.weight``. Keys without any
        applicable rule pass through unchanged (e.g. bare ``norm.weight``).

    Raises
    ------
    AssertionError
        If the cascade does not converge within ``max_iter`` rewrites (i.e.
        a rule keeps re-firing — bug in the rename table).
    """
    for _ in range(8):
        for rgx, repl in _MODEL_TO_DISK_RENAMES_COMPILED:
            new_key, n = rgx.subn(repl, model_key, count=1)
            if n > 0:
                model_key = new_key
                break
        else:
            return model_key
    raise AssertionError(
        f"_model_key_to_disk_key did not converge on {model_key!r}; "
        f"likely a self-retriggering rule in _MODEL_TO_DISK_RENAMES."
    )


def _split_per_expert(model_key: str, t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reverse of HF's ``[MergeModulelist(dim=0), Concatenate(dim=1)]`` for
    DSV4 MoE expert weights — split a stacked expert tensor into per-expert
    leaves with disk-style ``mlp.experts.{i}.w{1,2,3}.weight`` keys (still
    model-side; structural prefix ``mlp.``→``ffn.`` applied later by
    :func:`_model_key_to_disk_key`).

    Forward shapes (from ``modeling_deepseek_v4.py``):

    * ``gate_up_proj``: ``(num_experts, 2 * intermediate_dim, hidden_dim)``
      — w1 stack ``Concatenate(dim=1)`` w3 stack along dim=1.
    * ``down_proj``: ``(num_experts, hidden_dim, intermediate_dim)``
      — ``MergeModulelist(dim=0)`` of per-expert ``(H, I)`` tensors.

    Reverse:

    * gate_up_proj → ``chunk(2, dim=1)`` → two ``(N, I, H)`` halves
      (w1_stack, w3_stack) → per-expert ``t[i]`` ⇒ ``(I, H)``.
    * down_proj → per-expert ``t[i]`` ⇒ ``(H, I)``.

    Parameters
    ----------
    model_key : str
        Model FQN without ``model.`` prefix, e.g.
        ``layers.0.mlp.experts.gate_up_proj``.
    t : torch.Tensor
        Stacked expert tensor with shape ``(num_experts, ...)``. For
        non-expert keys, returned wrapped as ``{model_key: t}`` unchanged.

    Returns
    -------
    dict[str, torch.Tensor]
        One entry per ``(expert_idx, w_kind)`` for stacked expert tensors;
        a single-entry dict for everything else.

    Raises
    ------
    AssertionError
        If ``model_key`` contains ``.experts.`` but is not a recognized
        stacked-expert key (and is not ``.shared_experts.``), or if
        ``gate_up_proj`` second-dim is odd (cannot split into w1/w3).
    """
    m = _EXPERT_STACKED_KEY_RE.match(model_key)
    if m is None:
        # Defensive: any .experts.-bearing key must either be a known stacked
        # tensor handled above, or shared_experts (dense, not actually expert).
        assert ".experts." not in model_key or ".shared_experts." in model_key, (
            f"unexpected expert key not handled by _split_per_expert: {model_key!r}"
        )
        return {model_key: t}

    key_prefix, layer, w_kind = m.group(1), m.group(2), m.group(3)
    n = t.shape[0]

    if w_kind == "down_proj":
        return {
            f"{key_prefix}.{layer}.mlp.experts.{i}.w2.weight": t[i].contiguous()
            for i in range(n)
        }

    # gate_up_proj
    assert t.shape[1] % 2 == 0, (
        f"{model_key} shape {tuple(t.shape)} not divisible by 2 on dim=1 "
        f"(cannot split into w1/w3)"
    )
    w1_stack, w3_stack = t.chunk(2, dim=1)
    out: dict[str, torch.Tensor] = {}
    for i in range(n):
        out[f"{key_prefix}.{layer}.mlp.experts.{i}.w1.weight"] = w1_stack[i].contiguous()
        out[f"{key_prefix}.{layer}.mlp.experts.{i}.w3.weight"] = w3_stack[i].contiguous()
    return out


def _rank0_finalize(
    self: nn.Module,
    save_path: str,
    orig_ckpt_dir: str,
    gather_list: list[dict],
    placeholder_shard_suffix: str,
    world_size: int,
    max_shard_size: int,
    *,
    dtype_format: Literal["quantized", "bf16"],
    preserve_mtp: bool = True,
) -> None:
    """Finalize the parallel save on rank 0: global rename + index + aux.

    Globally numbers per-rank shards (ordered by ``(rank↑, local_idx↑)``),
    renames every rank's ``model-rank{r:02d}-{i:05d}-of-XXXXX.safetensors``
    to ``model-{g:05d}-of-{N:05d}.safetensors``, writes
    ``model.safetensors.index.json`` (natural-key sorted), handles
    ``quantization_config`` per ``dtype_format`` (preserve for ``"quantized"``,
    strip for ``"bf16"``), and copies tokenizer + README from ``orig_ckpt_dir``.

    Pulled out of :func:`_save_checkpoint_hp` so the caller can wrap the
    entire rank-0 block in a single try/except and propagate failures
    via the cross-rank fail-flag helper without nesting indentation.
    """

    log_debug(f"DEBUG save_checkpoint_hp: preserve_mtp={preserve_mtp}")

    sorted_meta = sorted(gather_list, key=lambda m: m["rank"])
    total_size_bytes = sum(m["total_size_bytes"] for m in sorted_meta)

    # Phase 1: Rename per-rank shards to sequential intermediate names.
    # Final "-of-{total}" suffix is deferred until after MTP shards are added.
    _INTERMEDIATE_SUFFIX = "of-XXXXX.safetensors"
    rename_map: dict[str, str] = {}  # old rank-placeholder → intermediate
    global_idx = 0
    tensor_to_filename: dict[str, str] = {}
    for m in sorted_meta:
        r = m["rank"]
        for local_idx in range(1, m["shard_count"] + 1):
            global_idx += 1
            old_name = (f"model-rank{r:02d}-{local_idx:05d}-{placeholder_shard_suffix}")
            intermediate_name = f"model-{global_idx:05d}-{_INTERMEDIATE_SUFFIX}"
            os.rename(
                os.path.join(save_path, old_name),
                os.path.join(save_path, intermediate_name),
            )
            rename_map[old_name] = intermediate_name
        for key, fname in m["tensor_to_filename"].items():
            tensor_to_filename[key] = rename_map[fname]

    # ---- MTP passthrough: copy MTP tensors verbatim from source checkpoint ----
    if preserve_mtp:
        src_index_path = os.path.join(orig_ckpt_dir, "model.safetensors.index.json")
        src_single_path = os.path.join(orig_ckpt_dir, "model.safetensors")
        if os.path.isfile(src_index_path):
            with open(src_index_path) as f:
                src_index = json.load(f)
            src_weight_map = src_index["weight_map"]
            mtp_by_shard: dict[str, list[str]] = defaultdict(list)
            for key, shard_name in src_weight_map.items():
                if key.startswith("mtp.") and key not in tensor_to_filename:
                    mtp_by_shard[shard_name].append(key)

            if mtp_by_shard:
                mtp_pending: dict[str, torch.Tensor] = {}
                mtp_pending_size = 0
                mtp_total_bytes = 0

                def _flush_mtp() -> None:
                    nonlocal global_idx, mtp_pending, mtp_pending_size
                    if not mtp_pending:
                        return
                    global_idx += 1
                    shard_name = f"model-{global_idx:05d}-{_INTERMEDIATE_SUFFIX}"
                    st_torch.save_file(mtp_pending, os.path.join(save_path, shard_name))
                    for k in mtp_pending:
                        tensor_to_filename[k] = shard_name
                    mtp_pending = {}
                    mtp_pending_size = 0

                for src_shard, keys in sorted(mtp_by_shard.items()):
                    src_path = os.path.join(orig_ckpt_dir, src_shard)
                    with safetensors.safe_open(src_path, framework="pt", device="cpu") as sf:
                        for key in keys:
                            t = sf.get_tensor(key)
                            t_bytes = t.element_size() * t.nelement()
                            mtp_pending[key] = t
                            mtp_pending_size += t_bytes
                            mtp_total_bytes += t_bytes
                            if mtp_pending_size >= max_shard_size:
                                _flush_mtp()
                _flush_mtp()

                total_size_bytes += mtp_total_bytes
                print(
                    f"[save_checkpoint_hp] MTP passthrough: copied {len(mtp_by_shard)} "
                    f"source shard(s), {mtp_total_bytes / (1024**3):.2f} GiB "
                    f"of MTP weights from {orig_ckpt_dir}",
                    flush=True,
                )
        elif os.path.isfile(src_single_path):
            mtp_pending: dict[str, torch.Tensor] = {}
            mtp_pending_size = 0
            mtp_total_bytes = 0
            with safetensors.safe_open(src_single_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if not key.startswith("mtp.") or key in tensor_to_filename:
                        continue
                    t = sf.get_tensor(key)
                    t_bytes = t.element_size() * t.nelement()
                    mtp_pending[key] = t
                    mtp_pending_size += t_bytes
                    mtp_total_bytes += t_bytes
                    if mtp_pending_size >= max_shard_size:
                        global_idx += 1
                        shard_name = f"model-{global_idx:05d}-{_INTERMEDIATE_SUFFIX}"
                        st_torch.save_file(mtp_pending, os.path.join(save_path, shard_name))
                        for k in mtp_pending:
                            tensor_to_filename[k] = shard_name
                        mtp_pending = {}
                        mtp_pending_size = 0
            if mtp_pending:
                global_idx += 1
                shard_name = f"model-{global_idx:05d}-{_INTERMEDIATE_SUFFIX}"
                st_torch.save_file(mtp_pending, os.path.join(save_path, shard_name))
                for k in mtp_pending:
                    tensor_to_filename[k] = shard_name
            if mtp_total_bytes > 0:
                total_size_bytes += mtp_total_bytes
                print(
                    f"[save_checkpoint_hp] MTP passthrough(single-file): copied "
                    f"{mtp_total_bytes / (1024**3):.2f} GiB from {orig_ckpt_dir}",
                    flush=True,
                )
        else:
            print(
                f"[save_checkpoint_hp] WARN: preserve_mtp=True but no "
                f"model.safetensors.index.json/model.safetensors in {orig_ckpt_dir}; "
                f"skipping MTP copy",
                flush=True,
            )

    # Phase 2: Now that total shard count is known, rename all intermediate
    # shards to the canonical "model-{g:05d}-of-{total:05d}.safetensors" form.
    total = global_idx
    final_rename: dict[str, str] = {}
    for idx in range(1, total + 1):
        intermediate_name = f"model-{idx:05d}-{_INTERMEDIATE_SUFFIX}"
        final_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
        os.rename(
            os.path.join(save_path, intermediate_name),
            os.path.join(save_path, final_name),
        )
        final_rename[intermediate_name] = final_name
    for key in tensor_to_filename:
        tensor_to_filename[key] = final_rename[tensor_to_filename[key]]

    # Write model.safetensors.index.json.
    # CRITICAL: keys must be in natural (integer) order for experts.* —
    # MergeModulelist on the loader side iterates weight_map in dict order.
    # Lexicographic order interleaves (0, 1, 10, 100, ..., 99) and corrupts
    # expert slot assignments.
    def _natural_key(key: str) -> list:
        return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", key)]

    index = {
        "metadata": {
            "total_size": total_size_bytes
        },
        "weight_map": dict(sorted(tensor_to_filename.items(), key=lambda kv: _natural_key(kv[0]))),
    }
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        # sort_keys=False is mandatory — sort_keys=True would undo the natural sort.
        json.dump(index, f, indent=2, sort_keys=False)

    # Handle quantization_config in saved config.json:
    #
    # * dtype_format="quantized" — emit FP4/FP8/E8M0 + ``.scale`` companion keys
    #   on disk, layout matches DSV4-Flash original. Preserve
    #   ``quantization_config`` so HF ``from_pretrained`` auto-applies the
    #   FineGrainedFP8 dequant path on reload (same code path as orig ckpt).
    #
    # * dtype_format="bf16" — fold FP4/FP8 onto bf16 at save time. Must strip
    #   ``quantization_config`` because keeping ``quant_method="fp8"`` would
    #   make HF look for ``*.scale`` keys that don't exist in a bf16 save.
    #   Setting ``config.quantization_config = None`` is NOT enough:
    #   ``PretrainedConfig`` still writes ``"quantization_config": null`` and
    #   HF's ``supports_quant_method`` crashes with
    #   ``AttributeError: 'NoneType'.get('quant_method')``. Fix: save, then
    #   re-read + pop the key from the raw JSON.
    config_for_save = deepcopy(self.config)
    if dtype_format == "bf16":
        config_for_save.quantization_config = None
        config_for_save.save_pretrained(save_path)
        cfg_json = os.path.join(save_path, "config.json")
        with open(cfg_json, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("quantization_config", None)
        with open(cfg_json, "w") as f:
            json.dump(cfg_dict, f, indent=2, sort_keys=True)
    else:  # dtype_format == "quantized"
        config_for_save.save_pretrained(save_path)

    if self.can_generate():
        self.generation_config.save_pretrained(save_path)

    # Copy tokenizer and README from orig_ckpt_dir so save_path is a
    # self-contained from_pretrained() target. Tokenizer files are required;
    # README is optional.
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(orig_ckpt_dir, fname)
        assert os.path.isfile(src), (
            f"required aux file missing in orig_ckpt_dir={orig_ckpt_dir!r}: "
            f"{fname}; cannot produce a self-contained save dir without it"
        )
        shutil.copy2(src, os.path.join(save_path, fname))
    readme = os.path.join(orig_ckpt_dir, "README.md")
    if os.path.isfile(readme):
        shutil.copy2(readme, os.path.join(save_path, "README.md"))

    print(
        f"[save_checkpoint_hp] done: {total} shards ("
        f"{total_size_bytes / (1024**3):.2f} GiB total) across "
        f"{world_size} writer ranks → {save_path}",
        flush=True,
    )


def _save_checkpoint_hp(
    self: nn.Module,
    save_path: str,
    orig_ckpt_dir: str,
    max_shard_size: int = 5 * 1024**3,
    *,
    dtype_format: Literal["quantized", "bf16"] = "quantized",
    preserve_mtp: bool = True,
) -> None:
    """Save an EP+FSDP2(+CP) model as a HF-compatible checkpoint dir.

    Parameters
    ----------
    save_path
        Output directory; will be populated with ``model-*.safetensors`` shards,
        ``model.safetensors.index.json``, ``config.json``, ``generation_config.json``,
        and copies of ``tokenizer.json`` / ``tokenizer_config.json`` /
        ``README.md`` from ``orig_ckpt_dir``.
    orig_ckpt_dir
        Source DSV4-Flash checkpoint dir, used as the byte-for-byte source for
        tokenizer + README files.
    max_shard_size
        Approximate per-shard byte budget (5 GiB default).
    dtype_format
        Disk dtype layout, keyword-only. Two paths are supported:

        * ``"quantized"`` (default) — FP4 packed-int8 expert weights + FP8 e4m3
          dense linear weights + E8M0 (``float8_e8m0fnu``) per-block ``.scale``
          companion keys + bf16/fp32 norms + int64 buffers. Layout matches
          upstream DSV4-Flash exactly; ``quantization_config`` is preserved in
          ``config.json`` so HF ``from_pretrained(save_path,
          FineGrainedFP8Config(dequantize=True))`` reloads via the standard
          fine-grained-FP8 dequant path. Total disk size ≈ 150 GiB on full
          43-layer DSV4-Flash.
        * ``"bf16"`` — every FP4/FP8 weight is folded onto bf16 at save time;
          ``quantization_config`` is stripped from ``config.json`` so vanilla
          ``from_pretrained(save_path, torch_dtype=bfloat16)`` reloads without
          touching the FP8 dequant path. Total disk size ≈ 530 GiB on full
          43L (~3.5× larger than quantized).

        The two paths share the wave-based parallel save framework, EP gather,
        rename table, and per-expert split — they only differ in the wave-B
        dtype dispatch and ``_rank0_finalize`` config-strip step.

    Notes
    -----
    The saved directory is a self-contained ``from_pretrained(save_path)``
    target. Disk-key layout matches the upstream DSV4-Flash convention so HF's
    ``conversion_mapping["deepseek_v4"]`` forward path picks it up: outer
    ``attn`` / ``ffn`` prefixes, per-expert ``w1`` / ``w2`` / ``w3`` split,
    nested ``compressor`` / ``indexer.compressor`` leaves, etc.

    Reverse mapping is driven by an explicit
    :data:`_MODEL_TO_DISK_RENAMES` table + manual :func:`_split_per_expert`,
    NOT by ``WeightConverter.reverse_transform()``. This decouples the save
    path from HF's internal reverse machinery (which has multiple BLOCKER
    bugs around two-capture rules and many-to-one reverse) and shrinks
    ~290 lines of workaround to ~50 lines of straight rewrite. The cost is
    that HF version bumps no longer auto-propagate: after any transformers
    upgrade, run ``tests/test_gfused/test_dsv4_rename_table.py`` (<5s pure
    CPU) to verify the table is still in sync with HF forward mapping.

    ``tokenizer.json`` / ``tokenizer_config.json`` /
    ``generation_config.json`` / ``README.md`` are byte-for-byte copied
    from ``orig_ckpt_dir`` (DSV4-Flash has no ``.py`` modeling files, so
    ``trust_remote_code`` is not required).

    Strategy: wave-based, single-thread
    -----------------------------------
    Mirrors :func:`_load_checkpoint_hp`'s Phase 4 — distribute work
    across ranks via round-robin ownership over a ``(-global_numel,
    name)``-sorted item list, processed in **waves** of ``world_size``
    params each. No background threads, queues, CUDA events, or
    cross-thread CUDA state.

    Each wave runs two sub-phases:

    * **Wave-A (gather, ``world_size`` collectives)**: For each of the
      ≤``world_size`` params in the wave, every rank participates in
      one gather (``DTensor.from_local + full_tensor + dist.all_gather``
      2-step for experts; plain ``full_tensor()`` otherwise). The
      round-robin **owner rank** ``i % world_size`` (i within wave)
      synchronously D2H-copies the result to CPU; non-owners drop.
      Each rank holds **at most one** owned tensor in CPU memory after
      Wave-A (last wave may have fewer).
    * **Wave-B (per-rank parallel CPU + disk)**: With no further
      collectives, every rank independently splits stacked experts
      into per-expert leaves, renames model FQN → disk-form key via
      :func:`_model_key_to_disk_key`, casts to on-disk dtype, and
      streams to a rank-private shard file
      (``model-rank{r:02d}-{i:05d}-of-XXXXX.safetensors``). All
      ``world_size`` ranks run Wave-B in parallel — the speedup
      source. Pending shard buffer carries over to the next wave so
      shards are sized by ``max_shard_size``, not by wave count.

    After all waves:

    * **Finalize (rank 0 only)**: collect per-rank shard metadata via
      ``dist.gather_object``, globally rename shards to the canonical
      ``model-{g:05d}-of-{N:05d}.safetensors`` form, write
      ``model.safetensors.index.json`` + ``config.json`` + aux files.

    Rationale: a naive "owner-only writes" variant without wave/phase
    separation still serializes — the next ``all_gather`` blocks at the
    collective barrier until the current owner's CPU + disk work
    finishes, giving no real speedup. An earlier thread-pipelined
    variant fixed that but required passing GPU tensors + ``cuda.Event``
    between threads, which depends on implementation-detail behavior
    of PyTorch NCCL stream sync. A bulk two-phase variant (gather
    everything first, then write everything) gives the same parallel
    speedup but accumulates ``~17 GiB`` of owned tensors in CPU memory
    sustained across the entire walk. Wave-based execution keeps the
    cluster speedup while bounding CPU memory to **one owned tensor
    per rank** between waves — only the transient split intermediate
    matters.

    Memory budget (DSV4-Flash, fp32 case, per rank):

    * GPU peak: dominated by the 2-step expert gather inside Wave-A
      — ``ep_local_full`` (~3.75 GiB) + ``ep_chunks`` (``ep_size`` ×
      ~3.75 = ~30 GiB) + ``cat(t)`` (~30 GiB) all alive simultaneously
      just before ``del ep_chunks``, plus FSDP-sharded local model +
      caches. Roughly ~64-80 GiB transient on H20 96 GiB. After
      ``del ep_chunks`` peak drops to ~30 GiB (just ``t``).
    * CPU peak in Wave-A: **at most one owned tensor** per rank
      (~30 GiB for the biggest expert), not the sum of all owned.
    * CPU peak in Wave-B: one expert split intermediate (~30 GiB) +
      pending shard buffer (~5 GiB) ≈ ~35 GiB transient. Drops to
      pending size between waves.

    bf16-loaded paths halve all expert sizes.

    DSV4-Pro caveat: fp32 ``gate_up_proj`` is ~67 GiB → 2-step gather
    transient on H20 will OOM. A chunked variant of the 2-step gather
    (mirror of load Phase B's chunked all_reduce) is left as
    follow-up; bf16 path works as-is.

    Notes
    -----
    * Every rank writes its share of the checkpoint to disk; rank 0
      additionally writes the index, config, tokenizer, and README at
      the end. All ranks participate in Phase 1 collectives.
    * A ``dist.barrier()`` at the end guarantees disk writes are durable
      before any rank returns (so subsequent reloads on other ranks
      observe the complete checkpoint).
    * Failure handling: every risky section is wrapped in try/except
      and the local error is spread to all ranks via
      ``_spread_fail`` (an ``all_reduce(MAX)`` over a fail flag) at the
      next synchronization point. Without this any uncaught exception
      on a subset of ranks would deadlock the rest at the next
      collective.
    * Pre-existing shards in ``save_path`` are NOT cleaned up
      automatically — caller should ``rmtree`` first.
    """
    import time as _time

    import safetensors.torch

    assert dtype_format in ("quantized", "bf16"), dtype_format

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0 = rank == 0

    def _spread_fail(local_err: BaseException | None, fallback_msg: str) -> None:
        """Spread a local failure to every rank so they bail together.

        Any uncaught exception on a subset of ranks while others are
        waiting at a collective deadlocks the cluster. Wrap risky
        sections in try/except, call this helper once at the next
        synchronization point — every rank participates in the same
        ``all_reduce(MAX)`` and raises uniformly if any rank failed.
        """
        flag = torch.tensor(
            1 if local_err is not None else 0,
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            if local_err is not None:
                raise local_err
            raise RuntimeError(fallback_msg)

    # ------------------------------------------------------------------
    # 0) mkdir on rank 0 + sync.
    # ------------------------------------------------------------------
    mkdir_err: BaseException | None = None
    if is_rank0:
        try:
            os.makedirs(save_path, exist_ok=True)
        except BaseException as e:  # noqa: BLE001
            mkdir_err = e
    _spread_fail(mkdir_err, f"parallel save: rank-0 mkdir failed (local rank={rank})")
    # Wait for rank-0's mkdir to land on shared FS before others try to write.
    dist.barrier()

    # Per-rank shard write state. Each rank writes to its own files; rank 0
    # globally renames everything in Phase 3.
    placeholder_shard_suffix = "of-XXXXX.safetensors"
    pending: dict[str, torch.Tensor] = {}
    pending_size = 0
    local_shard_idx = 0
    local_tensor_to_filename: dict[str, str] = {}
    local_total_size_bytes = 0

    def _flush() -> None:
        """Write current ``pending`` to a per-rank numbered shard file."""
        nonlocal pending_size, local_shard_idx, local_total_size_bytes
        if not pending:
            return
        local_shard_idx += 1
        shard_name = (f"model-rank{rank:02d}-{local_shard_idx:05d}-{placeholder_shard_suffix}")
        shard_path = os.path.join(save_path, shard_name)
        safetensors.torch.save_file(pending, shard_path)
        for key, tensor in pending.items():
            local_tensor_to_filename[key] = shard_name
            local_total_size_bytes += tensor.element_size() * tensor.nelement()
        pending.clear()
        pending_size = 0

    def _add(key: str, tensor: torch.Tensor) -> None:
        """Append a CPU tensor to ``pending``; flush at ``max_shard_size``."""
        nonlocal pending_size
        tensor = tensor.contiguous()
        pending[key] = tensor
        pending_size += tensor.element_size() * tensor.nelement()
        if pending_size >= max_shard_size:
            _flush()

    # ------------------------------------------------------------------
    # 1) Sort by (-global_numel, name) — same key as load Phase 4 — so big
    #    experts get spread evenly across owners via round-robin.
    # ------------------------------------------------------------------
    def _global_numel(name: str, p: torch.Tensor) -> int:
        # ``DTensor.shape`` is GLOBAL for FSDP-only sharding. For expert
        # weights, ``apply_hp`` first slices to ``(num_local, ...)`` as a
        # plain Parameter and then FSDP2 wraps it, so ``p.shape[0]`` is
        # ``num_local`` (EP-sliced), NOT ``num_experts``. Multiply by
        # ``self._ep_size`` to restore the full expert-dim.
        n = 1
        for d in p.shape:
            n *= int(d)
        if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            n *= self._ep_size
        return n

    state_dict = self.state_dict()
    sorted_items = sorted(state_dict.items(), key=lambda kv: (-_global_numel(kv[0], kv[1]), kv[0]))

    if is_rank0:
        n_experts = sum(
            1 for name, _ in sorted_items
            if ".experts.gate_up_proj" in name or ".experts.down_proj" in name
        )
        print(
            f"[save_checkpoint_hp] plan: {len(sorted_items)} model params "
            f"({n_experts} stacked-expert tensors), world_size={world_size}, "
            f"ep_size={self._ep_size}, save_path={save_path}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 2) Wave loop — each wave gathers ``world_size`` params (last may be
    #    short) so every rank holds AT MOST ONE owned tensor in CPU at
    #    a time, then writes it before the next wave's gather. Errors
    #    in Wave-B are recorded locally; subsequent waves still
    #    participate in the gather collectives so no rank deadlocks
    #    waiting on a failed peer. Final ``_spread_fail`` propagates
    #    after the loop.
    # ------------------------------------------------------------------
    waves: list[list[tuple[str, torch.Tensor]]] = [
        sorted_items[i:i + world_size] for i in range(0, len(sorted_items), world_size)
    ]
    if is_rank0:
        print(
            f"[save_checkpoint_hp] waves: {len(waves)} (≤{world_size} params each, "
            f"last={len(waves[-1]) if waves else 0})",
            flush=True,
        )

    write_err: BaseException | None = None
    _t_gather_total = 0.0
    _t_write_total = 0.0
    for wave_idx, wave in enumerate(waves):
        _tw0 = _time.time()

        # ---------- Wave-A: gather (at most one owned per rank) ----------
        my_owned: list[tuple[str, torch.Tensor]] = []
        for i, (name, p) in enumerate(wave):
            owner = i % world_size
            is_expert_weight = (".experts.gate_up_proj" in name or ".experts.down_proj" in name)

            if isinstance(p, DTensor):
                if is_expert_weight and self._ep_size > 1:
                    # Two-step gather to recover expert weights in MODEL ORDER.
                    #
                    # Expert weights are sharded on two mesh dims:
                    #   ep      (slow): ep_idx holds experts [ep_idx * num_local ..)
                    #   ep_fsdp (fast): FSDP2 further shards the local block
                    #
                    # A naive ``DTensor.from_local(mesh=(ep_fsdp, ep),
                    # placements=[Shard(0), Shard(0)]).full_tensor()`` reassembles in
                    # MESH row-major order (ep_fsdp slow, ep fast), transposing the
                    # expert layout. Fix: gather ep_fsdp first (correct order within
                    # each EP slot), then ``dist.all_gather`` over ep to concatenate
                    # slots in ep_idx order → full expert tensor.
                    local = p.to_local().contiguous()
                    dt_fsdp = DTensor.from_local(
                        local,
                        device_mesh=self._ep_fsdp_mesh,
                        placements=[Shard(0)],
                    )
                    ep_local_full = dt_fsdp.full_tensor()  # (num_local, ...)

                    ep_chunks = [torch.empty_like(ep_local_full) for _ in range(self._ep_size)]
                    dist.all_gather(ep_chunks, ep_local_full, group=self._ep_group)
                    t = torch.cat(ep_chunks, dim=0)  # (num_experts, ...)
                    del ep_local_full, ep_chunks
                else:
                    t = p.full_tensor()
            else:
                t = p.detach()

            if rank != owner:
                del t
                continue

            # ``state_dict()`` keys have a ``model.`` prefix for
            # inner-model params; rename table operates on the bare FQN.
            model_key = name.removeprefix("model.")
            my_owned.append((model_key, t.cpu()))
            del t

        # Reclaim GPU caches from the wave's gather transients before disk work.
        torch.cuda.empty_cache()

        _tw1 = _time.time()
        _t_gather_total += _tw1 - _tw0

        # ---------- Wave-B: per-rank parallel CPU + disk (no collective) ----------
        # If a previous wave already failed, skip work but still participate
        # in subsequent gather collectives (must drop refs first).
        if write_err is None:
            try:
                while my_owned:
                    model_key, t_cpu = my_owned.pop(0)
                    realized = _split_per_expert(model_key, t_cpu)
                    del t_cpu
                    for mk, mv in realized.items():
                        mtp_m = re.match(r"^mtp\.layers\.(\d+)\.(.+)$", mk)
                        if mtp_m is not None:
                            mtp_depth = int(mtp_m.group(1))
                            logical_layer_key = f"layers.{mtp_depth}.{mtp_m.group(2)}"
                            disk_key = _model_key_to_disk_key(logical_layer_key)
                            disk_key = disk_key.replace(
                                f"layers.{mtp_depth}.",
                                f"mtp.{mtp_depth}.",
                                1,
                            )
                        else:
                            disk_key = _model_key_to_disk_key(mk)
                        cls = _classify_for_save(disk_key, mv)
                        if dtype_format == "quantized":
                            # Strict DSV4-Flash layout: emit FP4/FP8 codes plus
                            # E8M0 ``.scale`` companion key for every quantized
                            # weight; non-quantized leaves stay bf16/fp32/int.
                            if cls == "fp4_expert":
                                packed, scale = quant_fp4_e2m1_scale_e8m0_packed(mv)
                                _add(disk_key, packed)
                                _add(_scale_key(disk_key), scale)
                            elif cls == "fp8_e4m3":
                                qfp8, scale = quant_fp8_e4m3_scale_e8m0(mv)
                                _add(disk_key, qfp8)
                                _add(_scale_key(disk_key), scale)
                            elif cls == "f32_passthrough":
                                _add(disk_key, mv.to(torch.float32))
                            elif cls == "bf16_passthrough":
                                _add(disk_key, mv.to(torch.bfloat16))
                            elif cls == "int_passthrough":
                                _add(disk_key, mv)
                            else:
                                raise RuntimeError(f"unhandled save class {cls!r} for {disk_key}")
                        else:  # dtype_format == "bf16"
                            # Legacy bf16 fold: fp4_expert / fp8_e4m3 / bf16_passthrough
                            # all → bf16. f32_passthrough → fp32. int → unchanged.
                            if cls in ("fp4_expert", "fp8_e4m3", "bf16_passthrough"):
                                _add(disk_key, mv.to(torch.bfloat16))
                            elif cls == "f32_passthrough":
                                _add(disk_key, mv.to(torch.float32))
                            elif cls == "int_passthrough":
                                _add(disk_key, mv)
                            else:
                                raise RuntimeError(f"unhandled save class {cls!r} for {disk_key}")
                        del mv
            except BaseException as e:  # noqa: BLE001
                write_err = e
        my_owned.clear()  # drop refs even on the skipped-after-failure path

        _tw2 = _time.time()
        _t_write_total += _tw2 - _tw1

        if is_rank0:
            n_owned_r0 = sum(1 for i in range(len(wave)) if i % world_size == 0)
            print(
                f"[save_checkpoint_hp]   wave {wave_idx + 1}/{len(waves)} "
                f"{len(wave)} params ({n_owned_r0} owned by rank0): "
                f"gather={_tw1 - _tw0:.2f}s write(rank0)={_tw2 - _tw1:.2f}s",
                flush=True,
            )

    # Flush remaining pending (carried over the last wave's threshold).
    if write_err is None:
        try:
            _tw_flush0 = _time.time()
            _flush()
            _t_write_total += _time.time() - _tw_flush0
        except BaseException as e:  # noqa: BLE001
            write_err = e

    if is_rank0:
        print(
            f"[save_checkpoint_hp] totals: gather={_t_gather_total:.2f}s "
            f"write(rank0)={_t_write_total:.2f}s "
            f"(rank0 wrote {local_shard_idx} shards, "
            f"{local_total_size_bytes / (1024**3):.2f} GiB)",
            flush=True,
        )

    _spread_fail(
        write_err,
        f"parallel save: wave loop (CPU+disk) failed on another rank "
        f"(local rank={rank} ok)",
    )

    # ------------------------------------------------------------------
    # 4) Phase 3 — collect per-rank shard meta onto rank 0.
    # ------------------------------------------------------------------
    local_meta = {
        "rank": rank,
        "shard_count": local_shard_idx,
        "tensor_to_filename": local_tensor_to_filename,
        "total_size_bytes": local_total_size_bytes,
    }
    gather_list: list[dict] | None = (
        [None] * world_size if is_rank0 else None  # type: ignore[list-item]
    )
    dist.gather_object(local_meta, gather_list, dst=0)

    # ------------------------------------------------------------------
    # 5) Phase 3 (rank 0 only) — global rename, index.json, config, tokenizer.
    #
    # P1: wrap with try/except + spread the fail flag — any uncaught
    # exception inside the rank-0 block would otherwise leave non-rank-0
    # ranks hanging forever at the closing ``dist.barrier()``.
    # ------------------------------------------------------------------
    finalize_err: BaseException | None = None
    if is_rank0:
        try:
            _rank0_finalize(
                self=self,
                save_path=save_path,
                orig_ckpt_dir=orig_ckpt_dir,
                gather_list=gather_list,
                placeholder_shard_suffix=placeholder_shard_suffix,
                world_size=world_size,
                max_shard_size=max_shard_size,
                dtype_format=dtype_format,
                preserve_mtp=preserve_mtp,
            )
        except BaseException as e:  # noqa: BLE001
            finalize_err = e
    _spread_fail(
        finalize_err,
        f"parallel save: rank-0 finalize failed (local rank={rank} ok)",
    )

    dist.barrier()
