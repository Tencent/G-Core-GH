# copying form verl

import dataclasses
import hashlib
import pickle
import re
from collections import Counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.router import TopKRouter

try:
    from mbridge.peft.canonical_lora import (
        CanonicalLoRA,
        LoRALinearSplitFC1UpGate,
        LoRALinearSplitGDNInProj,
        LoRALinearSplitQKV,
    )
    from mbridge.peft.lora import LoRA
    from mbridge.peft.lora_layers import (
        LinearAdapter,
        LoRAGroupedLinear,
        LoRALinear,
        LoRATopKRouter,
        TEFusedLoRALinear,
        TELinearAdapter,
    )
    from mbridge.peft.utils import ParallelLinearAdapter
except ImportError:
    CanonicalLoRA = None
    LoRALinearSplitFC1UpGate = None
    LoRALinearSplitGDNInProj = None
    LoRALinearSplitQKV = None
    LoRA = None
    LinearAdapter = None
    LoRAGroupedLinear = None
    LoRALinear = None
    LoRATopKRouter = None
    TEFusedLoRALinear = None
    TELinearAdapter = None
    ParallelLinearAdapter = None

try:
    from megatron.bridge.peft.canonical_lora import (
        CanonicalLoRA as MBridgeCanonicalLoRA,
    )
    from megatron.bridge.peft.lora import LoRA as MBridgeLoRA
except ImportError:
    MBridgeCanonicalLoRA = None
    MBridgeLoRA = None

from gpatch_v4.utils.common_utils import logging_rank0
from gpatch_v4.utils.logging_utils import log_info


def is_peft_enabled(policy_config) -> bool:
    """Whether ``policy.lora`` enables PEFT."""
    return policy_config.lora.enabled()


def get_peft_cls(policy_config, bridge=None, provider=None, dtype=None, use_mbridge=False):
    """Build a PEFT instance from ``policy.lora``.

    Parameters
    ----------
    policy_config : BasePolicyConfig
    bridge : AutoBridge / mbridge Bridge, optional
    provider : Megatron model provider, optional
    dtype : torch.dtype, optional
    use_mbridge : bool
        If True, import PEFT classes from mbridge instead of Megatron-Bridge.

    Returns
    -------
    LoRA, CanonicalLoRA, DoRA, or None
    """
    lora_cfg = policy_config.lora
    if not lora_cfg.enabled():
        return None

    if use_mbridge:
        assert LoRA is not None, "mbridge.peft is not available"
        _LoRA, _CanonicalLoRA = LoRA, CanonicalLoRA
    else:
        assert MBridgeLoRA is not None, "megatron.bridge.peft is not available"
        assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"
        _LoRA, _CanonicalLoRA = MBridgeLoRA, MBridgeCanonicalLoRA

    lora_dtype = lora_cfg.dtype if lora_cfg.dtype is not None else dtype

    user_targets = lora_cfg.target_modules

    lora_type = lora_cfg.type
    peft_cls = None
    common_kwargs = dict(
        dim=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        dropout_position=lora_cfg.dropout_position,
        lora_A_init_method=lora_cfg.lora_A_init_method,
        lora_B_init_method=lora_cfg.lora_B_init_method,
        exclude_modules=lora_cfg.exclude_modules,
        share_expert_adapters=lora_cfg.share_expert_adapters,
    )
    if lora_type == "lora":
        kwargs = {
            **common_kwargs, "a2a_experimental": lora_cfg.a2a_experimental,
            "lora_dtype": lora_dtype
        }
        if user_targets is not None:
            kwargs["target_modules"] = user_targets
        peft_cls = _LoRA(**kwargs)
    elif lora_type == "canonical_lora":
        if user_targets is not None:
            common_kwargs["target_modules"] = user_targets
        # canonical_mapping 断言 exclude_modules 必须为空；改为应用层跳过
        # （构造后包装 transform，见下），构造参数清空以绕过断言
        _shim_excludes = list(common_kwargs.get("exclude_modules") or [])
        if _shim_excludes:
            common_kwargs["exclude_modules"] = []
        peft_cls = _CanonicalLoRA(**common_kwargs)
        if _shim_excludes:
            _rxs = [re.compile("^" + pt.replace("*", "(.*)") + "$") for pt in _shim_excludes]
            _orig_transform = peft_cls.transform

            def _transform_skip_excluded(module, name=None, prefix=None, *args, **kw):
                full = f"{prefix}.{name}" if prefix else (name or "")
                if full and any(r.match(full) for r in _rxs):
                    return module
                return _orig_transform(module, name=name, prefix=prefix, *args, **kw)

            peft_cls.transform = _transform_skip_excluded
            logging_rank0(f"canonical_lora exclude via transform-skip: {_shim_excludes}")
    else:
        raise ValueError(
            f"Unknown PEFT type {lora_type!r}; expected lora, vlm_lora, canonical_lora, or dora"
        )

    logging_rank0(
        f"Enabling {lora_type.upper()} with rank={lora_cfg.rank}, "
        f"alpha={lora_cfg.alpha}, dropout={lora_cfg.dropout}"
    )
    return peft_cls


def _log_lora_coverage(model_chunks):
    """Log which linear layers have LoRA adapters and which do not.

    Walks all modules in the model, classifying each linear layer as either
    wrapped by a LoRA adapter or left as a plain linear.  Results are
    deduplicated by a layer-agnostic pattern (``layers.N.`` → ``layers.*.``)
    so the output stays compact for deep models.
    """
    adapter_types = (
        LinearAdapter,
        LoRAGroupedLinear,
        LoRALinear,
        TEFusedLoRALinear,
        TELinearAdapter,
        LoRALinearSplitQKV,
        LoRALinearSplitFC1UpGate,
        LoRALinearSplitGDNInProj,
        LoRATopKRouter,
    )
    linear_types = (
        ColumnParallelLinear,
        RowParallelLinear,
        nn.Linear,
        TopKRouter,
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    models = model_chunks if isinstance(model_chunks, list) else [model_chunks]

    with_lora: list[str] = []
    without_lora: list[str] = []
    adapter_child_prefixes: list[str] = []

    for model in models:
        for name, module in model.named_modules():
            if isinstance(module, adapter_types):
                with_lora.append(name)
                adapter_child_prefixes.append(name + ".")

    for model in models:
        for name, module in model.named_modules():
            if not isinstance(module, linear_types):
                continue
            if isinstance(module, adapter_types):
                continue
            if any(name.startswith(p) for p in adapter_child_prefixes):
                continue
            without_lora.append(name)

    def _to_pattern(name: str) -> str:
        return re.sub(r"\.(\d+)\.", ".*.", name)

    def _dedup_report(names: list[str]) -> list[str]:
        counts = Counter(_to_pattern(n) for n in names)
        return [f"{pat} (x{cnt})" if cnt > 1 else pat for pat, cnt in counts.items()]

    if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_context_parallel_rank(
    ) == 0:
        log_info("=== LoRA Coverage Report ===")
        log_info(f"Linear layers WITH LoRA ({len(with_lora)}):")
        for line in _dedup_report(with_lora):
            log_info(f"  [+] {line}")
        log_info(f"Linear layers WITHOUT LoRA ({len(without_lora)}):")
        for line in _dedup_report(without_lora):
            log_info(f"  [-] {line}")
        log_info("=== End LoRA Coverage Report ===")

    return with_lora, without_lora


def apply_peft_pre_wrap_hook(
    model_chunks,
    peft,
    use_mbridge,
    check_lora_all_coverage,
    verify_weight_consistency=False,
    **kwargs
):
    """Pre-wrap hook: apply PEFT transformation before DDP wrapping.

    Extra ``kwargs`` are accepted so this can be used as both a Megatron-Bridge
    pre-wrap hook and an mbridge ``post_model_creation_callback`` (which passes
    ``pre_process``, ``post_process``, ``config``, ``hf_config``).
    """
    result = peft(model_chunks, training=True)
    if use_mbridge:
        _, without_lora = _log_lora_coverage(result)
        if check_lora_all_coverage:
            # Filter out output_layer and router
            without_lora_filter = [n for n in without_lora if not n.endswith("output_layer")]
            without_lora_filter = [n for n in without_lora_filter if not n.endswith("router")]
            assert len(without_lora_filter) == 0, "LoRA is not all coverage"

        if verify_weight_consistency:
            _verify_lora_weight_consistency(result, tag="after_peft_inject")

    return result


def peft_to_run_config_dict(peft) -> dict[str, Any]:
    """Serialize PEFT settings for ``run_config.yaml`` (HF adapter export)."""
    peft_dict = {}
    for f in dataclasses.fields(peft):
        val = getattr(peft, f.name)
        if val is None or isinstance(val, (str, int, float, bool)):
            peft_dict[f.name] = val
        elif isinstance(val, (list, tuple)):
            peft_dict[f.name] = list(val)
        else:
            peft_dict[f.name] = str(val)
    class_name = type(peft).__name__
    module = type(peft).__module__
    peft_dict["_target_"] = f"{module}.{class_name}"
    return peft_dict


def _verify_lora_weight_consistency(model_chunks, tag: str = "init"):
    """Verify LoRA weight consistency across TP/DP/CP ranks.

    Walks all adapter types (same set as ``_log_lora_coverage``) and checks
    that their LoRA weights (``linear_in`` / ``linear_out``) are consistent
    across parallel dimensions:

    - **TP**: replicated adapters (``linear_in`` is ``nn.Linear``) must have
      identical weights; TP-sharded adapters (``linear_in`` is
      ``ColumnParallelLinear`` / ``RowParallelLinear``) are skipped because
      their weights are intentionally partitioned across TP ranks.
    - **DP / CP**: all adapters must have identical weights for the same
      TP shard — same partition on the same TP rank, same full weight on
      replicated adapters.

    Also verifies ``average_gradients_across_tp_domain`` on replicated
    adapters (parallel adapters handle gradient sync internally via
    Column/RowParallelLinear).

    Parameters
    ----------
    model_chunks : nn.Module or list[nn.Module]
        Model(s) with LoRA adapters injected.
    tag : str
        Label for log messages (e.g. "after_peft_inject", "step_0").
    """
    # Same adapter types as _log_lora_coverage
    _ALL_ADAPTER_TYPES = (
        LinearAdapter,
        TELinearAdapter,
        ParallelLinearAdapter,
        LoRALinear,
        LoRAGroupedLinear,
        LoRATopKRouter,
        TEFusedLoRALinear,
        LoRALinearSplitQKV,
        LoRALinearSplitFC1UpGate,
        LoRALinearSplitGDNInProj,
    )

    if not dist.is_initialized():
        return

    models = model_chunks if isinstance(model_chunks, list) else [model_chunks]

    # Collect leaf adapters that directly own LoRA weights (linear_in / linear_out).
    # Wrapper types (LoRALinear, LoRALinearSplitQKV, etc.) delegate to inner
    # adapters which are discovered as separate named_modules, so we only need
    # to collect adapters with a direct ``linear_in`` attribute.
    adapters = {}
    for model in models:
        for name, module in model.named_modules():
            if isinstance(module, _ALL_ADAPTER_TYPES) and hasattr(module, "linear_in"):
                adapters[name] = module

    if not adapters:
        return

    def _is_tp_sharded(adapter) -> bool:
        """Whether adapter's LoRA weights are TP-sharded.

        Determined by the type of ``linear_in`` submodule:
        - ``ColumnParallelLinear`` / ``RowParallelLinear`` → TP-sharded
        - ``nn.Linear`` / other → replicated (identical across TP ranks)
        """
        linear_in = getattr(adapter, "linear_in", None)
        return isinstance(linear_in, (ColumnParallelLinear, RowParallelLinear))

    def _param_hash(param: torch.Tensor) -> str:
        data = param.data.detach().cpu().float()
        h = hashlib.md5(data.numpy().tobytes()).hexdigest()[:12]
        return f"shape={tuple(data.shape)} md5={h}"

    def _allgather_hashes(local_hashes: dict, group) -> dict:
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        local_bytes = pickle.dumps(local_hashes)
        local_tensor = torch.ByteTensor(list(local_bytes)).cuda()
        size_tensor = torch.zeros(world_size, dtype=torch.long, device="cuda")
        size_tensor[rank] = local_tensor.numel()
        dist.all_reduce(size_tensor, group=group)
        max_size = size_tensor.max().item()
        if max_size == 0:
            return {}
        padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        padded[:local_tensor.numel()] = local_tensor
        gathered = [
            torch.zeros(max_size, dtype=torch.uint8, device="cuda") for _ in range(world_size)
        ]
        dist.all_gather(gathered, padded, group=group)
        result = {}
        for r in range(world_size):
            sz = size_tensor[r].item()
            if sz > 0:
                result[r] = pickle.loads(bytes(gathered[r][:sz].cpu().tolist()))
        return result

    def _collect_lora_hashes(adapter, prefix: str) -> dict:
        """Collect hashes of LoRA parameters (linear_in / linear_out)."""
        hashes = {}
        for pname, param in adapter.named_parameters():
            if "linear_in" in pname or "linear_out" in pname:
                hashes[f"{prefix}.{pname}"] = _param_hash(param)
        return hashes

    def _check_group(adapters_dict, group, group_name: str, skip_tp_sharded: bool = False) -> bool:
        group_size = dist.get_world_size(group=group)
        if group_size <= 1:
            return True
        ok = True
        for name, adapter in adapters_dict.items():
            if skip_tp_sharded and _is_tp_sharded(adapter):
                continue
            local_hashes = _collect_lora_hashes(adapter, name)
            if not local_hashes:
                continue
            all_hashes = _allgather_hashes(local_hashes, group)
            ref_rank = min(all_hashes.keys())
            ref = all_hashes[ref_rank]
            for r, hashes in all_hashes.items():
                if hashes != ref:
                    for k in set(list(hashes.keys()) + list(ref.keys())):
                        if hashes.get(k) != ref.get(k):
                            logging_rank0(
                                f"[VERIFY][{tag}] MISMATCH in {group_name}: "
                                f"{k} rank={r} hash={hashes.get(k)} "
                                f"ref_rank={ref_rank} ref_hash={ref.get(k)}"
                            )
                    ok = False
        return ok

    # TP: skip TP-sharded adapters (their weights are intentionally partitioned)
    tp_ok = _check_group(
        adapters, parallel_state.get_tensor_model_parallel_group(), "TP", skip_tp_sharded=True
    )
    # DP: compare ALL adapters (same TP shard should be identical across DP ranks)
    dp_ok = _check_group(adapters, parallel_state.get_data_parallel_group(), "DP")
    # CP: compare ALL adapters (same TP shard should be identical across CP ranks)
    cp_ok = _check_group(adapters, parallel_state.get_context_parallel_group(), "CP")

    # Check average_gradients_across_tp_domain on replicated adapters only.
    # TP-sharded adapters (ParallelLinearAdapter) use Column/RowParallelLinear
    # which handle gradient sync internally, so the flag is intentionally absent.
    for name, adapter in adapters.items():
        if _is_tp_sharded(adapter):
            continue
        for pname, param in adapter.named_parameters():
            if ("linear_in" in pname or "linear_out" in pname) and param.requires_grad:
                has_flag = getattr(param, "average_gradients_across_tp_domain", False)
                if not has_flag:
                    logging_rank0(
                        f"[VERIFY][{tag}] WARNING: {name}.{pname} "
                        f"missing average_gradients_across_tp_domain=True"
                    )

    all_ok = tp_ok and dp_ok and cp_ok
    status = "PASS" if all_ok else "FAIL"
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    n_replicated = sum(1 for a in adapters.values() if not _is_tp_sharded(a))
    n_parallel = sum(1 for a in adapters.values() if _is_tp_sharded(a))
    logging_rank0(
        f"[VERIFY][{tag}] {status}: "
        f"{n_replicated} replicated + {n_parallel} parallel adapters, "
        f"TP={tp_size} DP={dp_size} CP={cp_size}"
    )


__all__ = [
    "apply_peft_pre_wrap_hook",
    "get_peft_cls",
    "is_peft_enabled",
    "peft_to_run_config_dict",
]
