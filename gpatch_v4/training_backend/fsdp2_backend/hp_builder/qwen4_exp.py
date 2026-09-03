# coding=utf-8
"""Qwen4-Exp HP builder."""

from __future__ import annotations

from typing import Any

from torch.nn import Module

from gpatch_v4.training_backend.fsdp2_backend.hp_builder.base import HpModelBuilder
from gpatch_v4.utils import log

_QWEN4_EXP_IMPORT_ERROR = None
try:
    from gpatch_v4.models.qwen4_exp import apply_hp as apply_hp_qwen4_exp
except Exception as error:
    apply_hp_qwen4_exp = None
    _QWEN4_EXP_IMPORT_ERROR = error


class Qwen4ExpHpBuilder(HpModelBuilder):
    def validate(self, engine: Any) -> None:
        if apply_hp_qwen4_exp is None:
            raise ImportError(
                "model_arch=qwen4_exp requires gpatch_v4.models.qwen4_exp"
            ) from _QWEN4_EXP_IMPORT_ERROR
        if engine.training_config.enable_mtp:
            raise NotImplementedError(
                "qwen4_exp MTP finetune is not supported; the released 4 B MTP head is "
                "not loaded"
            )
        if engine.training_config.enable_dspark:
            raise NotImplementedError("DSpark is only implemented for DeepSeek-V4")
        if engine.policy_config.dist_config.dynamic_context_parallel:
            raise NotImplementedError(
                "qwen4_exp FSDP2 does not support dynamic_context_parallel; use static CP"
            )
        if engine.policy_config.indexer_backend != "eager":
            raise NotImplementedError(
                "qwen4_exp has no separate indexer kernel backend; use "
                "indexer_backend='eager'"
            )
        unsupported_flags = [
            name for name, enabled in (
                ("fp8_qat", engine.policy_config.fp8_qat),
                ("fp4_qat", engine.policy_config.fp4_qat),
                ("fp4_qat_indexer", engine.policy_config.fp4_qat_indexer),
                ("fp8", engine.policy_config.fp8),
                ("fsdp_fp8_gather", engine.policy_config.fsdp_fp8_gather),
                (
                    "moe_router_force_load_balancing",
                    engine.policy_config.moe_router_force_load_balancing,
                ),
            ) if enabled
        ]
        if unsupported_flags:
            raise NotImplementedError(
                f"{unsupported_flags} are DeepSeek-V4 features and are not implemented "
                "for qwen4_exp"
            )

    def prepare_config(self, model_cls: type, hf_model_path: str, engine: Any) -> Any:
        cfg = model_cls.config_class.from_pretrained(hf_model_path)
        # The released config enables a DynamicCache, which SFT never consumes.
        cfg.use_cache = False

        n_layers_dbg = engine.config.debug.debug_truncate_num_hidden_layers
        if n_layers_dbg is not None:
            log(
                f"DEBUG: truncating qwen4_exp from num_hidden_layers="
                f"{cfg.num_hidden_layers} -> {n_layers_dbg}",
                rank=0,
            )
            cfg.num_hidden_layers = n_layers_dbg
            cfg.layer_types = list(cfg.layer_types[:n_layers_dbg])
            # `ple_layer_ids` is one-indexed; drop any Engram layer past the new depth.
            cfg.ple_layer_ids = [i for i in cfg.ple_layer_ids if i <= n_layers_dbg]
        return cfg

    def apply_hp(self, model: Module, engine: Any) -> Module:
        return apply_hp_qwen4_exp(
            model,
            engine.ep_2d_mesh,
            cp_mesh=engine.cp_mesh_for_hp,
            attn_backend=engine.policy_config.attn_implementation,
            ep_backend=engine.policy_config.ep_backend,
        )
