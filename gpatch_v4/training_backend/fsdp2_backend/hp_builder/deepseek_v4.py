# coding=utf-8
"""DeepSeek-V4 (and default) HP builder."""

from __future__ import annotations

from typing import Any

from torch.nn import Module

from gpatch_v4.training_backend.fsdp2_backend.hp_builder.base import HpModelBuilder
from gpatch_v4.utils import log

try:
    from gpatch_v4.models.deepseek_v4 import apply_hp as apply_hp_deepseek_v4
    from gpatch_v4.models.deepseek_v4.checkpoint import infer_dspark_num_layers
except ImportError:
    apply_hp_deepseek_v4 = None
    infer_dspark_num_layers = None


class DefaultHpBuilder(HpModelBuilder):
    """DSV4-style HP path (also the fallback for unregistered HpModule arches)."""
    def prepare_config(self, model_cls: type, hf_model_path: str, engine: Any) -> Any:
        cfg = model_cls.config_class.from_pretrained(hf_model_path, trust_remote_code=True)

        # HP Module 其实不用这个字段，写一个 'eager' fallback 下
        cfg._attn_implementation = 'eager'

        enable_mtp = engine.training_config.enable_mtp
        enable_dspark = engine.training_config.enable_dspark
        mtp_num_layers = int(cfg.num_nextn_predict_layers)
        if enable_dspark:
            assert cfg.dspark_block_size > 0
            assert cfg.dspark_noise_token_id is not None
            assert cfg.dspark_target_layer_ids
            assert infer_dspark_num_layers is not None
            cfg.dspark_num_layers = infer_dspark_num_layers(hf_model_path)
            # Module init requires num_anchors > 0; load-only may omit the train knob.
            cfg.dspark_num_anchors = (
                engine.training_config.dspark_num_anchors
                if engine.training_config.dspark_num_anchors is not None else 1
            )
        elif enable_mtp:
            assert "dspark_block_size" not in cfg.to_dict(
            ), ("DSpark checkpoints require training.enable_dspark, not enable_mtp")
            assert mtp_num_layers > 0, (
                "training.enable_mtp=True but hf config has no MTP layers "
                f"(num_nextn_predict_layers={mtp_num_layers})"
            )
            cfg.mtp_loss_scaling_factor = float(
                getattr(engine.training_config, "mtp_loss_scaling_factor", 0.1)
            )
        else:
            cfg.num_nextn_predict_layers = 0

        # DEBUG: optionally truncate to N decoder layers (matches the
        # `_truncate_config` helper in tests/test_gfused/test_deepseek_v4_ep_cp.py).
        # Used for OOM smoke runs; mismatched checkpoint layers are simply
        # ignored by the streaming load path.
        n_layers_dbg = engine.config.debug.debug_truncate_num_hidden_layers
        if n_layers_dbg is not None:
            log(
                f"DEBUG: truncating model config from "
                f"num_hidden_layers={cfg.num_hidden_layers} -> {n_layers_dbg}",
                rank=0,
            )
            cfg.num_hidden_layers = n_layers_dbg
            if hasattr(cfg, 'layer_types') and cfg.layer_types is not None:
                cfg.layer_types = cfg.layer_types[:n_layers_dbg]
            if hasattr(cfg, 'mlp_layer_types') and cfg.mlp_layer_types is not None:
                cfg.mlp_layer_types = cfg.mlp_layer_types[:n_layers_dbg]
            if enable_dspark:
                # debug, 截断测试阶段时候才能用这个路径
                num_target_layers = len(cfg.dspark_target_layer_ids)
                assert n_layers_dbg >= num_target_layers
                cfg.dspark_target_layer_ids = list(
                    range(n_layers_dbg - num_target_layers, n_layers_dbg)
                )
        return cfg

    def apply_hp(self, model: Module, engine: Any) -> Module:
        return apply_hp_deepseek_v4(
            model,
            engine.ep_2d_mesh,
            cp_mesh=engine.cp_mesh_for_hp,
            attn_backend=engine.policy_config.attn_implementation,
            indexer_backend=engine.policy_config.indexer_backend,
            ep_backend=engine.policy_config.ep_backend,
            deepep_num_sms=engine.policy_config.deepep_num_sms,
            fp8_qat=engine.policy_config.fp8_qat,
            fp4_qat=engine.policy_config.fp4_qat,
            fp4_qat_indexer=engine.policy_config.fp4_qat_indexer,
            fp8=engine.policy_config.fp8,
            fsdp_fp8_gather=engine.policy_config.fsdp_fp8_gather,
            moe_router_force_load_balancing=(engine.policy_config.moe_router_force_load_balancing),
        )
