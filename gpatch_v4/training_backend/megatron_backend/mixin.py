import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from einops import rearrange
from transformers import AutoConfig, AutoTokenizer

from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.data_schedule_utils import get_thd_partitioned_indices
from megatron.core.distributed import finalize_model_grads
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import divide, get_attr_wrapped_model

try:
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
except:
    MTPLossLoggingHelper = None

try:
    from megatron.core.transformer.moe.moe_logging import (
        get_moe_metrics_tracker,
        get_moe_overload_factor_tracker,
    )
except:
    get_moe_metrics_tracker = None
    get_moe_overload_factor_tracker = None

try:
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossLoggingHelper,
    )
except:
    DSAIndexerLossLoggingHelper = None

from gpatch_v4.configs.config import (
    DpoConfig,
    EmbeddingConfig,
    OnPolicyDistillConfig,
    RewardConfig,
)
from gpatch_v4.configs.transformer_config import merge_core_transformer_config
from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
from gpatch_v4.core.parallel_state import (
    cpu_group,
    get_last_rank,
    get_model_parallel_group_gloo,
    is_last_rank,
    is_mp_and_cp_head,
    is_mp_head,
)
from gpatch_v4.core.smart_pad_helper import CatedSmartPadInferHelper, _sample_idx_key
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.loss import PolicyLossInput as PolicyLossInputV2
from gpatch_v4.training_backend.loss import get_loss_fn
from gpatch_v4.training_backend.loss.registry import finalize_histogram_metrics
from gpatch_v4.training_backend.loss_factory import (
    FinetuneLossInput,
    PolicyLossInput,
    get_policy_loss_fn,
    is_seq_mean_rl_loss_fn,
)
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    bridge_save_hf,
    get_dataloader_save_path,
    get_latest_checkpoint_folder,
    load_checkpoint,
    save_checkpoint,
)
from gpatch_v4.training_backend.megatron_backend.mcore_peft import (
    apply_peft_pre_wrap_hook,
    get_peft_cls,
)
from gpatch_v4.training_backend.megatron_backend.megatron_utils import (
    get_model_config,
    unwrap_model,
)
from gpatch_v4.training_backend.megatron_backend.model_forward import (
    gptmodel_pack_foward,
)
from gpatch_v4.training_backend.megatron_backend.optimizer import (
    should_use_distributed_optimizer,
)

try:
    from gpatch_v4.training_backend.megatron_backend.qat import qat_parameters_context
except ImportError:
    qat_parameters_context = None

from gpatch_v4.training_backend.megatron_backend.router_replay_manager import (
    RouterReplay,
    RouterReplayAction,
    RouterReplayCtx,
    RouterReplayManager,
)
from gpatch_v4.training_backend.vocab_parallel_entropy import vocab_parallel_entropy
from gpatch_v4.utils import log_info
from gpatch_v4.utils.common_utils import (
    cache_hf_metadata_files,
    clear_memory,
    log,
    logging_rank0,
    profile_memory_and_time,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.communication_utils import BroadcastUtils
from gpatch_v4.utils.dynamic_cp_utils import (
    as_sequence_sample_mask,
    build_grpo_compact_ce_mask_dyn_cp,
    build_per_gid_fullseq_dump,
    dynamic_cp_local_packed_token_count,
    jagged_to_response_padded,
    packed_to_jagged,
    packed_to_response_padded,
    reconstruct_dynamic_cp_packed_tensor,
    reverse_and_collect_dump_metrics,
)
from gpatch_v4.utils.training_utils import (
    build_grpo_compact_ce_mask,
    expand_rollout_batches,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_opd_topk_logprobs,
    from_parallel_logits_to_token_prob_and_rank,
    from_parallel_logits_to_topk_logprobs,
    get_batches_max_seqlen,
    get_im_end_metrics_token_id,
    get_iterator_k_split_list,
    get_max_seqlen_within_dp,
    get_max_seqlen_within_ep,
    get_tensor_on_this_cp_rank,
    logprobs_from_compact_ce,
    logprobs_from_linear_ce,
    masked_mean,
    opd_topk_logprobs_from_linear_ce,
    update_square_averaging_token_len,
)


def _cp_partition_mode_from_fwd_kwargs(fwd_kwargs: Optional[Dict[str, Any]]) -> str:
    if not fwd_kwargs:
        return "zigzag"
    packed_seq_params = fwd_kwargs.get("packed_seq_params")
    if packed_seq_params is None:
        return "zigzag"
    return getattr(packed_seq_params, "cp_partition_mode", None) or "zigzag"


def build_megatron_bridge_ddp_config_dict(
    *,
    use_megatron_fsdp: bool,
    override_ddp_config: dict[str, Any],
    build_from_mbridge: bool,
    wrap_with_ddp: bool,
) -> dict[str, Any]:
    ddp_config_dict = {"use_distributed_optimizer": True}
    if use_megatron_fsdp:
        assert wrap_with_ddp, "policy.use_megatron_fsdp=True requires policy.wrap_with_ddp=True."
        assert not build_from_mbridge, (
            "policy.use_megatron_fsdp=True requires training.build_from_mbridge=False "
            "so the model is built through Megatron-Bridge."
        )
        ddp_config_dict.update(
            {
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    ddp_config_dict.update(override_ddp_config)
    if use_megatron_fsdp:
        assert ddp_config_dict["use_distributed_optimizer"] is True, (
            "policy.use_megatron_fsdp=True requires use_distributed_optimizer=True."
        )
        assert ddp_config_dict["use_megatron_fsdp"] is True, (
            "policy.use_megatron_fsdp=True requires ddp use_megatron_fsdp=True."
        )
        assert ddp_config_dict["data_parallel_sharding_strategy"] in {
            "optim_grads_params",
        }, "Unsupported Megatron-FSDP data_parallel_sharding_strategy."
    return ddp_config_dict


def build_megatron_bridge_provide_model_kwargs(
    *,
    wrap_with_ddp: bool,
    ddp_config: Any,
    use_megatron_fsdp: bool,
) -> dict[str, Any]:
    kwargs = {
        "wrap_with_ddp": wrap_with_ddp,
        "ddp_config": ddp_config,
    }
    if use_megatron_fsdp:
        kwargs["use_megatron_fsdp"] = True
        # Megatron-Bridge defaults this to True and then broadcasts DTensor
        # parameters after FSDP wrapping. HF/checkpoint loading synchronizes
        # weights later, so skip the init-time broadcast for Megatron-FSDP.
        kwargs["data_parallel_random_init"] = False
    return kwargs


def get_megatron_bridge_weight_load_model(model: Any, *, use_megatron_fsdp: bool) -> Any:
    # Keep the FSDP wrapper. Bridge.load_weights_hf_to_megatron detects
    # FullyShardedDataParallel and calls install_optimized_model_weights()
    # (main/DTensor buffer -> model_weight_buffer). Unwrapping skips that
    # sync, so the first forward still sees random compute weights.
    return model


def log_freeze_status(model, *args, **kwargs):
    """Post-creation callback that prints requires_grad stats per submodule.

    Runs after the multimodal freeze callback to verify each sub-tower
    (language_model / vision_model / merger / audio_model) was actually
    frozen. Only PP ranks holding a submodule report; others stay silent.
    """
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0

    submodules = {
        "language_model": getattr(model, "language_model", None),
        "vision_model": getattr(model, "vision_model", None),
        "vision_projection(merger)": (
            getattr(getattr(model, "vision_model", None), "merger", None)
        ),
        "audio_model": getattr(model, "audio_model", None),
    }

    for name, module in submodules.items():
        if module is None:
            continue
        total = 0
        frozen = 0
        trainable_params = 0
        frozen_params = 0
        for p in module.parameters():
            total += 1
            numel = p.numel()
            if p.requires_grad:
                trainable_params += numel
            else:
                frozen += 1
                frozen_params += numel
        if total == 0:
            continue
        log(
            f"FREEZE DETAIL module={name} tensors={total} "
            f"frozen_tensors={frozen} trainable_tensors={total - frozen} "
            f"frozen_params={frozen_params} trainable_params={trainable_params}",
            rank=0,
        )

    return model


class BridgeUtilsMixin:
    def get_hf_mtp_num_layers(self, hf_config):
        hf_text_config = getattr(hf_config, "text_config", hf_config)
        mtp_num_layers = getattr(hf_text_config, "mtp_num_hidden_layers", None)
        if mtp_num_layers is None:
            mtp_num_layers = getattr(hf_text_config, "num_nextn_predict_layers", None)
        if mtp_num_layers is None:
            mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", None)

        return mtp_num_layers

    def build_bridge(self, hf_model_path, override_transformer_config=None):
        cache_hf_metadata_files(hf_model_path, self.checkpoint_config.save_ckpt_path)
        if self.training_config.return_hidden_states_for_ce:
            # mbridge hidden+weight early-return; this does not select the SFT CE kernel.
            if override_transformer_config is None:
                override_transformer_config = {}
            override_transformer_config['cross_entropy_loss_fusion'] = True
            override_transformer_config['cross_entropy_fusion_impl'] = 'linear'
            logging_rank0(
                "Linear CE or compact CE: inject cross_entropy_loss_fusion=True, "
                "cross_entropy_fusion_impl='linear' into override_transformer_config"
            )

        if self.config.training.apply_deterministic_mode:
            if override_transformer_config is None:
                override_transformer_config = {}
            override_transformer_config['deterministic_mode'] = True

        if self.config.training.build_from_mbridge:
            return self.build_mbridge(hf_model_path, override_transformer_config)
        else:
            assert not self.policy_config.post_wrap_with_ddp, (
                "policy.post_wrap_with_ddp is only supported when "
                "training.build_from_mbridge=True."
            )
            return self.build_megatron_bridge(hf_model_path, override_transformer_config)

    def wrap_mbridge_model_with_ddp(self, model: List[torch.nn.Module]) -> List[torch.nn.Module]:
        """Wrap an already-loaded mbridge model with Megatron DDP.

        ``mbridge.get_model(wrap_with_ddp=True)`` wraps before HF weights are
        loaded. For very large tensors this makes ``bridge.load_weights`` peak
        on top of DDP buffers. Delaying DDP keeps the load path memory profile
        close to the mbridge forward-only examples while preserving the final
        training wrapper.
        """
        from megatron.core.distributed import (
            DistributedDataParallel,
            DistributedDataParallelConfig,
        )

        config = get_model_config(model[0])
        use_dist_opt = should_use_distributed_optimizer(self.config.optimizer)
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=use_dist_opt,
            **self.policy_config.override_ddp_config,
        )

        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000,
                1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True),
            )
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        wrapped_model = [
            DistributedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
            ) for model_chunk_idx, model_chunk in enumerate(model)
        ]

        if self.config.training.data_parallel_random_init:
            for model_chunk in wrapped_model:
                model_chunk.broadcast_params()

        for model_chunk in wrapped_model:
            unwrapped = unwrap_model(model_chunk)
            if hasattr(unwrapped, "decoder"):
                for layer in unwrapped.decoder.layers:
                    if (
                        hasattr(layer, "mlp") and hasattr(layer.mlp, "router") and
                        hasattr(layer.mlp.router, "_maintain_float32_expert_bias")
                    ):
                        layer.mlp.router._maintain_float32_expert_bias()

        return wrapped_model

    def get_model_from_bridge(
        self,
        bridge,
        hf_model_path,
        load_weights_from_bridge=False,
        model_type="model",
        wrap_with_ddp=True,
        build_value_model=False,
    ):
        if self.config.training.build_from_mbridge:
            return self.get_model_from_mbridge(
                bridge, hf_model_path, load_weights_from_bridge, model_type, wrap_with_ddp,
                build_value_model
            )
        else:
            return self.get_model_from_megatron_bridge(
                bridge, hf_model_path, load_weights_from_bridge, model_type, wrap_with_ddp,
                build_value_model
            )

    def build_megatron_bridge(self, hf_model_path, override_transformer_config=None):
        from gpatch_v4.training_backend.megatron_backend.megatron_bridge import (
            AutoBridge,
        )

        bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)
        provider, hf_config = self._megatron_bridge_set_extra_args(
            bridge, hf_model_path, override_transformer_config
        )
        self.provider = provider
        return bridge, hf_config

    def _megatron_bridge_set_extra_args(
        self, bridge, hf_model_path, override_transformer_config=None
    ):
        provider = bridge.to_megatron_provider(load_weights=False)
        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        provider.params_dtype = hf_config.torch_dtype
        provider.tensor_model_parallel_size = self.dist_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = self.dist_config.pipeline_model_parallel_size
        provider.expert_model_parallel_size = self.dist_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = self.dist_config.expert_tensor_parallel_size
        provider.virtual_pipeline_model_parallel_size = self.dist_config.virtual_pipeline_model_parallel_size
        provider.context_parallel_size = self.dist_config.context_parallel_size
        provider.sequence_parallel = self.dist_config.sequence_parallel
        if self.dist_config.num_layers_in_first_pipeline_stage is not None:
            provider.num_layers_in_first_pipeline_stage = self.dist_config.num_layers_in_first_pipeline_stage
        if self.dist_config.num_layers_in_last_pipeline_stage is not None:
            provider.num_layers_in_last_pipeline_stage = self.dist_config.num_layers_in_last_pipeline_stage

        provider.finalize_model_grads_func = finalize_model_grads
        provider.attention_backend = AttnBackend[self.config.training.attention_backend]
        provider.moe_token_dispatcher_type = self.config.training.moe_token_dispatcher_type
        provider.moe_router_load_balancing_type = self.config.training.moe_router_load_balancing_type
        if getattr(self.config.training, "moe_router_replay", False):
            # add args for moe router replay
            if hasattr(TransformerConfig, "enable_routing_replay"):
                provider.enable_routing_replay = self.config.training.moe_router_replay
            else:
                provider.moe_enable_routing_replay = self.config.training.moe_router_replay

        if self.config.training.enable_mtp:
            provider.mtp_num_layers = self.get_hf_mtp_num_layers(hf_config)
            if self.config.training.mtp_loss_scaling_factor is not None:
                provider.mtp_loss_scaling_factor = self.config.training.mtp_loss_scaling_factor
        else:
            provider.mtp_num_layers = None

        if self.config.training.enable_mtp and getattr(
            self.config.training, "online_mtp_sft", False
        ):
            assert (
                provider.mtp_num_layers is not None and provider.mtp_num_layers > 0
            ), f"online_mtp_sft is enabled while model got no mtp layers"
            assert hasattr(TransformerConfig, "online_mtp_sft")
            provider.online_mtp_sft = True

        #TODO: 看看需要还需要给 provider 补充什么参数
        if override_transformer_config is not None:
            assert isinstance(
                override_transformer_config, dict
            ), f"{type(override_transformer_config)=}"
            # 规定 override_transformer_config 优先级更高，会覆盖重复参数
            for k, v in override_transformer_config.items():
                setattr(provider, k, v)

        provider.finalize()
        return provider, hf_config

    def get_model_from_megatron_bridge(
        self,
        bridge,
        hf_model_path,
        load_weights_from_bridge=False,
        model_type="model",
        wrap_with_ddp=True,
        build_value_model=False,
    ):
        from gpatch_v4.training_backend.megatron_backend.megatron_bridge import (
            freeze_moe_router,
            freeze_multimodal,
            make_value_model,
        )

        post_model_creation_callbacks = []
        if self.training_config.freeze_moe_router:
            post_model_creation_callbacks.append(
                partial(
                    freeze_moe_router,
                    freeze_moe_shared_experts=self.training_config.freeze_moe_shared_experts
                )
            )
        if build_value_model:
            post_model_creation_callbacks.append(make_value_model)
        if any(
            [
                self.config.training.freeze_llm,
                self.config.training.freeze_vit,
                self.config.training.freeze_projector,
                self.config.training.freeze_audio,
                self.config.training.freeze_qformer,
            ]
        ):
            post_model_creation_callbacks.append(
                partial(
                    freeze_multimodal,
                    freeze_language_model=self.config.training.freeze_llm,
                    freeze_vision_model=self.config.training.freeze_vit,
                    freeze_vision_projection=self.config.training.freeze_projector,
                    freeze_audio_model=self.config.training.freeze_audio,
                    freeze_audio_qformer=self.config.training.freeze_qformer,
                    freeze_audio_projection=self.config.training.freeze_projector,
                )
            )
        post_model_creation_callbacks.append(log_freeze_status)

        self.peft = get_peft_cls(
            policy_config=self.policy_config,
            bridge=bridge,
            provider=self.provider,
            dtype=self.provider.params_dtype,
        )
        for callback in post_model_creation_callbacks:
            self.provider.register_pre_wrap_hook(callback)
        if self.peft is not None:
            self.provider.register_pre_wrap_hook(
                partial(
                    apply_peft_pre_wrap_hook,
                    peft=self.peft,
                    use_mbridge=False,
                    check_lora_all_coverage=self.policy_config.lora.check_lora_all_coverage,
                    verify_weight_consistency=self.policy_config.lora.verify_weight_consistency,
                )
            )

        ddp_config = None
        if wrap_with_ddp:
            from megatron.bridge.training.config import DistributedDataParallelConfig
            use_megatron_fsdp = self.policy_config.use_megatron_fsdp
            ddp_config_dict = build_megatron_bridge_ddp_config_dict(
                use_megatron_fsdp=use_megatron_fsdp,
                override_ddp_config=self.policy_config.override_ddp_config,
                build_from_mbridge=self.config.training.build_from_mbridge,
                wrap_with_ddp=wrap_with_ddp,
            )
            use_dist_opt = should_use_distributed_optimizer(self.config.optimizer)
            ddp_config_dict["use_distributed_optimizer"] = use_dist_opt

            ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
            ddp_config.finalize()

        with profile_memory_and_time(f"get {model_type} from megatron_bridge", rank=0):
            provide_model_kwargs = build_megatron_bridge_provide_model_kwargs(
                wrap_with_ddp=wrap_with_ddp,
                ddp_config=ddp_config,
                use_megatron_fsdp=self.policy_config.use_megatron_fsdp,
            )
            if self.policy_config.use_megatron_fsdp:
                provide_model_sig = inspect.signature(self.provider.provide_distributed_model)
                assert "use_megatron_fsdp" in provide_model_sig.parameters, (
                    "Megatron-Bridge provider.provide_distributed_model does not support "
                    "use_megatron_fsdp. Please use a Megatron-Bridge version with Megatron-FSDP support."
                )
                provide_model_kwargs["use_megatron_fsdp"] = True
            model = self.provider.provide_distributed_model(**provide_model_kwargs)

            tf_config = get_model_config(model[0] if isinstance(model, list) else model)

            if load_weights_from_bridge:
                logging_rank0(f"loading {model_type} weights from megatron_bridge {hf_model_path=}")
                allowed_mismatched_params = []
                load_model = get_megatron_bridge_weight_load_model(
                    model,
                    use_megatron_fsdp=self.policy_config.use_megatron_fsdp,
                )
                bridge.load_hf_weights(
                    load_model, hf_model_path, allowed_mismatched_params=allowed_mismatched_params
                )

            if self.peft is not None:
                self.peft.set_params_to_save(model)

        return model, tf_config

    # build from mbridge function
    def build_mbridge(self, hf_model_path, override_transformer_config=None):
        from gpatch_v4.training_backend.megatron_backend.mbridge import AutoBridge
        with profile_memory_and_time("build bridge", rank=0):
            hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
            from_config_sig = inspect.signature(AutoBridge.from_config)
            export_buf_bytes = self.policy_config.export_weights_buffer_max_size_mb * 1024**2
            extra_args = {"export_weights_buffer_max_size_bytes": export_buf_bytes}
            if "mcore_extra_config" in from_config_sig.parameters:
                extra_args["mcore_extra_config"] = override_transformer_config

            bridge = AutoBridge.from_config(
                hf_config,
                **extra_args,
            )

            bridge = self._mbridge_set_extra_args(bridge, hf_config, override_transformer_config)
            bridge.config = merge_core_transformer_config(bridge.config, self.config)
            logging_rank0(f"bridge.config: {bridge.config}")
        return bridge, hf_config

    def _mbridge_set_extra_args(self, bridge, hf_config, override_transformer_config=None):
        extra_kwargs = {}
        if self.dist_config.num_layers_in_first_pipeline_stage is not None:
            extra_kwargs["num_layers_in_first_pipeline_stage"
                        ] = self.dist_config.num_layers_in_first_pipeline_stage
        if self.dist_config.num_layers_in_last_pipeline_stage is not None:
            extra_kwargs["num_layers_in_last_pipeline_stage"
                        ] = self.dist_config.num_layers_in_last_pipeline_stage

        extra_kwargs["finalize_model_grads_func"] = finalize_model_grads
        extra_kwargs["attention_backend"] = AttnBackend[self.config.training.attention_backend]
        extra_kwargs["variable_seq_lengths"] = True
        extra_kwargs["dynamic_context_parallel"] = self.dist_config.dynamic_context_parallel
        extra_kwargs["max_seqlen_per_dp_cp_rank"] = self.dist_config.max_seqlen_per_dp_cp_rank
        extra_kwargs["min_dynamic_context_parallel_size"
                    ] = self.dist_config.min_dynamic_context_parallel_size
        extra_kwargs["moe_token_dispatcher_type"] = self.config.training.moe_token_dispatcher_type
        extra_kwargs["moe_router_load_balancing_type"
                    ] = self.config.training.moe_router_load_balancing_type

        # add override_transformer_config for moe router replay
        if getattr(self.config.training, "moe_router_replay", False):
            if hasattr(TransformerConfig, "enable_routing_replay"):
                extra_kwargs["enable_routing_replay"] = True
            else:
                extra_kwargs["moe_enable_routing_replay"] = True

        if self.config.training.enable_mtp:
            extra_kwargs["mtp_num_layers"] = self.get_hf_mtp_num_layers(hf_config)
            if self.config.training.mtp_loss_scaling_factor is not None:
                extra_kwargs["mtp_loss_scaling_factor"
                            ] = self.config.training.mtp_loss_scaling_factor
        else:
            extra_kwargs["mtp_num_layers"] = None

        if self.config.training.enable_mtp and getattr(
            self.config.training, "online_mtp_sft", False
        ):
            assert (
                extra_kwargs["mtp_num_layers"] is not None and extra_kwargs["mtp_num_layers"] > 0
            ), f"online_mtp_sft is enabled while model got no mtp layers"
            assert hasattr(TransformerConfig, "online_mtp_sft")
            extra_kwargs["online_mtp_sft"] = True

        if override_transformer_config is not None:
            assert isinstance(
                override_transformer_config, dict
            ), f"{type(override_transformer_config)=}"
            # 规定 override_transformer_config 优先级更高，会覆盖重复参数
            for k, v in override_transformer_config.items():
                extra_kwargs[k] = v

        bridge.set_extra_args(**extra_kwargs)
        return bridge

    def get_model_from_mbridge(
        self,
        bridge,
        hf_model_path,
        load_weights_from_bridge=False,
        model_type="model",
        wrap_with_ddp=True,
        build_value_model=False,
    ):
        from gpatch_v4.training_backend.megatron_backend.mbridge import (
            apply_freeze_unfreeze_patterns,
            freeze_moe_router,
            freeze_multimodal,
            make_value_model,
        )

        # register extra bridge
        # 1、预期 mcore 可能后面还有些修改， dsv4 mbridge 适配暂时放在 mcore extend model 目录里, 待时机合适再提PR 到 开源 repo
        # 2、内部模型的 mbridge 适配放在 mcore extend model 目录里可能也是个不错的选择
        try:
            import megatron.core.extended_models
        except:
            pass

        kwargs = {}
        post_model_creation_callbacks = []
        kwargs["post_model_creation_callbacks"] = post_model_creation_callbacks
        if self.training_config.freeze_moe_router:
            post_model_creation_callbacks.append(
                partial(
                    freeze_moe_router,
                    freeze_moe_shared_experts=self.training_config.freeze_moe_shared_experts
                )
            )

        if build_value_model:
            post_model_creation_callbacks.append(make_value_model)

        if self.config.training.freeze_llm or self.config.training.freeze_vit or self.config.training.freeze_projector or self.config.training.freeze_audio or self.config.training.freeze_qformer:
            post_model_creation_callbacks.append(
                partial(
                    freeze_multimodal,
                    freeze_language_model=self.config.training.freeze_llm,
                    freeze_vision_model=self.config.training.freeze_vit,
                    freeze_vision_projection=self.config.training.freeze_projector,
                    freeze_audio_model=self.config.training.freeze_audio,
                    freeze_audio_qformer=self.config.training.freeze_qformer,
                    freeze_audio_projection=self.config.training.freeze_projector,
                )
            )
        post_model_creation_callbacks.append(log_freeze_status)

        self.peft = get_peft_cls(
            policy_config=self.policy_config,
            bridge=bridge,
            provider=None,
            dtype=None,
            use_mbridge=True,
        )

        with profile_memory_and_time(f"get {model_type} from mbridge", rank=0):
            post_wrap_with_ddp = (
                wrap_with_ddp and self.policy_config.post_wrap_with_ddp and load_weights_from_bridge
            )
            # When PEFT is enabled, always defer DDP wrapping so we can
            # load base weights first, then inject adapters, then wrap.
            post_wrap_with_ddp = post_wrap_with_ddp or (self.peft is not None and wrap_with_ddp)
            # ddp wrap 之后，不能再 freeze/unfreeze 参数，所以需要提前判断是否需要 freeze/unfreeze 参数了
            has_freeze_patterns = bool(
                self.policy_config.freeze_patterns or self.policy_config.unfreeze_patterns
            )
            post_wrap_with_ddp = post_wrap_with_ddp or (wrap_with_ddp and has_freeze_patterns)
            # mbridge DDP defaults to use_distributed_optimizer=True; defer wrap so we can
            # set it False for layer-wise emerging optimizers (needs all-reduce).
            if wrap_with_ddp and not should_use_distributed_optimizer(self.config.optimizer):
                post_wrap_with_ddp = True
            # mbridge defaults this to True and broadcasts DP params at wrap;
            # pass GCore's flag (default False) so HF-load jobs skip that collective.
            model = bridge.get_model(
                bf16=True,
                wrap_with_ddp=wrap_with_ddp and not post_wrap_with_ddp,
                data_parallel_random_init=self.config.training.data_parallel_random_init,
                ddp_config=self.policy_config.override_ddp_config,
                **kwargs,
            )
            if load_weights_from_bridge:
                logging_rank0(f"loading {model_type} weights from mbridge {hf_model_path=}")
                bridge.load_weights(model, hf_model_path, memory_efficient=True)
            else:
                bridge.safetensor_io = bridge._get_safetensor_io(hf_model_path)

            if self.peft is not None:

                for model_chunk in model:
                    apply_peft_pre_wrap_hook(
                        model_chunk,
                        peft=self.peft,
                        use_mbridge=True,
                        check_lora_all_coverage=self.policy_config.lora.check_lora_all_coverage,
                        verify_weight_consistency=self.policy_config.lora.verify_weight_consistency,
                    )

            if self.policy_config.freeze_patterns or self.policy_config.unfreeze_patterns:
                for model_chunk in model:
                    apply_freeze_unfreeze_patterns(
                        model_chunk,
                        freeze_patterns=self.policy_config.freeze_patterns,
                        unfreeze_patterns=self.policy_config.unfreeze_patterns,
                    )

            if post_wrap_with_ddp:
                model = self.wrap_mbridge_model_with_ddp(model)
        return model, bridge.config


class CheckpointMixin:
    def save_checkpoint(self, global_step: int, dataloader=None):
        override_tokenizer_special_token = None

        if len(self.config.checkpoint.override_tokenizer_special_token) > 0:
            raw_override_tokenizer_special_token = self.config.checkpoint.override_tokenizer_special_token

            override_tokenizer_special_token = dict()
            for k, v in raw_override_tokenizer_special_token.items():
                token_id = self.tokenizer(v).input_ids[0]
                override_tokenizer_special_token[k] = [token_id, v]

        save_checkpoint(
            self.config,
            self.model,
            self.optimizer,
            self.optimizer_scheduler,
            global_step,
            self.bridge,
            override_tokenizer_special_token=override_tokenizer_special_token,
            peft=getattr(self, "peft", None),
            use_megatron_fsdp=self.policy_config.use_megatron_fsdp,
        )
        self._save_dataloader_state(global_step, dataloader)

    def _save_dataloader_state(self, global_step: int, dataloader=None):
        """Persist savable dataloader state under the checkpoint root.

        Writes ``{save_ckpt_path}/dataloader/iter_XXXXXXX/dp_rank_YYY.pt``.

        Only the MP+CP head of each DP group writes: TP/PP/CP ranks share the
        same ``dp_rank`` and would otherwise race on the same file. Restore
        still loads that file on every rank in the DP group.

        No-op when ``dataloader`` is None or does not implement ``save_state``.
        """
        if dataloader is None or not hasattr(dataloader, "save_state"):
            return
        if not is_mp_and_cp_head():
            return
        save_ckpt_path = self.config.checkpoint.save_ckpt_path
        if not save_ckpt_path:
            return
        dp_rank = mpu.get_data_parallel_rank()
        out_dir, state_path = get_dataloader_save_path(self.config.checkpoint, global_step, dp_rank)
        assert out_dir is not None and state_path is not None, f"get_dataloader_save_path failed"
        os.makedirs(out_dir, exist_ok=True)
        state = dataloader.save_state()
        # Megatron / ``energon checkpoint redist`` expect
        # ``{"dataloader_state_dict": SavableDataLoaderState}``. Our Energon
        # wrapper also carries lightweight ``checkpoint_metadata``; keep that
        # beside the official key so redist can ignore it.
        if isinstance(state, dict) and "loader_state" in state:
            save_obj = {
                "dataloader_state_dict": state["loader_state"],
                "gcore_checkpoint_metadata": state.get("checkpoint_metadata") or {},
            }
        else:
            save_obj = {"dataloader_state_dict": state}
        torch.save(save_obj, state_path)
        logging_rank0(f"saved dataloader state to {state_path}")

    def convert_to_hf_checkpoint(self):
        assert not self.checkpoint_config.skip_save_mcore_model, f"the model weight not save in checkpoint"
        if self.checkpoint_config.convert_target_step is not None:
            # 临时修改 latest_checkpointed_step
            target_step = int(self.checkpoint_config.convert_target_step)
        else:
            target_step = get_latest_checkpoint_folder(self.checkpoint_config.save_ckpt_path)
            assert target_step is not None, f"Cannot find latest checkpoint in {self.checkpoint_config.save_ckpt_path}"

        self.model, tf_config = self.get_model_from_mbridge(
            self.bridge,
            self.policy_config.hf_model_path,
            load_weights_from_bridge=False,
            model_type=f"policy model",
            wrap_with_ddp=self.policy_config.wrap_with_ddp,
        )
        _step = load_checkpoint(
            self.config, self.model, None, None, target_step, bridge=self.bridge
        )
        assert _step == target_step, f"Loaded checkpoint step {_step} does not match target step {target_step}"

        override_tokenizer_special_token = None
        if len(self.config.checkpoint.override_tokenizer_special_token) > 0:
            raw_override_tokenizer_special_token = self.config.checkpoint.override_tokenizer_special_token
            override_tokenizer_special_token = dict()
            for k, v in raw_override_tokenizer_special_token.items():
                token_id = self.tokenizer(v).input_ids[0]
                override_tokenizer_special_token[k] = [token_id, v]
            log(f"Inspect {override_tokenizer_special_token=}", rank=0)
        bridge_save_hf(
            self.config,
            target_step,
            self.model,
            self.bridge,
            override_tokenizer_special_token=override_tokenizer_special_token
        )


class RouterReplayMixin:
    def get_mcore_config(self):
        model = unwrap_model(self.model)
        model_config = get_model_config(model[0] if isinstance(model, list) else model)
        return model_config

    def get_router_replay_manager(self):
        return RouterReplayManager.get_instance()

    def get_router_replay_ctx(self):
        ctx = RouterReplayCtx.get_instance(
        ) if self.config.training.moe_router_replay else nullcontext()
        return ctx

    def get_cur_cp_tp_index(self, index, *, dynamic_cp: bool = False):
        config = self.get_mcore_config()
        # 这里其实可以不加这个判断的，但是为了稳妥还是加上了，dynamic cp 的恒为 cp size == 1
        if not dynamic_cp and config.context_parallel_size > 1:
            index = get_tensor_on_this_cp_rank(index, seq_dim=0)
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            index = index.view(config.tensor_model_parallel_size,
                               -1)[mpu.get_tensor_model_parallel_rank()]
        return index

    @torch.no_grad()
    def prepare_for_router_replay(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        *,
        packed_thd: bool = False,
    ):
        config = self.get_mcore_config()
        num_layer = get_num_layers_to_build(config)
        offset = get_transformer_layer_offset(config)
        routed_experts = []
        for batch in batches:
            # seq, layer, experts
            routed_experts_batch = batch["routed_experts"]
            assert routed_experts_batch.ndim == 3, (
                f"routed_experts must be [seq, layer, topk], got "
                f"{routed_experts_batch.shape=}"
            )
            if packed_thd:
                # Dynamic CP already applied the exact THD partition index in
                # grpo_train_with_dynamic_cp. Only TP sequence parallel remains.
                assert len(batches) == 1, "packed THD replay expects one packed microbatch"
                assert routed_experts_batch.shape[0] == seqlen, (
                    "packed THD replay must already match the local token shard: "
                    f"{routed_experts_batch.shape[0]=} != {seqlen=}"
                )
                index = torch.arange(seqlen, device="cuda").long()
                index = self.get_cur_cp_tp_index(index, dynamic_cp=True)
            else:
                full_index = torch.arange(seqlen, device="cuda").long()
                # real seqlen of the request before padding - 1, last token is not
                # calculated by inference engine (hence no router), nor by train loss.
                seq_batch = routed_experts_batch.shape[0] - 1
                assert seq_batch > 0
                # index = full_index[:seq_batch] + some kind of random router padding
                index = (
                    full_index + (full_index // seq_batch) * 6 * mpu.get_data_parallel_rank()
                ) % seq_batch
                # support fixed CP and sequence_parallel
                index = self.get_cur_cp_tp_index(index)
            routed_experts_truncated = routed_experts_batch[:, offset:offset + num_layer]
            routed_experts_truncated = routed_experts_truncated.cuda(non_blocking=True)
            routed_experts_padded = routed_experts_truncated[index]
            routed_experts.append(routed_experts_padded)

        layers_routered_experts = []
        for i in range(num_layer):
            # list of tensor with shape [seq, expert] of size b
            layer_routered_experts_list = [e[:, i] for e in routed_experts]
            # seq, b, experts
            # 2 * experts_per_token * pp layer * seqlen may occupy about dozens of Ms of memory, it is ok to put in gpu memory
            layer_routered_experts = torch.stack(layer_routered_experts_list,
                                                 dim=1).contiguous().view(
                                                     -1, config.moe_router_topk
                                                 )
            layers_routered_experts.append(layer_routered_experts)

        self.get_router_replay_manager().append_micro_batch(layers_routered_experts)

    def maybe_prepare_for_router_replay(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        *,
        packed_thd: bool = False,
    ):
        if self.get_router_replay_manager().enabled:
            self.prepare_for_router_replay(batches, seqlen, packed_thd=packed_thd)

    def router_replay_pre_forward(self):
        """"
        pre forward may be the post backward of a microbatch
        """
        mgr = self.get_router_replay_manager()
        if mgr.enabled:
            mgr.set_replay_action(RouterReplayAction.REPLAY_FORWARD)
        else:
            mgr.clear_replay_action()

    def router_replay_post_forward(self):
        """"
        post forward may be the pre backward of a microbatch
        """
        mgr = self.get_router_replay_manager()
        if mgr.enabled:
            if torch.is_grad_enabled():
                mgr.set_replay_action(RouterReplayAction.REPLAY_BACKWARD)
            else:
                mgr.clear_indices()

    @contextmanager
    def router_replay_step_ctx(self):
        try:
            self.router_replay_pre_forward()
            yield
        finally:
            self.router_replay_post_forward()

    @contextmanager
    def disable_moe_router_replay(self):
        moe_router_replay = self.config.training.moe_router_replay
        try:
            self.config.training.moe_router_replay = False
            yield
        finally:
            self.config.training.moe_router_replay = moe_router_replay


def _gather_and_unpack_contiguous_thd_logprobs(
    local_logprobs: torch.Tensor,
    packed_seq_params,
    output_width: int,
) -> torch.Tensor:
    """Restore fixed-width per-sample log-probs from contiguous CP THD shards."""
    cp_group = mpu.get_context_parallel_group()
    gathered_logprobs = [torch.empty_like(local_logprobs) for _ in range(cp_group.size())]
    dist.all_gather(gathered_logprobs, local_logprobs.contiguous(), group=cp_group)
    packed_logprobs = torch.cat(gathered_logprobs, dim=1)

    cu_seqlens = packed_seq_params.cu_seqlens_q
    cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
    assert cu_seqlens is not None and cu_seqlens_padded is not None
    per_sample_logprobs = packed_logprobs.new_zeros(cu_seqlens.numel() - 1, output_width)
    for sample_idx, (logical_start, logical_end, physical_start) in enumerate(
        zip(cu_seqlens[:-1], cu_seqlens[1:], cu_seqlens_padded[:-1], strict=True)
    ):
        sample_length = min(int(logical_end.item()) - int(logical_start.item()), output_width)
        start = int(physical_start.item())
        per_sample_logprobs[sample_idx, :sample_length] = packed_logprobs[0, start:start +
                                                                          sample_length]
    return per_sample_logprobs


class ForwardStepMixin(RouterReplayMixin):
    @property
    def calc_per_token_loss(self) -> bool:
        """Whether per-token gradient normalization is enabled."""
        model = self.model[0] if isinstance(self.model, list) else self.model
        return get_model_config(model).calculate_per_token_loss

    def get_logprob_temperature(self) -> float:
        """Temperature for actor/ref logprob recomputation.

        When ``ppo.use_original_logprob`` is True (default), the sampler
        returns pre-temperature logprobs, so actor/ref recompute with
        temperature=1.0. When False, recompute with the sampler generate
        temperature so both logprobs live on the same (post-temperature)
        scale for importance sampling.
        """
        ppo = getattr(self.config, "ppo", None)
        if ppo is None or ppo.use_original_logprob:
            return 1.0
        sampler = getattr(self.config, "sampler", None)
        if sampler is not None:
            engines = sampler.infer_engine_configs
            if engines:
                generate_params = engines[0].generate_params
                assert float(
                    generate_params.temperature
                ) > 0, f"temperature must be positive, got {generate_params.temperature}"
                return float(generate_params.temperature)
        return 1.0

    def get_im_end_metrics(
        self,
        parallel_logits: torch.Tensor,
        target: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        ignore_cp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not self.config.training.im_end_metrics_enable:
            return {}

        im_end_token_id = get_im_end_metrics_token_id(self.tokenizer)
        im_end_prob, im_end_rank = from_parallel_logits_to_token_prob_and_rank(
            vocab_parallel_logits=parallel_logits,
            token_id=im_end_token_id,
            ignore_cp=ignore_cp,
        )
        im_end_prob = im_end_prob[:, :-1].contiguous()
        im_end_rank = im_end_rank[:, :-1].contiguous()

        mask_bool = response_mask.to(device=im_end_prob.device).bool()
        target_ids = target.roll(shifts=-1, dims=-1)[:, :-1].to(device=im_end_prob.device)
        nl_mask = mask_bool & (target_ids != im_end_token_id)

        metrics = {}
        metric_values = {
            "p_mean": im_end_prob,
            "p_top1": (im_end_rank < 1).to(torch.float32),
            "p_top5": (im_end_rank < 5).to(torch.float32),
            "p_top50": (im_end_rank < 50).to(torch.float32),
        }
        mask_float = mask_bool.to(torch.float32)
        nl_mask_float = nl_mask.to(torch.float32)
        for suffix, values in metric_values.items():
            metrics[f"eos/im_end/{suffix}"] = torch.stack(
                [(values * mask_float).sum(), mask_float.sum()]
            )
            metrics[f"eos/im_end/{suffix}_NL"] = torch.stack(
                [(values * nl_mask_float).sum(),
                 nl_mask_float.sum()]
            )
        return metrics

    def _aggregate_metrics(
        self,
        metrics_micro_batch: List[Dict[str, torch.Tensor]],
        skip_keys: tuple = (),
    ) -> Dict[str, float]:
        """Aggregate per-microbatch metrics across DP+CP ranks.

        Two-element [sum, count] tensors are weight-averaged via all_reduce;
        scalars are simple-averaged across microbatches.
        Must be called on all pipeline-last-stage ranks (all_reduce collective).
        """
        result: Dict[str, float] = {}
        if not is_pipeline_last_stage() or not metrics_micro_batch:
            return result
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        for key in metrics_micro_batch[0]:
            if key in skip_keys:
                continue
            vals = [m[key] for m in metrics_micro_batch]
            if vals[0].numel() == 2:
                stacked = torch.vstack(vals).sum(dim=0)  # 跨 microbatch 求和
                torch.distributed.all_reduce(stacked, group=dp_cp_group)
                result[key] = (stacked[0] / torch.clamp(stacked[1], min=1)).cpu().item()
            else:
                result[key] = torch.stack(vals).mean().cpu().item()
        return result

    def get_logprob_output_only_func(
        self,
        seq_len,
        inference_only=True,
        compute_topk: bool = False,
        gather_target_ids_key: str = None,
        return_per_token_entropy: bool = False,
    ):
        """
        Args:
            compute_topk : bool, optional
                (used by student) if true, also gather the top-K log-probs / ids.
            gather_target_ids_key : str, optional
                Per-sample key in the batch dict whose shape is ``[S-1, K]``;
                when set, also gather log-probs at those ids
                (used by ref / teacher on ``stu_topk_ids``).

        设置 top-K 相关参数时，返回 dict 而非单个 tensor。
        当 Linear CE 或 compact CE 启用时，直接从 hidden states 计算 logprobs。
        """
        if self.training_config.return_hidden_states_for_ce:
            assert not compute_topk, (
                "Linear CE and compact CE 不支持 compute_topk（dynamic top-K 需完整 logits）；"
                "gather_target_ids_key（已知 ids gather）走 opd_topk_logprobs_from_linear_ce 融合路径"
            )

        def log_prob_output_only_func(seq_len, dataloader_iter, model):
            # for r3
            with self.router_replay_step_ctx():
                batches: List[Dict[str, Any]] = next(dataloader_iter)
                # log for batch_get_*_logprobs
                self.batch_iters += 1
                if self.total_iters > 0 and torch.distributed.get_rank() == 0:
                    log(f"{self.batch_log_str} {self.batch_iters:8d}/{self.total_iters:8d}", rank=0)
                # for r3
                self.maybe_prepare_for_router_replay(batches, seq_len)
                use_thd_pack = (
                    self.policy_config.ppo_pack_seq and
                    not self.dist_config.dynamic_context_parallel and
                    not self.policy_config.smart_pad_infer
                )
                model_fwd_args = self.prepare_data.model_forward_only(
                    batches,
                    seq_len,
                    self.tokenizer.pad_token_id,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self.vocab_size,
                    ppo_pack_seq=use_thd_pack,
                )
                target = model_fwd_args.pop("target")
                packed_seq_params = model_fwd_args.get("packed_seq_params")

                # Protocol flags injected by model_forward_only implementations that
                # pre-pack inputs into THD format with contiguous CP slicing (e.g.
                # DeepseekV4PrepareDataForwardLLM._model_forward_only_mcore_thd).
                #
                # _logprob_ignore_cp=True
                #   The caller has already CP-sliced target contiguously; skip
                #   from_parallel_logits_to_logprobs' internal CP reorder/slice/gather.
                #
                # _logprob_pre_shifted=True
                #   target is already next-token shifted (tok[1:]); skip the internal
                #   roll(-1) and trailing [:, :-1] truncation.
                #
                # _logprob_contiguous_gather=True
                #   After computing local logprobs [1, T/cp], perform an explicit
                #   contiguous all-gather (dist.all_gather + torch.cat) to reconstruct
                #   [1, T].  This differs from Megatron's zigzag all_gather_from_cp.
                #
                _logprob_ignore_cp = model_fwd_args.pop("_logprob_ignore_cp", False)
                _logprob_pre_shifted = model_fwd_args.pop("_logprob_pre_shifted", False)
                _logprob_contiguous_gather = model_fwd_args.pop("_logprob_contiguous_gather", False)
                if _logprob_contiguous_gather:
                    assert packed_seq_params is not None, (
                        "contiguous THD logprobs require packed_seq_params"
                    )

                use_linear_ce = self.training_config.use_linear_ce
                fp32_output = not self.training_config.return_hidden_states_for_ce
                if use_thd_pack:
                    # Static ppo_pack_seq path unpacks to BSHD logits; Linear CE
                    # and compact CE are not wired here yet (same as grpo_train).
                    assert not self.training_config.return_hidden_states_for_ce, "暂不支持"
                    seq_lens = torch.stack([b["sequence_lengths"]
                                            for b in batches]).cuda(non_blocking=True)
                    pack_batch = {"sequence_lengths": seq_lens}
                    model_output = gptmodel_pack_foward(
                        unwrap_model(model), pack_batch, model_fwd_args, self.config
                    )
                else:
                    model_output = model(**model_fwd_args, fp32_output=fp32_output)

                ce_compaction_mask = None
                if self.training_config.ce_compaction:
                    # Per-sample rollout → [B, S-1]; train_step later uses batch["mask"].
                    ce_compaction_mask = build_grpo_compact_ce_mask(batches, target)

                if self.config.training.forward_clear_memory and \
                   self.batch_iters % self.config.training.forward_clear_memory_interval == 0:
                    logging_rank0(f"clear memory in forward only {self.batch_iters=}")
                    clear_memory()

                if isinstance(model_output, tuple):
                    model_output = model_output[0]
                if self.training_config.return_hidden_states_for_ce and mpu.is_pipeline_last_stage():
                    assert isinstance(model_output, dict)
                    assert model_output["hidden_states"].dtype == torch.bfloat16
                    linear_ce_output = model_output
                    model_output = model_output["hidden_states"]
                else:
                    assert isinstance(model_output, torch.Tensor)

                def id_func(model_output, non_loss_data=True):
                    logprob_temperature = self.get_logprob_temperature()
                    if use_linear_ce:
                        linear_ce_result = logprobs_from_linear_ce(
                            linear_ce_backend=self.training_config.linear_ce_backend,
                            linear_ce_output=linear_ce_output,
                            target=target,
                            ignore_cp=_logprob_ignore_cp,
                            pre_shifted=_logprob_pre_shifted,
                            mask=ce_compaction_mask,
                            return_entropy=return_per_token_entropy,
                            temperature=logprob_temperature,
                            token_compaction=self.training_config.ce_compaction,
                        )
                        if return_per_token_entropy:
                            logprobs, _, per_token_entropy = linear_ce_result
                        else:
                            logprobs = linear_ce_result
                    elif self.training_config.ce_compaction:
                        compact_ce_result = logprobs_from_compact_ce(
                            linear_ce_output=linear_ce_output,
                            target=target,
                            mask=ce_compaction_mask,
                            ignore_cp=_logprob_ignore_cp,
                            pre_shifted=_logprob_pre_shifted,
                            return_entropy=return_per_token_entropy,
                            temperature=logprob_temperature,
                        )
                        if return_per_token_entropy:
                            logprobs, _, per_token_entropy = compact_ce_result
                        else:
                            logprobs = compact_ce_result
                    else:
                        # TODO(@nrwu): 检查 sp 情况下，此处 output tensor shape 是否应该是 [b, s, v//tp] ？
                        if compute_topk or gather_target_ids_key is not None:
                            output_tensor_for_topk = model_output.clone()
                        logprobs = from_parallel_logits_to_logprobs(
                            vocab_parallel_logits=model_output,
                            target=target,
                            inference_only=inference_only,
                            ignore_cp=_logprob_ignore_cp or use_thd_pack,
                            pre_shifted=_logprob_pre_shifted,
                            temperature=logprob_temperature,
                            # thd pack already gathers logits across CP back to full
                            # [b, s, v//tp]; skip the per-CP slice/gather here (mirrors
                            # the training forward at grpo_train).
                        )
                        if return_per_token_entropy:
                            _, per_token_entropy = vocab_parallel_entropy(model_output.float())
                    if not compute_topk and gather_target_ids_key is None and not return_per_token_entropy:
                        if _logprob_contiguous_gather:
                            logprobs = _gather_and_unpack_contiguous_thd_logprobs(
                                logprobs, packed_seq_params, seq_len - 1
                            )
                        return logprobs

                    # thd is not supported yet for the logic below
                    assert not _logprob_ignore_cp
                    assert not _logprob_pre_shifted
                    assert not _logprob_contiguous_gather

                    result = {"logprobs": logprobs}
                    if return_per_token_entropy:
                        result["prev_per_token_entropy"] = per_token_entropy
                    if compute_topk:
                        topk_logprobs, topk_token_ids = from_parallel_logits_to_topk_logprobs(
                            vocab_parallel_logits=output_tensor_for_topk,
                            topk=getattr(self.ppo_config, "log_prob_top_k", 0),
                            ignore_cp=use_thd_pack,
                            temperature=logprob_temperature,
                        )
                        result["topk_logprobs"] = topk_logprobs[:, :-1].contiguous()
                        result["topk_ids"] = topk_token_ids[:, :-1].to(torch.int32).contiguous()
                    if gather_target_ids_key is not None:
                        # 将各 sample 的 [S_i-1, K] ids pad 到 [B, seq_len, K]，超出部分填 0。
                        per_sample = [b[gather_target_ids_key] for b in batches]
                        k_dim = per_sample[0].shape[-1]
                        batch_size = target.shape[0]
                        ids_device = target.device
                        padded_ids = torch.zeros(
                            (batch_size, seq_len, k_dim),
                            dtype=torch.long,
                            device=ids_device,
                        )
                        for i, ids in enumerate(per_sample):
                            li = min(ids.shape[0], seq_len)
                            padded_ids[i, :li] = ids[:li].to(ids_device, dtype=torch.long)
                        if use_linear_ce:
                            gather_lp = opd_topk_logprobs_from_linear_ce(
                                linear_ce_backend=self.training_config.linear_ce_backend,
                                linear_ce_output=linear_ce_output,
                                target_ids=padded_ids,
                                ignore_cp=False,
                                temperature=logprob_temperature,
                            )
                        else:
                            gather_lp = from_parallel_logits_to_opd_topk_logprobs(
                                vocab_parallel_logits=output_tensor_for_topk,
                                target_ids=padded_ids,
                                ignore_cp=use_thd_pack,
                                temperature=logprob_temperature,
                            )
                        result["gather_logprobs"] = gather_lp[:, :-1].contiguous()
                    return result

                return model_output, id_func

        return partial(log_prob_output_only_func, seq_len)

    def get_logprob_output_only_func_dynamic_cp(self, seqlen):
        """Forward step func for dynamic CP path (THD packed format).

        Compatible with Megatron's ``forward_backward_func``, enabling PP support
        and progress logging. Each microbatch is a pre-packed Dict[str, Tensor]
        from ``rl_reroute_data_for_dynamic_cp``.
        """
        def log_prob_output_only_func_dynamic_cp(seqlen, dataloader_iter, model):
            with self.router_replay_step_ctx():
                mb_data = next(dataloader_iter)

                self.batch_iters += 1
                if self.total_iters > 0 and torch.distributed.get_rank() == 0:
                    log(
                        f"{self.batch_log_str} {self.batch_iters:8d}/{self.total_iters:8d}",
                        rank=0,
                    )

                mb_copy = dict(mb_data)
                batch, fwd_kwargs = self.prepare_data.grpo_train_with_dynamic_cp(
                    [mb_copy],
                    seqlen=seqlen,
                    pad_token_id=self.tokenizer.pad_token_id,
                    ppo_pack_seq=self.policy_config.ppo_pack_seq,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self.vocab_size,
                )
                cp_partition_mode = _cp_partition_mode_from_fwd_kwargs(fwd_kwargs)
                self.maybe_prepare_for_router_replay(
                    [batch],
                    batch["tokens"].shape[-1],
                    packed_thd=True,
                )

                ce_compaction_mask = None
                if self.training_config.ce_compaction:
                    # Packed [1, T] → this rank's [1, T_local], matching local hidden/target.
                    ce_compaction_mask = build_grpo_compact_ce_mask_dyn_cp(
                        batch,
                        batch["target"],
                        cp_partition_mode=cp_partition_mode,
                    )
                fp32_output = not self.training_config.return_hidden_states_for_ce
                model_output = model(**fwd_kwargs, fp32_output=fp32_output)
                if isinstance(model_output, tuple):
                    model_output = model_output[0]

                if self.training_config.return_hidden_states_for_ce and mpu.is_pipeline_last_stage():
                    assert isinstance(model_output, dict)
                    linear_ce_output = model_output
                    model_output = model_output["hidden_states"]
                else:
                    if mpu.is_pipeline_last_stage():
                        assert isinstance(model_output, torch.Tensor)

            def id_func(model_output, non_loss_data=True):
                target = batch["target"]
                logprob_temperature = self.get_logprob_temperature()
                if self.training_config.use_linear_ce:
                    logprobs = logprobs_from_linear_ce(
                        linear_ce_backend=self.training_config.linear_ce_backend,
                        linear_ce_output=linear_ce_output,
                        target=target,
                        ignore_cp=True,
                        pre_shifted=True,
                        temperature=logprob_temperature,
                        mask=ce_compaction_mask,
                        token_compaction=self.training_config.ce_compaction,
                    )
                elif self.training_config.ce_compaction:
                    logprobs = logprobs_from_compact_ce(
                        linear_ce_output=linear_ce_output,
                        target=target,
                        mask=ce_compaction_mask,
                        ignore_cp=True,
                        pre_shifted=True,
                        temperature=logprob_temperature,
                    )
                else:
                    logprobs = from_parallel_logits_to_logprobs(
                        vocab_parallel_logits=model_output.float(),
                        target=target,
                        ignore_cp=True,
                        pre_shifted=True,
                        temperature=logprob_temperature,
                    )

                # Reassemble from CP shards if local_cp_size > 1.
                lcp = batch.get("local_cp_size")
                lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
                if lcp_val > 1:
                    cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
                    total_tokens = mb_data["tokens"].size(0)
                    if cp_partition_mode == "contiguous":
                        gathered = [torch.empty_like(logprobs) for _ in range(lcp_val)]
                        torch.distributed.all_gather(
                            gathered, logprobs.contiguous(), group=cp_group
                        )
                        logprobs = torch.cat(gathered, dim=1)
                        assert logprobs.size(1) == total_tokens
                    else:
                        cp_rank = cp_group.rank()
                        index = get_thd_partitioned_indices(
                            mb_data["cu_seqlens_padded"], total_tokens, lcp_val, cp_rank
                        )
                        full_lp = torch.zeros(
                            1, total_tokens, device=logprobs.device, dtype=logprobs.dtype
                        )
                        full_lp.scatter_(1, index.unsqueeze(0), logprobs)
                        torch.distributed.all_reduce(
                            full_lp, group=cp_group, op=torch.distributed.ReduceOp.SUM
                        )
                        logprobs = full_lp

                # Split packed logprobs back to per-sample tensors.
                cu_seqlens = mb_data["cu_seqlens"]
                cu_seqlens_padded = mb_data["cu_seqlens_padded"]
                num_samples = cu_seqlens.shape[0] - 1
                results = []
                logprobs_flat = logprobs.squeeze(0)
                for s_idx in range(num_samples):
                    pad_start = cu_seqlens_padded[s_idx].item()
                    orig_len = cu_seqlens[s_idx + 1].item() - cu_seqlens[s_idx].item()
                    results.append(logprobs_flat[pad_start:pad_start + orig_len])
                return results

            return model_output, id_func

        return partial(log_prob_output_only_func_dynamic_cp, seqlen)

    def _get_dpo_forward_seq_length(self, batches: List[Dict[str, Any]]) -> int:
        """Return the shared padded sequence length for DPO ref and train forwards."""
        if self.config.debug.experimental_pad_to_max_length:
            return self.training_config.seq_length

        seq_length = get_batches_max_seqlen(batches, self.training_config.pad_to_mulitiple_of)
        # Keep reference and policy training forwards in the same DP-reduced
        # padded shape so their deterministic padding tokens also stay aligned.
        seq_length = get_max_seqlen_within_dp(seq_length)
        return min(seq_length, self.training_config.seq_length)

    @torch.no_grad()
    def compute_logprobs(
        self,
        model,
        batches_list: List[Dict[str, Any]],
        batch_log_str: str,
        compute_topk: bool = False,
        gather_target_ids_key: str = None,
        return_per_token_entropy: bool = False,
    ):
        """Run a forward-only pass and gather per-sample log-probs.

        设置 top-K 相关参数时，每个 sample 返回 dict（含 3D 字段）；否则返回 2D tensor。
        """
        # DPO expands every preference pair to [chosen, rejected] so the
        # reference forward matches policy training. DSv4 contiguous THD now
        # packs and reconstructs every sample in this microbatch.
        forward_only_mbs = (
            self.training_config.train_mbs *
            2 if isinstance(self.config, DpoConfig) else self.forward_only_mbs
        )
        assert forward_only_mbs > 0, f"{forward_only_mbs=}"
        self.batch_iters = 0
        self.total_iters = divide(len(batches_list), forward_only_mbs)
        self.batch_log_str = batch_log_str

        total_samples = len(batches_list)
        if isinstance(self.config, DpoConfig):
            seq_length = self._get_dpo_forward_seq_length(batches_list)
        else:
            seq_length = get_batches_max_seqlen(
                batches_list, self.training_config.pad_to_mulitiple_of
            )
            seq_length = get_max_seqlen_within_ep(seq_length)
        num_microbatches = divide(total_samples, forward_only_mbs)
        batch_iter = get_iterator_k_split_list(batches_list, num_microbatches)

        fwd_bwd_function = get_forward_backward_func()
        fwd_results = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(
                seq_length,
                inference_only=True,
                compute_topk=compute_topk,
                gather_target_ids_key=gather_target_ids_key,
                return_per_token_entropy=return_per_token_entropy,
            ),
            data_iterator=batch_iter,
            model=model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=forward_only_mbs,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )

        if not compute_topk and gather_target_ids_key is None and not return_per_token_entropy:
            # Label-based logps path: tensor-cat → 2D PP broadcast → per-sample chunk.
            logprobs = torch.cat(fwd_results) if len(fwd_results) > 0 else None

            # Broadcast it from last PP stage to everything else.
            logprobs = BroadcastUtils.broadcast_2d_tensor_within_pp(logprobs)
            assert logprobs.dtype == torch.float32, f'{logprobs.dtype=}'

            assert logprobs.shape[0] == total_samples
            logprobs = [logprob.squeeze(0) for logprob in logprobs.cpu().chunk(total_samples)]
            clear_memory()
            return logprobs

        # 上一个条件不通过，那么在 PP last stage 组装 per-sample dict，broadcast 到其他 stage。
        if mpu.is_pipeline_last_stage() and len(fwd_results) > 0:
            per_sample_list = []
            for mb_dict in fwd_results:
                n = mb_dict["logprobs"].shape[0]
                for i in range(n):
                    per_sample_list.append(
                        {
                            k: (v[i].float().cpu() if k == "logprobs" else v[i].cpu())
                            for k, v in mb_dict.items()
                        }
                    )
            assert len(per_sample_list
                      ) == total_samples, (f'{len(per_sample_list)=} {total_samples=}')
        else:
            per_sample_list = []

        per_sample_list = BroadcastUtils.broadcast_object_within_pp(per_sample_list)
        assert len(per_sample_list) == total_samples, \
            f"len(per_sample_list) expect {total_samples}, but get {len(per_sample_list)}"
        clear_memory()
        return per_sample_list

    def _smart_pad_forward_step(
        self,
        batch_iter,
        num_microbatches,
        micro_batch_size,
        seq_length,
        return_per_token_entropy: bool = False,
    ):
        """Wrapped forward step for smart pad helper callback.

        Parameters
        ----------
        batch_iter : iterator
            Yields micro-batches (each is a list of sample dicts).
        num_microbatches : int
        micro_batch_size : int
            Forward-only mbs.
        seq_length : int
            Padded sequence length for this group.
        return_per_token_entropy : bool
            Return per-sample dicts so smart-pad can preserve entropy alongside logprobs.

        Returns
        -------
        list
            Per-micro-batch tensors, or lists of per-sample dicts when entropy is requested.
        """
        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(
                seq_length,
                inference_only=True,
                return_per_token_entropy=return_per_token_entropy,
            ),
            data_iterator=batch_iter,
            model=self._smart_pad_current_model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )
        if return_per_token_entropy and mpu.is_pipeline_last_stage():
            per_sample_output = []
            for microbatch_output in output_tensor:
                assert isinstance(microbatch_output, dict)
                assert set(microbatch_output) == {"logprobs", "prev_per_token_entropy"}
                batch_size = microbatch_output["logprobs"].shape[0]
                assert all(value.shape[0] == batch_size for value in microbatch_output.values())
                per_sample_output.append(
                    [
                        {
                            key: value[sample_idx]
                            for key, value in microbatch_output.items()
                        } for sample_idx in range(batch_size)
                    ]
                )
            output_tensor = per_sample_output
        clear_memory()
        return output_tensor

    def _smart_pad_forward_step_logits(
        self, batch_iter, num_microbatches, micro_batch_size, seq_length
    ):
        """Like _smart_pad_forward_step but writes per-sample logits into logits_cpu_buffer."""
        assert not self.training_config.return_hidden_states_for_ce, (
            "Linear CE and compact CE are incompatible with the full-logits smart-pad path "
            "(it exists to materialize and cache the [s, V] logits)"
        )
        self._smart_pad_logits_microbatches = []
        seq_shard_global = self._smart_pad_logits_seq_length_shard_global

        # NOTE 这里又写了一遍，和logits_output_only_func 基本一致，其实本质上就是为了增加一个
        # self._smart_pad_logits_microbatches 的记录 用于对应每个 logits 和原始的 sample id 的映射关系

        def logits_output_only_step(seq_len_fixed, dataloader_iter, model):
            batches: List[Dict[str, Any]] = next(dataloader_iter)
            if self.total_iters > 0 and torch.distributed.get_rank() == 0:
                self.batch_iters += 1
                log(f"{self.batch_log_str} {self.batch_iters:8d}/{self.total_iters:8d}", rank=0)
            self._smart_pad_logits_microbatches.append(batches)  # 记录原始的 sample id 和 logits 的映射关系
            model_fwd_args = self.prepare_data.model_forward_only(
                batches,
                seq_len_fixed,
                self.tokenizer.pad_token_id,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self.vocab_size,
            )
            model_fwd_args.pop("target")
            output_tensor = model(**model_fwd_args)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            assert isinstance(output_tensor, torch.Tensor)

            def id_func(output_tensor, non_loss_data=True):
                return output_tensor

            return output_tensor, id_func

        forward_step_func = partial(logits_output_only_step, seq_length)
        fwd_bwd_function = get_forward_backward_func()
        logits_list = fwd_bwd_function(
            forward_step_func=forward_step_func,
            data_iterator=batch_iter,
            model=self._smart_pad_current_model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )

        if mpu.is_pipeline_last_stage():
            assert len(logits_list) == len(
                self._smart_pad_logits_microbatches
            ), (f"{len(logits_list)=} {len(self._smart_pad_logits_microbatches)=}")
            for logits, batches in zip(logits_list, self._smart_pad_logits_microbatches):
                assert logits.shape[0] == len(batches), (f"{logits.shape[0]=} {len(batches)=}")
                s_copy = logits.shape[1]
                for r, batch in enumerate(batches):
                    global_idx = int(batch[_sample_idx_key].item())
                    row_logits = logits[r]
                    cpu_slice = self.logits_cpu_buffer[global_idx, :s_copy, :]
                    cpu_slice.copy_(row_logits, non_blocking=True)
                    if s_copy < seq_shard_global:
                        self.logits_cpu_buffer[global_idx, s_copy:seq_shard_global, :].zero_()
                    self._smart_pad_logits_seq_shards[global_idx] = s_copy
        else:
            assert len(logits_list) == 0, f"{len(logits_list)=}"

        self._smart_pad_logits_microbatches = []
        clear_memory()
        return []

    @torch.no_grad()
    def smart_pad_compute_logprobs(
        self,
        model,
        batches_list: List[Dict[str, Any]],
        batch_log_str: str,
        return_per_token_entropy: bool = False,
    ):
        """Compute logprobs using smart pad to group samples by seqlen.

        Parameters
        ----------
        model : module
        batches_list : list of dict
            Expanded per-sample dicts.
        batch_log_str : str
            Log prefix for progress.
        return_per_token_entropy : bool
            Return ``logprobs`` and ``prev_per_token_entropy`` in each sample dict.

        Returns
        -------
        list
            Per-sample CPU logprobs tensors, or per-sample dicts containing logprobs
            and entropy.
        """
        self.batch_iters = 0
        total_samples = len(batches_list)
        self.total_iters = total_samples // self.forward_only_mbs
        self.batch_log_str = batch_log_str

        # Store model reference for the forward step callback
        self._smart_pad_current_model = model

        dynamic_mbs_target_seqlen = getattr(
            self.policy_config, 'dynamic_mbs_target_seqlen_fwd_only', None
        )
        dynamic_mbs_limit = getattr(self.policy_config, 'dynamic_mbs_limit_fwd_only', None)

        smart_pad_helper = CatedSmartPadInferHelper(batches_list, self.forward_only_mbs)
        get_seqlen_func = lambda sample: sample["tokens"].shape[-1]

        # Prepare smart pad batches first (gen_* steps)
        smart_pad_helper.gen_row_based_batches()
        smart_pad_helper.gen_extend_batches(get_seqlen_func)
        smart_pad_helper.gen_sorted_batches()
        smart_pad_helper.gen_smart_pad_batches(self.training_config.pad_to_mulitiple_of)

        # Pre-calculate actual total forward steps and update total_iters before forward
        smart_pad_helper.forward_per_seqlen_batches(
            forward_step_wrapped_func=partial(
                self._smart_pad_forward_step,
                return_per_token_entropy=return_per_token_entropy,
            ),
            dynamic_mbs_target_seqlen=dynamic_mbs_target_seqlen,
            dynamic_mbs_limit=dynamic_mbs_limit,
            update_total_iters_callback=lambda total_steps:
            setattr(self, 'total_iters', total_steps),
        )

        logprobs_list = smart_pad_helper.get_rowed_based_forward_results(is_row_based_rets=True)

        flatten_logprobs_list = []
        if mpu.is_pipeline_last_stage():
            for per_forward_step_results in logprobs_list:
                for result in per_forward_step_results:
                    if return_per_token_entropy:
                        assert isinstance(result, dict)
                        assert set(result) == {"logprobs", "prev_per_token_entropy"}
                        flatten_logprobs_list.append(
                            {
                                key: (value.float().cpu() if key == "logprobs" else value.cpu())
                                for key, value in result.items()
                            }
                        )
                    else:
                        assert isinstance(result, torch.Tensor)
                        flatten_logprobs_list.append(result.cpu())

        # Broadcast it from last PP stage to everything else.
        flatten_logprobs_list = BroadcastUtils.broadcast_object_within_pp(flatten_logprobs_list)
        assert len(
            flatten_logprobs_list
        ) == total_samples, f"len(logprobs) expect {total_samples}, but get {len(flatten_logprobs_list)}"

        self._smart_pad_current_model = None
        clear_memory()
        return flatten_logprobs_list

    @torch.no_grad()
    def smart_pad_compute_logits(
        self, model, batches_list: List[Dict[str, Any]], batch_log_str: str
    ):
        """Compute logits using smart pad to group samples by seqlen.

        Parameters
        ----------
        model : module
        batches_list : list of dict
            Expanded per-sample dicts.
        batch_log_str : str
            Log prefix for progress.

        Returns
        -------
        tuple of (None, list of Tensor)
            Per-sample logits tensors on CPU.
        """
        self.set_model_eval()
        begine_t = sync_cuda_and_get_time()
        total_samples = len(batches_list)
        self.batch_iters = 0
        self.batch_log_str = batch_log_str

        # NOTE: The smart-pad path intentionally does NOT honor `enable_data_with_alpha`
        # optmization temporarily. The optimization (skip teacher forward for the
        # alpha<=0 tail) would force teacher and student to bucket identically.
        self.total_iters = total_samples // self.forward_only_mbs

        seq_length = get_batches_max_seqlen(batches_list, self.training_config.pad_to_mulitiple_of)
        seq_length = get_max_seqlen_within_dp(seq_length)
        max_seq_length = min(seq_length, self.training_config.seq_length)

        target_logits_dtype = (
            torch.bfloat16 if self.config.training.teacher_logits_dtype == "bf16" else torch.float32
        )

        seq_length_shard = max_seq_length // mpu.get_context_parallel_world_size()
        self._smart_pad_logits_seq_length_shard_global = seq_length_shard

        if mpu.is_pipeline_last_stage():
            assert self.vocab_size % mpu.get_tensor_model_parallel_world_size(
            ) == 0, f"{self.vocab_size=} {mpu.get_tensor_model_parallel_world_size()=}"
            assert max_seq_length % mpu.get_context_parallel_world_size(
            ) == 0, f"{max_seq_length=} {mpu.get_context_parallel_world_size()=}"
            vacab_size_shard = self.vocab_size // mpu.get_tensor_model_parallel_world_size()
            # _smart_pad_logits_seq_length_shard_global 是为了记录 seq_length_shard，便于
            # _smart_pad_forward_step_logits 对不够中 seq_length_shard 的logits 进行 buffer zero_() 操作

            if self.logits_cpu_buffer is None or seq_length_shard > self.logits_cpu_buffer.shape[1]:
                self.logits_cpu_buffer = torch.empty(
                    (total_samples, seq_length_shard, vacab_size_shard),
                    dtype=target_logits_dtype,
                    device="cpu",
                    pin_memory=True,
                )

        # Store model reference for the forward step callback
        self._smart_pad_current_model = model
        # 记录每个 sample 实际的 seq shard 长度，由 _smart_pad_forward_step_logits 填充
        self._smart_pad_logits_seq_shards = {}

        dynamic_mbs_target_seqlen = getattr(
            self.policy_config, "dynamic_mbs_target_seqlen_fwd_only", None
        )
        dynamic_mbs_limit = getattr(self.policy_config, "dynamic_mbs_limit_fwd_only", None)

        smart_pad_helper = CatedSmartPadInferHelper(batches_list, self.forward_only_mbs)
        # NOTE: Cap seqlen at `max_seq_length` to keep smart-pad consistent with the non-smart-pad path (`_compute_logits`)
        get_seqlen_func = lambda sample: min(sample["tokens"].shape[-1], max_seq_length)

        # Prepare smart pad batches first (gen_* steps)
        smart_pad_helper.gen_row_based_batches()
        smart_pad_helper.gen_extend_batches(get_seqlen_func)
        smart_pad_helper.gen_sorted_batches()
        smart_pad_helper.gen_smart_pad_batches(self.training_config.pad_to_mulitiple_of)

        # Pre-calculate actual total forward steps and update total_iters before forward
        smart_pad_helper.forward_per_seqlen_batches(
            forward_step_wrapped_func=self._smart_pad_forward_step_logits,
            dynamic_mbs_target_seqlen=dynamic_mbs_target_seqlen,
            dynamic_mbs_limit=dynamic_mbs_limit,
            update_total_iters_callback=lambda total_steps:
            setattr(self, "total_iters", total_steps),
            skip_batch_id_merge=True,
        )

        logits_output = None
        if mpu.is_pipeline_last_stage():
            torch.cuda.synchronize()
            # Slice each row to its actual per-bucket shard length rather than
            # the global `seq_length_shard` so student's `seq_len_shard_by_cp`
            # matches `teacher_logits.shape[0]` exactly in `sft_train`.
            logits_output = [
                self.logits_cpu_buffer[i, :self._smart_pad_logits_seq_shards[i], :]
                for i in range(total_samples)
            ]
            assert len(logits_output) == total_samples, (
                f"Expected {total_samples} logits but got {len(logits_output)}"
            )
        else:
            logits_output = [None for _ in range(len(batches_list))]

        self._smart_pad_current_model = None
        self._smart_pad_logits_seq_shards = None
        clear_memory()

        end_t = sync_cuda_and_get_time()
        log(f"smart_pad_compute_logits using time {end_t - begine_t}", rank=0)

        return None, logits_output

    def _rl_local_response_token_count(
        self,
        batch: Dict[str, Any],
        mask: torch.Tensor,
        *,
        response_padded_dyn_cp: bool,
        cp_partition_mode: str = "zigzag",
    ) -> torch.Tensor:
        """Token count for Megatron per-token grad scale on this rank."""
        if not response_padded_dyn_cp:
            return mask.sum()
        # Rollout mask is replicated packed; only this rank's THD shard has grads.
        return dynamic_cp_local_packed_token_count(
            mask,
            batch["cu_seqlens_padded"],
            batch["local_cp_size"],
            cp_partition_mode=cp_partition_mode,
        )

    def _rl_megatron_per_rank_tokens(
        self,
        token_count: torch.Tensor,
        *,
        dyn_cp: bool,
        cp_mask_is_sharded: bool = False,
    ) -> torch.Tensor:
        """Adjust token count before DP×CP all-reduce in finalize_model_grads.

        When ``cp_size > 1`` and not dyn-CP:
        - mcore-THD (``cp_mask_is_sharded``): mask is already local → as-is.
        - HpModule ppo_pack_seq: full mask is replicated on every CP rank →
          divide by ``cp_size`` (only local shard keeps autograd after gather).
        """
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1 and not dyn_cp and not cp_mask_is_sharded:
            return token_count / cp_size
        return token_count

    def _rl_return_sum_loss_for_megatron(
        self,
        bwd_loss: torch.Tensor,
        metrics_dict: Dict[str, Any],
        *,
        mask: torch.Tensor,
        local_response_token_count: torch.Tensor,
        response_padded_dyn_cp: bool,
        dyn_cp: bool,
        cp_mask_is_sharded: bool,
    ):
        """Package sum-style ``bwd_loss`` for Megatron (new loss / GRPO seq-mean)."""
        if self.calc_per_token_loss:
            total_tokens = (local_response_token_count if response_padded_dyn_cp else mask.sum())
            per_rank_tokens = self._rl_megatron_per_rank_tokens(
                total_tokens,
                dyn_cp=dyn_cp,
                cp_mask_is_sharded=cp_mask_is_sharded,
            )
            return (
                bwd_loss,
                per_rank_tokens.clamp(min=1).to(torch.int),
                metrics_dict,
            )
        # Megatron divides by num_mbs; scale so effective loss is
        # bwd_loss / n_alive_global after DP×CP grad AVG.
        num_mbs = self._step_num_microbatches
        n_alive_global = max(self._step_effective_global_batch_size, 1)
        dp_cp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        scaled_loss = bwd_loss * num_mbs / n_alive_global * dp_cp_size
        return (
            scaled_loss,
            torch.tensor(1, dtype=torch.int, device=mask.device),
            metrics_dict,
        )

    def _rl_return_mean_loss_for_megatron(
        self,
        bwd_loss: torch.Tensor,
        metrics_dict: Dict[str, Any],
        *,
        mask: torch.Tensor,
        local_response_token_count: torch.Tensor,
        reconstructed_response_token_count: torch.Tensor,
        response_padded_dyn_cp: bool,
        dyn_cp: bool,
        cp_mask_is_sharded: bool,
    ):
        """Legacy micro-batch-mean losses: convert mean → sum for per-token path."""
        if not self.calc_per_token_loss:
            return (bwd_loss, metrics_dict)
        total_tokens = (local_response_token_count if response_padded_dyn_cp else mask.sum())
        loss_sum = bwd_loss * reconstructed_response_token_count
        per_rank_tokens = self._rl_megatron_per_rank_tokens(
            total_tokens,
            dyn_cp=dyn_cp,
            cp_mask_is_sharded=cp_mask_is_sharded,
        )
        return (loss_sum, per_rank_tokens.to(torch.int), metrics_dict)

    def _rl_compute_curr_policy_outputs(
        self,
        *,
        model_output: torch.Tensor,
        linear_ce_output: Optional[Dict[str, Any]],
        use_linear_ce: bool,
        target: torch.Tensor,
        mask: torch.Tensor,
        local_output_mask: Optional[torch.Tensor],
        ignore_cp: bool,
        pre_shifted: bool,
        response_padded_dyn_cp: bool,
        prev_topk_logprobs: Optional[torch.Tensor],
        opd_topk_ids: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Curr logprobs / entropy / optional top-k from logits, Linear CE, or compact CE."""
        curr_topk_logprobs = None
        dumped_topk_logprobs = None
        dumped_topk_token_ids = None
        logprob_temperature = self.get_logprob_temperature()

        if use_linear_ce:
            curr_log_probs, scaled_entropy, per_token_entropy = logprobs_from_linear_ce(
                linear_ce_backend=self.training_config.linear_ce_backend,
                linear_ce_output=linear_ce_output,
                target=target,
                mask=local_output_mask,
                pre_shifted=pre_shifted,
                ignore_cp=ignore_cp,
                return_entropy=True,
                temperature=logprob_temperature,
                token_compaction=self.training_config.ce_compaction,
            )
            im_end_metrics = {}
            if prev_topk_logprobs is not None and opd_topk_ids is not None:
                prev_topk_logprobs = prev_topk_logprobs.float()
                target_topk_ids = torch.nn.functional.pad(
                    opd_topk_ids.long(), (0, 0, 0, 1), value=0
                )
                curr_topk_logprobs = opd_topk_logprobs_from_linear_ce(
                    linear_ce_backend=self.training_config.linear_ce_backend,
                    linear_ce_output=linear_ce_output,
                    target_ids=target_topk_ids,
                    ignore_cp=ignore_cp,
                    temperature=logprob_temperature,
                )[:, :-1].contiguous()
        elif self.training_config.ce_compaction:
            curr_log_probs, scaled_entropy, per_token_entropy = logprobs_from_compact_ce(
                linear_ce_output=linear_ce_output,
                target=target,
                mask=local_output_mask,
                pre_shifted=pre_shifted,
                ignore_cp=ignore_cp,
                return_entropy=True,
                temperature=logprob_temperature,
            )
            im_end_metrics = {}
        else:
            parallel_logits = model_output.float()
            parallel_logits_clone = parallel_logits.clone()
            curr_log_probs = from_parallel_logits_to_logprobs(
                vocab_parallel_logits=parallel_logits,
                target=target,
                ignore_cp=ignore_cp,
                pre_shifted=pre_shifted,
                temperature=logprob_temperature,
            )
            if prev_topk_logprobs is not None and opd_topk_ids is not None:
                prev_topk_logprobs = prev_topk_logprobs.float()
                target_topk_ids = torch.nn.functional.pad(
                    opd_topk_ids.long(), (0, 0, 0, 1), value=0
                )
                curr_topk_logprobs = from_parallel_logits_to_opd_topk_logprobs(
                    vocab_parallel_logits=parallel_logits_clone,
                    target_ids=target_topk_ids,
                    ignore_cp=ignore_cp,
                    temperature=logprob_temperature,
                )[:, :-1].contiguous()

            im_end_metrics = (
                {} if response_padded_dyn_cp else self.get_im_end_metrics(
                    parallel_logits=parallel_logits_clone,
                    target=target,
                    response_mask=mask,
                    ignore_cp=ignore_cp,
                )
            )
            scaled_entropy, per_token_entropy = vocab_parallel_entropy(
                parallel_logits_clone,
                local_output_mask,
                ignore_cp=ignore_cp,
                pre_shifted=pre_shifted,
            )
            if (
                not response_padded_dyn_cp and self.should_dump_metrics and
                self.config.training.dump_metrics_logprobs_topk > 0
            ):
                dumped_topk_logprobs, dumped_topk_token_ids = (
                    from_parallel_logits_to_topk_logprobs(
                        vocab_parallel_logits=parallel_logits_clone,
                        topk=self.config.training.dump_metrics_logprobs_topk,
                        temperature=logprob_temperature,
                    )
                )
                dumped_topk_logprobs = dumped_topk_logprobs.to(dtype=torch.bfloat16, device="cpu")
                dumped_topk_token_ids = dumped_topk_token_ids.to(dtype=torch.int32, device="cpu")

        return {
            "curr_log_probs": curr_log_probs,
            "scaled_entropy": scaled_entropy,
            "per_token_entropy": per_token_entropy,
            "prev_topk_logprobs": prev_topk_logprobs,
            "curr_topk_logprobs": curr_topk_logprobs,
            "im_end_metrics": im_end_metrics,
            "dumped_topk_logprobs": dumped_topk_logprobs,
            "dumped_topk_token_ids": dumped_topk_token_ids,
        }

    def _rl_response_pad_dyn_cp_tensors(
        self,
        batch: Dict[str, Any],
        *,
        curr_log_probs: torch.Tensor,
        per_token_entropy: Optional[torch.Tensor],
        curr_topk_logprobs: Optional[torch.Tensor],
        mask: torch.Tensor,
        advantages: torch.Tensor,
        prev_log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor],
        rollout_log_probs: Optional[torch.Tensor],
        teacher_log_probs: Optional[torch.Tensor],
        prev_per_token_entropy: Optional[torch.Tensor],
        prev_topk_logprobs: Optional[torch.Tensor],
        scaled_entropy: torch.Tensor,
        cp_partition_mode: str = "zigzag",
    ) -> Dict[str, Any]:
        """Gather CP-sharded model outs and response-pad to ``[B, max_resp]``."""
        local_cp_size = batch["local_cp_size"]
        cu_seqlens_padded = batch["cu_seqlens_padded"]
        resp_start = batch["dyn_cp_response_start"]
        resp_length = batch["dyn_cp_response_length"]

        # Rollout tensors stay replicated in the DCP group; only model outputs
        # were THD-sharded and need a gradient-preserving CP gather.
        packed_model = {
            "curr_log_probs": curr_log_probs,
            "per_token_entropy": per_token_entropy,
            "curr_topk_logprobs": curr_topk_logprobs,
        }
        reconstructed = {
            key:
                (
                    reconstruct_dynamic_cp_packed_tensor(
                        value,
                        cu_seqlens_padded,
                        local_cp_size,
                        cp_partition_mode=cp_partition_mode,
                    ) if value is not None else None
                )
            for key, value in packed_model.items()
        }
        jagged = {
            key:
                (
                    packed_to_jagged(value, cu_seqlens_padded, batch["cu_seqlens"])
                    if value is not None else None
                )
            for key, value in reconstructed.items()
        }
        model_resp = jagged_to_response_padded(jagged, resp_start, resp_length)
        rollout_resp = packed_to_response_padded(
            {
                "mask": mask,
                "advantages": advantages,
                "prev_log_probs": prev_log_probs,
                "ref_log_probs": ref_log_probs,
                "rollout_log_probs": rollout_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "prev_per_token_entropy": prev_per_token_entropy,
                "prev_topk_logprobs": prev_topk_logprobs,
                "sample_mask": batch.get("sample_mask", None),
                "token_weights": batch.get("token_weights", None),
            },
            cu_seqlens_padded,
            resp_start,
            resp_length,
        )
        mask = rollout_resp["mask"]
        per_token_entropy = model_resp["per_token_entropy"]
        scaled_entropy = (
            (per_token_entropy * mask).sum() /
            mask.sum().clamp(min=1) if per_token_entropy is not None else scaled_entropy
        )
        return {
            "mask": mask,
            "advantages": rollout_resp["advantages"],
            "prev_log_probs": rollout_resp["prev_log_probs"],
            "ref_log_probs": rollout_resp["ref_log_probs"],
            "rollout_log_probs": rollout_resp["rollout_log_probs"],
            "teacher_log_probs": rollout_resp["teacher_log_probs"],
            "curr_log_probs": model_resp["curr_log_probs"],
            "per_token_entropy": per_token_entropy,
            "prev_per_token_entropy": rollout_resp["prev_per_token_entropy"],
            "prev_topk_logprobs": rollout_resp["prev_topk_logprobs"],
            "curr_topk_logprobs": model_resp["curr_topk_logprobs"],
            "sample_mask": as_sequence_sample_mask(rollout_resp["sample_mask"]),
            "token_weights": rollout_resp["token_weights"],
            "reconstructed_response_token_count": mask.sum(),
            "scaled_entropy": scaled_entropy,
        }

    def _rl_policy_loss_common_kwargs(
        self,
        *,
        advantages: torch.Tensor,
        prev_log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor],
        curr_log_probs: torch.Tensor,
        mask: torch.Tensor,
        scaled_entropy: torch.Tensor,
        rollout_log_probs: Optional[torch.Tensor],
        per_token_entropy: Optional[torch.Tensor],
        prev_per_token_entropy: Optional[torch.Tensor],
        model_output: torch.Tensor,
        response_padded_dyn_cp: bool,
        return_hidden_states_for_ce: bool,
        sample_mask: Optional[torch.Tensor],
        batch: Dict[str, Any],
        teacher_log_probs: Optional[torch.Tensor],
        dumped_topk_logprobs: Optional[torch.Tensor],
        dumped_topk_token_ids: Optional[torch.Tensor],
        prev_topk_logprobs: Optional[torch.Tensor],
        curr_topk_logprobs: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        return {
            "advantages": advantages,
            "prev_log_probs": prev_log_probs,
            "ref_log_probs": ref_log_probs,
            "curr_log_probs": curr_log_probs,
            "response_mask": mask,
            "scaled_entropy": scaled_entropy,
            "rollout_log_probs": rollout_log_probs,
            "per_token_entropy": per_token_entropy,
            "prev_per_token_entropy": prev_per_token_entropy,
            "parallel_logits": (None if response_padded_dyn_cp or return_hidden_states_for_ce else model_output),
            "sample_mask": sample_mask,
            "global_retention_ratio": batch.get("global_retention_ratio", None),
            "entropy_aux_figures": batch.get("entropy_aux_figures", None),
            "teacher_log_probs": teacher_log_probs,
            "dumped_topk_logprobs": dumped_topk_logprobs,
            "dumped_topk_token_ids": dumped_topk_token_ids,
            "should_dump_metrics": self.should_dump_metrics and not return_hidden_states_for_ce,
            "prev_topk_logprobs": prev_topk_logprobs,
            "curr_topk_logprobs": curr_topk_logprobs,
            "calculate_per_token_loss": self.calc_per_token_loss,
        }

    def rl_forward_step(self, seq_length: int):
        def fwd_output_and_loss_func(seq_length, data_iterator, model):
            dyn_cp = self.dist_config.dynamic_context_parallel
            if isinstance(self.config, OnPolicyDistillConfig):
                if dyn_cp:
                    prepare_data_func = self.prepare_data.opd_train_with_dynamic_cp
                else:
                    prepare_data_func = self.prepare_data.opd_train
            elif dyn_cp:
                prepare_data_func = self.prepare_data.grpo_train_with_dynamic_cp
            else:
                prepare_data_func = self.prepare_data.grpo_train

            # for r3
            with self.router_replay_step_ctx():
                batches: List[Dict[str, Any]] = next(data_iterator)
                # log for rl_train microbatch progress
                self.batch_iters += 1
                if self.total_iters > 0 and torch.distributed.get_rank() == 0:
                    log(f"{self.batch_log_str} {self.batch_iters:8d}/{self.total_iters:8d}", rank=0)
                unwrapped_model = unwrap_model(model)
                # for r3
                if not dyn_cp:
                    self.maybe_prepare_for_router_replay(batches, seq_length)

                batch, fwd_kwargs = prepare_data_func(
                    batches,
                    seq_length,
                    self.tokenizer.pad_token_id,
                    ppo_pack_seq=self.policy_config.ppo_pack_seq,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self.vocab_size,
                )
                if dyn_cp:
                    self.maybe_prepare_for_router_replay(
                        [batch],
                        batch["tokens"].shape[-1],
                        packed_thd=True,
                    )

                for key in ["mask", "advantages", "prev_log_probs", "target"]:
                    assert key in batch

                # Linear CE and compact CE both make the model return hidden+weight (return_hidden_states_for_ce).
                # use_linear_ce only picks fused vs ordinary CE.
                use_linear_ce = self.training_config.use_linear_ce
                return_hidden_states_for_ce = self.training_config.return_hidden_states_for_ce
                fp32_output = not return_hidden_states_for_ce
                linear_ce_output = None
                if self.policy_config.ppo_pack_seq and not dyn_cp:
                    assert not return_hidden_states_for_ce, "暂不支持"
                    model_output = gptmodel_pack_foward(
                        unwrapped_model, batch, fwd_kwargs, self.config
                    )
                else:
                    model_output = model(**fwd_kwargs, fp32_output=fp32_output)

                if isinstance(model_output, tuple):
                    model_output = model_output[0]
                if return_hidden_states_for_ce and mpu.is_pipeline_last_stage():
                    assert isinstance(model_output, dict)
                    assert model_output["hidden_states"].dtype == torch.bfloat16
                    linear_ce_output = model_output
                    model_output = model_output["hidden_states"]
                else:
                    assert isinstance(model_output, torch.Tensor)

                def loss_func(model_output):
                    mask = batch["mask"]
                    # Response-pad when prepare attached ``dyn_cp_response_*``.
                    response_padded_dyn_cp = dyn_cp and "dyn_cp_response_start" in batch
                    local_output_mask = None if response_padded_dyn_cp else mask
                    cp_partition_mode = _cp_partition_mode_from_fwd_kwargs(fwd_kwargs)
                    local_response_token_count = self._rl_local_response_token_count(
                        batch,
                        mask,
                        response_padded_dyn_cp=response_padded_dyn_cp,
                        cp_partition_mode=cp_partition_mode,
                    )

                    advantages = batch["advantages"].float()
                    prev_log_probs = batch["prev_log_probs"]
                    assert prev_log_probs.dtype == torch.float32
                    prev_per_token_entropy = batch.get("prev_per_token_entropy", None)
                    ref_log_probs = batch.get("ref_log_probs", None)
                    teacher_log_probs = batch.get("teacher_log_probs", None)
                    rollout_log_probs = batch.get("rollout_log_probs", None)
                    prev_topk_logprobs = batch.get("prev_topk_logprobs", None)
                    opd_topk_ids = batch.get("opd_topk_ids", None)
                    target = batch["target"]
                    if (response_padded_dyn_cp and self.training_config.ce_compaction):
                        # batch["mask"] is packed; slice it the same way as local target.
                        local_output_mask = build_grpo_compact_ce_mask_dyn_cp(
                            batch,
                            target,
                            cp_partition_mode=cp_partition_mode,
                        )

                    # TODO: merge ppo_pack_seq into dyn_cp when practical
                    ignore_cp = self.policy_config.ppo_pack_seq or dyn_cp
                    # ``full_packed_seq_params`` → mcore THD: mask/target already
                    # CP-sharded + next-token shifted (like dyn_cp).
                    cp_mask_is_sharded = batch.get("full_packed_seq_params") is not None
                    pre_shifted = dyn_cp or cp_mask_is_sharded

                    curr = self._rl_compute_curr_policy_outputs(
                        model_output=model_output,
                        linear_ce_output=linear_ce_output,
                        use_linear_ce=use_linear_ce,
                        target=target,
                        mask=mask,
                        local_output_mask=local_output_mask,
                        ignore_cp=ignore_cp,
                        pre_shifted=pre_shifted,
                        response_padded_dyn_cp=response_padded_dyn_cp,
                        prev_topk_logprobs=prev_topk_logprobs,
                        opd_topk_ids=opd_topk_ids,
                    )
                    curr_log_probs = curr["curr_log_probs"]
                    scaled_entropy = curr["scaled_entropy"]
                    per_token_entropy = curr["per_token_entropy"]
                    prev_topk_logprobs = curr["prev_topk_logprobs"]
                    curr_topk_logprobs = curr["curr_topk_logprobs"]
                    im_end_metrics = curr["im_end_metrics"]
                    dumped_topk_logprobs = curr["dumped_topk_logprobs"]
                    dumped_topk_token_ids = curr["dumped_topk_token_ids"]

                    if response_padded_dyn_cp:
                        padded = self._rl_response_pad_dyn_cp_tensors(
                            batch,
                            curr_log_probs=curr_log_probs,
                            per_token_entropy=per_token_entropy,
                            curr_topk_logprobs=curr_topk_logprobs,
                            mask=mask,
                            advantages=advantages,
                            prev_log_probs=prev_log_probs,
                            ref_log_probs=ref_log_probs,
                            rollout_log_probs=rollout_log_probs,
                            teacher_log_probs=teacher_log_probs,
                            prev_per_token_entropy=prev_per_token_entropy,
                            prev_topk_logprobs=prev_topk_logprobs,
                            scaled_entropy=scaled_entropy,
                            cp_partition_mode=cp_partition_mode,
                        )
                        mask = padded["mask"]
                        advantages = padded["advantages"]
                        prev_log_probs = padded["prev_log_probs"]
                        ref_log_probs = padded["ref_log_probs"]
                        rollout_log_probs = padded["rollout_log_probs"]
                        teacher_log_probs = padded["teacher_log_probs"]
                        curr_log_probs = padded["curr_log_probs"]
                        per_token_entropy = padded["per_token_entropy"]
                        prev_per_token_entropy = padded["prev_per_token_entropy"]
                        prev_topk_logprobs = padded["prev_topk_logprobs"]
                        curr_topk_logprobs = padded["curr_topk_logprobs"]
                        response_sample_mask = padded["sample_mask"]
                        response_token_weights = padded["token_weights"]
                        reconstructed_response_token_count = padded[
                            "reconstructed_response_token_count"]
                        scaled_entropy = padded["scaled_entropy"]
                    else:
                        response_sample_mask = batch.get("sample_mask", None)
                        response_token_weights = batch.get("token_weights", None)
                        reconstructed_response_token_count = local_response_token_count

                    common_kwargs = self._rl_policy_loss_common_kwargs(
                        advantages=advantages,
                        prev_log_probs=prev_log_probs,
                        ref_log_probs=ref_log_probs,
                        curr_log_probs=curr_log_probs,
                        mask=mask,
                        scaled_entropy=scaled_entropy,
                        rollout_log_probs=rollout_log_probs,
                        per_token_entropy=per_token_entropy,
                        prev_per_token_entropy=prev_per_token_entropy,
                        model_output=model_output,
                        response_padded_dyn_cp=response_padded_dyn_cp,
                        return_hidden_states_for_ce=return_hidden_states_for_ce,
                        sample_mask=response_sample_mask,
                        batch=batch,
                        teacher_log_probs=teacher_log_probs,
                        dumped_topk_logprobs=dumped_topk_logprobs,
                        dumped_topk_token_ids=dumped_topk_token_ids,
                        prev_topk_logprobs=prev_topk_logprobs,
                        curr_topk_logprobs=curr_topk_logprobs,
                    )
                    return_kw = dict(
                        mask=mask,
                        local_response_token_count=local_response_token_count,
                        response_padded_dyn_cp=response_padded_dyn_cp,
                        dyn_cp=dyn_cp,
                        cp_mask_is_sharded=cp_mask_is_sharded,
                    )

                    if not self.ppo_config.use_legacy_loss:
                        if dyn_cp:
                            assert response_padded_dyn_cp, (
                                "new loss + dyn-CP requires response-padded [B, S] "
                                "tensors before loss (missing dyn_cp_response_start)"
                            )
                        loss_input = PolicyLossInputV2(
                            **common_kwargs,
                            token_weights=response_token_weights,
                        )
                        policy_loss_fn = get_loss_fn("mcore", self.ppo_config.loss_func)
                        bwd_loss, _bwd_count, metrics_dict = policy_loss_fn(self.config, loss_input)
                        metrics_dict.update(im_end_metrics)
                        return self._rl_return_sum_loss_for_megatron(
                            bwd_loss, metrics_dict, **return_kw
                        )

                    loss_input = PolicyLossInput(
                        **common_kwargs,
                        cu_seqlens_padded=(
                            None
                            if response_padded_dyn_cp else batch.get("cu_seqlens_padded", None)
                        ),
                        local_cp_size=(
                            1 if response_padded_dyn_cp else batch.get("local_cp_size", 1)
                        ),
                    )
                    policy_loss_fn = get_policy_loss_fn(self.ppo_config.loss_func)
                    bwd_loss, metrics_dict = policy_loss_fn(self.config, loss_input)
                    metrics_dict.update(im_end_metrics)

                    if not is_seq_mean_rl_loss_fn(policy_loss_fn):
                        # Legacy micro-batch-mean: bwd_loss is already a mean.
                        return self._rl_return_mean_loss_for_megatron(
                            bwd_loss,
                            metrics_dict,
                            reconstructed_response_token_count=(reconstructed_response_token_count),
                            **return_kw,
                        )
                    # Seq-mean RL (e.g. GRPO): bwd_loss is already the GBS numerator.
                    return self._rl_return_sum_loss_for_megatron(
                        bwd_loss, metrics_dict, **return_kw
                    )

                return model_output, loss_func

        return partial(fwd_output_and_loss_func, seq_length)

    def _extract_dump_metrics_micro_batch(self, metrics_micro_batch):
        """Collect ``dump/*`` fields from each micro-batch metrics dict."""
        if not self.should_dump_metrics or not metrics_micro_batch:
            return []
        return [
            {
                k: v
                for k, v in sample.items() if k.startswith("dump/")
            } for sample in metrics_micro_batch
        ]

    def _broadcast_dumped_metrics(
        self,
        metrics,
        dump_metrics_micro_batch,
        packed_batches=None,
        routing_info=None,
    ):
        """Broadcast per-sample dump metrics from the last PP stage; strip ``dump/`` prefix.
        dyn-CP(with routing_info) reverse-reroutes first."""
        if not self.should_dump_metrics:
            return
        if routing_info is not None:
            dumped_loss_fn_metrics = self._reverse_dyn_cp_dumped_metrics(
                dump_metrics_micro_batch,
                packed_batches,
                routing_info,
            )
        elif is_pipeline_last_stage():
            dumped_loss_fn_metrics = []
            for sample in dump_metrics_micro_batch:
                batch_size = next(
                    (v.shape[0] for v in sample.values() if v is not None and torch.is_tensor(v)),
                    None,
                )
                if batch_size is None:
                    continue
                for idx in range(batch_size):
                    dumped_loss_fn_metrics.append(
                        {
                            k[len("dump/"):]: v[idx].clone()
                            for k, v in sample.items() if v is not None
                        }
                    )
        else:
            dumped_loss_fn_metrics = []
        obj = [dumped_loss_fn_metrics]
        torch.distributed.broadcast_object_list(
            obj,
            src=get_pipeline_model_parallel_last_rank(),
            group=get_pipeline_model_parallel_group(),
        )
        metrics["dumped_loss_fn_metrics"] = obj[0]

    def _reverse_dyn_cp_dumped_metrics(
        self,
        dump_metrics_micro_batch,
        packed_batches,
        routing_info,
    ):
        """Last PP: scatter dump to ``[T-1]`` and reverse-reroute to original DP order."""
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        if not is_pipeline_last_stage():
            return []
        assert packed_batches is not None, "dyn-CP dump reverse requires packed_batches"
        per_gid = build_per_gid_fullseq_dump(dump_metrics_micro_batch, packed_batches)
        if per_gid:
            local_keys = sorted(set.intersection(*(set(f) for f in per_gid.values())))
            sample = next(iter(per_gid.values()))
            local_dtypes = {k: sample[k].dtype for k in local_keys}
        else:
            local_keys, local_dtypes = [], {}
        gathered = [None] * dp_cp_group.size()
        dist.all_gather_object(gathered, (local_keys, local_dtypes), group=dp_cp_group)
        dump_keys, dtypes = next(((k, d) for k, d in gathered if k), ([], {}))
        assert dump_keys, "dyn-CP dump produced no 1D per-token fields"
        assert all(not keys or keys == dump_keys
                   for keys, _ in gathered), ("dyn-CP dump keys disagree across DP×CP ranks")
        return reverse_and_collect_dump_metrics(
            per_gid,
            routing_info,
            dp_cp_group,
            dump_keys,
            dtypes,
        )

    def collate_microbatch_metrics(
        self,
        metric_prefix: str,
        metrics_micro_batch: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collate per-microbatch metrics into ``{prefix/key: list}`` for DP reduce.

        Drops ``dump/`` keys and moves CUDA tensors to CPU. Nested lists are
        extended in place so each value is ready for
        ``reduce_metrics_across_data_parallel_group``.
        """
        def _to_cpu(val: Any) -> Any:
            if isinstance(val, torch.Tensor) and val.is_cuda:
                return val.detach().cpu()
            if isinstance(val, list):
                return [_to_cpu(v) for v in val]
            return val

        metrics: Dict[str, List[Any]] = {}
        if len(metrics_micro_batch) == 0:
            return metrics

        for key in metrics_micro_batch[0].keys():
            # 过滤 "dump/" 开头的 key
            if key.startswith("dump/"):
                continue
            for metric in metrics_micro_batch:
                assert key in metric, f"key {key} not in metric {metric}"
                val = _to_cpu(metric[key])
                if not isinstance(val, list):
                    val = [val]
                metrics.setdefault(f"{metric_prefix}/{key}", []).extend(val)

        return metrics

    def _compute_step_gbs_and_token_cnt(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[int, int, torch.Tensor]:
        """All-reduce gbs / effective_gbs / token_cnt for this train step.

        Must be called on the **pre-reroute** per-sample batch (before dyn-CP
        packing). CP siblings share the same sample list under static CP, so
        counts are reduced over the DP group only.

        ``sample_mask`` is all-or-nothing: if any sample has it, every sample
        must have it. A sample is dead when its mask is 0/False; otherwise
        alive. When the key is absent, ``effective_gbs == gbs``.

        ``gbs`` is the actual post-conversion sample count and may differ from
        the nominal trajectory count in ``training.train_gbs``.
        """
        device = torch.cuda.current_device()
        local_stats = torch.zeros(3, device=device, dtype=torch.float32)

        local_stats[0] = float(len(batch))
        if "sample_mask" in batch[0]:
            local_stats[1] = torch.stack([b["sample_mask"] for b in batch]).float().sum()
        else:
            local_stats[1] = float(len(batch))
        for b in batch:
            local_stats[2] += b["mask"].float().sum()
        dist.all_reduce(local_stats, group=mpu.get_data_parallel_group())

        gbs = int(local_stats[0].item())
        effective_gbs = int(local_stats[1].item())
        token_cnt = local_stats[2]
        return gbs, effective_gbs, token_cnt

    def _compute_pretrain_packed_batch_stats(
        self,
        microbatches: List[Dict[str, Any]],
    ) -> Tuple[int, int, int, int]:
        """Local packed-THD stats (sum over microbatches).

        Callers log under ``*_sum`` keys and DP-reduce via
        ``reduce_metrics_across_data_parallel_group_gloo``.
        """
        num_samples = 0
        num_tokens = 0
        num_pad_tokens = 0
        num_label_tokens = 0
        for mb in microbatches:
            assert "cu_seqlens_padded" in mb, "packed mb missing cu_seqlens_padded"
            assert "cu_seqlens" in mb, "packed mb missing cu_seqlens"
            assert "loss_mask" in mb, "packed mb missing loss_mask"
            num_samples += int(mb["cu_seqlens_padded"].shape[0] - 1)
            num_tokens += int(mb["cu_seqlens"][-1].item())
            num_pad_tokens += int(mb["cu_seqlens_padded"][-1].item())
            num_label_tokens += int(mb["loss_mask"].float().sum().item())

        num_pad_tokens -= num_tokens
        assert num_pad_tokens >= 0, "num_pad_tokens < 0"
        return num_samples, num_tokens, num_label_tokens, num_pad_tokens

    def _update_policy(
        self,
        batch: List[Dict[str, Any]],
        num_microbatches: int,
        routing_info=None,
    ):
        policy_config = self.policy_config
        dyn_cp = self.dist_config.dynamic_context_parallel
        batch_size = len(batch)
        dynamic_num_microbatches = 0
        dynamic_mbs = 0
        dyn_cp_max_local_cp = None

        if dyn_cp:
            if self.should_dump_metrics:
                assert routing_info is not None, (
                    "dyn-CP dump requires routing_info from rl_reroute_data_for_dynamic_cp"
                )
                assert not (getattr(self.training_config, "ppo_dump_moe_topk", 0) or 0
                           ), ("ppo_dump_moe_topk is not supported with dynamic_context_parallel")
            assert self.config.policy.model_arch != "deepseek_v3", (
                "Dynamic CP does not yet support MTP-using architectures (deepseek_v3)"
            )
            # batch is already rerouted into packed THD microbatches upstream.
            max_seqlens = []
            local_cp_sizes = []
            for b in batch:
                # Use actual max sub-seqlen from packed microbatches.
                ms = b.get('max_seqlen', None)
                if ms is not None:
                    max_seqlens.append(ms.item() if torch.is_tensor(ms) else int(ms))
                lcp = b.get('local_cp_size', None)
                if lcp is not None:
                    local_cp_sizes.append(lcp.item() if torch.is_tensor(lcp) else int(lcp))
            seq_length = max(
                max_seqlens
            ) if max_seqlens else self.dist_config.max_seqlen_per_dp_cp_rank
            dyn_cp_max_local_cp = max(local_cp_sizes) if local_cp_sizes else 1
            # Reroute leaves per-rank packs different; report / use DP×CP global max.
            stats = torch.tensor(
                [seq_length, dyn_cp_max_local_cp],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(
                stats,
                op=torch.distributed.ReduceOp.MAX,
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            seq_length = int(stats[0].item())
            dyn_cp_max_local_cp = int(stats[1].item())
            actual_num_microbatches = num_microbatches
            micro_batch_size = 1
        else:
            seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
            if policy_config.dynamic_mbs_target_seqlen is not None:
                seq_length = get_max_seqlen_within_dp(seq_length)
            else:
                seq_length = get_max_seqlen_within_ep(seq_length)

            if policy_config.dynamic_mbs_target_seqlen is not None:
                dynamic_mbs = policy_config.dynamic_mbs_target_seqlen // seq_length * self.training_config.train_mbs
                if dynamic_mbs == 0:
                    dynamic_mbs = 1
                dynamic_mbs = min(policy_config.dynamic_mbs_limit, dynamic_mbs)
                while dynamic_mbs >= 1:
                    if batch_size % dynamic_mbs == 0:
                        dynamic_num_microbatches = batch_size // dynamic_mbs
                        break
                    else:
                        dynamic_mbs -= 1
                if dynamic_num_microbatches > 0:
                    assert dynamic_num_microbatches * dynamic_mbs == batch_size, \
                        f"{dynamic_mbs=} {dynamic_num_microbatches=} {batch_size=} mismatch!"

            actual_num_microbatches = (
                dynamic_num_microbatches if dynamic_num_microbatches > 0 else num_microbatches
            )
            micro_batch_size = (
                dynamic_mbs if dynamic_num_microbatches > 0 else self.training_config.train_mbs
            )

        enable_dynamic_mbs = dynamic_num_microbatches > 0
        log(
            f"[TRAIN] {seq_length=} {batch_size=} {num_microbatches=} {enable_dynamic_mbs=} {dynamic_num_microbatches=} {dynamic_mbs=}",
            rank=0
        )
        self.batch_iters = 0
        self.total_iters = actual_num_microbatches
        rl_train_log_suffix = " (dyn_cp)" if dyn_cp else ""
        self.batch_log_str = f"rl_train{rl_train_log_suffix} microbatch "
        self._step_num_microbatches = actual_num_microbatches

        data_iter = get_iterator_k_split_list(batch, actual_num_microbatches)
        fwd_bwd_function = get_forward_backward_func()

        metrics_micro_batch = fwd_bwd_function(
            forward_step_func=self.rl_forward_step(seq_length),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=actual_num_microbatches,
            forward_only=False,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )
        dump_metrics_micro_batch = self._extract_dump_metrics_micro_batch(metrics_micro_batch)

        metrics = {}
        if is_pipeline_last_stage() and len(metrics_micro_batch) > 0:
            token_level_accumulated = {}
            scalar_accumulated = {}
            histogram_accumulated = {}

            # Dyn-CP collaborators each compute the same full reconstructed
            # loss/metrics for a local_cp_size>1 microbatch. Dividing by
            # local_cp_size before the DP×CP all_reduce keeps each sample
            # counted once (otherwise long CP-expanded MBs dominate).
            # Dyn-CP: batch items are already packed MBs (1:1 with metrics).
            # Non-dyn-CP: batch is samples; metrics are per microbatch.
            if dyn_cp:
                assert len(metrics_micro_batch) == len(batch), (
                    f"metrics_micro_batch={len(metrics_micro_batch)} != batch={len(batch)}"
                )
                mb_cp_sizes = []
                for mb in batch:
                    lcp = mb.get("local_cp_size", 1)
                    mb_cp_sizes.append(float(lcp.item() if torch.is_tensor(lcp) else lcp))
                mb_cp_scale = torch.tensor(
                    mb_cp_sizes,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                )
            else:
                mb_cp_scale = torch.ones(
                    len(metrics_micro_batch),
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                )

            for key in metrics_micro_batch[0].keys():
                if key.startswith("dump/"):
                    continue
                # values are
                # [
                #     [sum, count], # micro batch 1
                #     [sum, count], # micro batch 2
                #     ...
                #     [sum, count], # micro batch n
                # ] for metrics like loss, ppo_ratio, etc.
                # [
                #     scalar, # micro batch 1
                #     scalar, # micro batch 2
                #     ...
                #     scalar, # micro batch n
                # ] for scalar metrics
                values = torch.stack([loss_reduced[key] for loss_reduced in metrics_micro_batch])
                if values.dim() == 2 and values.shape[1] == 2:
                    # [[sum, count], ...] metrics
                    values = values / mb_cp_scale.unsqueeze(1).to(device=values.device)
                    token_level_accumulated[key] = values.sum(dim=0)
                elif key.endswith("_histogram"):
                    # histogram vectors are already counts; scale per-MB.
                    values = values / mb_cp_scale.unsqueeze(1).to(device=values.device)
                    histogram_accumulated[key] = values.sum(dim=0)
                else:
                    # [scalar, ...] metrics
                    if key.endswith("_min"):
                        scalar_accumulated[key] = values.min()
                    elif key.endswith("_max"):
                        scalar_accumulated[key] = values.max()
                    else:
                        scalar_accumulated[key] = values.mean()

            if token_level_accumulated:
                tk_keys = sorted(token_level_accumulated.keys())
                all_vals = torch.stack([token_level_accumulated[k] for k in tk_keys])
                torch.distributed.all_reduce(
                    all_vals, group=mpu.get_data_parallel_group(with_context_parallel=dyn_cp)
                )
                for i, k in enumerate(tk_keys):
                    token_level_accumulated[k] = all_vals[i]

            for key in sorted(histogram_accumulated):
                torch.distributed.all_reduce(
                    histogram_accumulated[key],
                    group=mpu.get_data_parallel_group(with_context_parallel=dyn_cp),
                )

            histogram_metrics = finalize_histogram_metrics(
                self.ppo_config.loss_func,
                self.config,
                histogram_accumulated,
            )

            if is_last_rank():
                metrics = {"policy/seq_length": seq_length}
                if dyn_cp_max_local_cp is not None:
                    metrics["policy/dyn_cp_local_cp_max"] = dyn_cp_max_local_cp
                for key, val in token_level_accumulated.items():
                    metric_key = key if key.startswith("eos/") else f"policy/{key}"
                    metrics[metric_key] = (val[0] / val[1].clamp(min=1)).cpu().item()
                for key, val in scalar_accumulated.items():
                    metric_key = key if key.startswith("eos/") else f"policy/{key}"
                    metrics[metric_key] = val.cpu().item()
                for key, val in histogram_metrics.items():
                    metrics[f"policy/{key}"] = val.cpu().item()

        aux_metrics = self._collect_aux_metrics(actual_num_microbatches)
        metrics.update(aux_metrics)
        obj_list = [metrics]
        torch.distributed.broadcast_object_list(
            obj_list, get_last_rank(cpu_group()), group=cpu_group()
        )
        metrics = obj_list[0]

        self._broadcast_dumped_metrics(
            metrics,
            dump_metrics_micro_batch,
            packed_batches=batch if routing_info is not None else None,
            routing_info=routing_info,
        )

        return metrics

    # 下面是 finetune 相关才用到的函数
    def _finetune_func(
        self,
        num_microbatches: int,
        seq_length: int,
        microbatch_loss_reweight: Optional[float] = None,
    ):
        cur_mbs_step = 0

        def fwd_output_and_loss_func(
            num_microbatches, seq_length, microbatch_loss_reweight, data_iterator, model
        ):
            batches: List[Dict[str, Any]] = next(data_iterator)
            unwrapped_model = unwrap_model(model)

            if self.dist_config.dynamic_context_parallel:
                prepare_data_func = self.prepare_data.sft_train_with_dynamic_cp
            else:
                prepare_data_func = self.prepare_data.sft_train
            batch, fwd_kwargs = prepare_data_func(
                batches,
                seq_length,
                self.tokenizer.pad_token_id,
                comput_attn_mask=self.training_config.comput_attn_mask,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                input_teacher_hidden_states=getattr(
                    self.config.training, "enable_teacher_kl_loss", False
                ),
                vocab_size=self.vocab_size,
            )

            fp32_output = not self.training_config.return_hidden_states_for_ce
            if not self.policy_config.ppo_pack_seq:
                model_output = model(**fwd_kwargs, fp32_output=fp32_output)
            else:
                assert not self.training_config.return_hidden_states_for_ce, "暂不支持"
                model_output = gptmodel_pack_foward(unwrapped_model, batch, fwd_kwargs, self.config)
            if isinstance(model_output, tuple):
                model_output = model_output[0]

            def loss_func(model_output):
                if self.training_config.return_hidden_states_for_ce:
                    assert isinstance(model_output, dict)
                    assert model_output["hidden_states"].dtype == torch.bfloat16
                else:
                    assert isinstance(model_output, torch.Tensor)

                nonlocal cur_mbs_step
                loss_fn = get_loss_fn("mcore", self.training_config.loss_func)
                batch["should_dump_metrics"] = self.should_dump_metrics
                loss_input = FinetuneLossInput(
                    logits=model_output if isinstance(model_output, torch.Tensor) else None,
                    batch=batch,
                    unwrapped_model=unwrapped_model,
                    skip_cp_loss_reduce=self.calc_per_token_loss,
                    linear_ce_input=model_output if isinstance(model_output, dict) else None,
                    cp_group=batch.get("cp_group", None),
                    teacher_output_weight=self.teacher_output_weight,
                    cur_mbs_step=cur_mbs_step,
                    num_mbs=num_microbatches,
                )
                result = loss_fn(self.config, loss_input)
                if microbatch_loss_reweight is not None:
                    # Reweight the per-microbatch loss so that, after Megatron
                    # divides by group_num_microbatches internally, the effective
                    # per-microbatch contribution is 1/total_num_microbatches.
                    result = (result[0] * microbatch_loss_reweight, ) + result[1:]
                cur_mbs_step += 1
                batch.pop("should_dump_metrics", None)
                return result

            return model_output, loss_func

        return partial(
            fwd_output_and_loss_func, num_microbatches, seq_length, microbatch_loss_reweight
        )

    def _pretrain_func(self, seq_length: int):
        """Lean packed-THD forward: model ``pretrain_packed`` + loss."""
        def fwd_output_and_loss_func(seq_length, data_iterator, model):
            microbatches: List[Dict[str, Any]] = next(data_iterator)
            assert len(microbatches
                      ) == 1, (f"pretrain expects micro_batch_size=1, got {len(microbatches)}")
            unwrapped_model = unwrap_model(model)
            batch, fwd_kwargs = self.prepare_data.pretrain_packed(microbatches[0])

            fp32_output = not self.training_config.return_hidden_states_for_ce
            model_output = model(**fwd_kwargs, fp32_output=fp32_output)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            if self.training_config.return_hidden_states_for_ce and mpu.is_pipeline_last_stage():
                assert isinstance(model_output, dict)
                assert model_output["hidden_states"].dtype == torch.bfloat16
            else:
                assert isinstance(model_output, torch.Tensor)

            def loss_func(model_output):
                loss_fn = get_loss_fn("mcore", self.training_config.loss_func)
                loss_input = FinetuneLossInput(
                    logits=model_output if isinstance(model_output, torch.Tensor) else None,
                    batch=batch,
                    unwrapped_model=unwrapped_model,
                    skip_cp_loss_reduce=self.calc_per_token_loss,
                    linear_ce_input=model_output if isinstance(model_output, dict) else None,
                    cp_group=batch["cp_group"],
                )
                return loss_fn(self.config, loss_input)

            return model_output, loss_func

        return partial(fwd_output_and_loss_func, seq_length)

    def _pretrain_step(
        self,
        microbatches: List[Dict[str, Any]],
        num_microbatches: int,
        forward_only: bool = False,
    ) -> Dict[str, Any]:
        """Run fwd/bwd on packed microbatches (CPU ok; H2D per mb in forward)."""
        assert num_microbatches == len(microbatches)
        assert num_microbatches >= 1
        max_seq_length = max(int(mb["tokens"].shape[-1]) for mb in microbatches)
        if self.training_config.loss_func == "square_averaging_cross_entropy":
            update_square_averaging_token_len(microbatches, max_seq_length)
        self._step_num_microbatches = num_microbatches

        # get_iterator_k_split_list expects a flat sample list; wrap each mb.
        data_iter = get_iterator_k_split_list(microbatches, num_microbatches)
        fwd_bwd_function = get_forward_backward_func()
        metrics_micro_batch = fwd_bwd_function(
            forward_step_func=self._pretrain_func(max_seq_length),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=max_seq_length,
            decoder_seq_length=max_seq_length,
            micro_batch_size=1,
        )
        if self.training_config.loss_func == "square_averaging_cross_entropy":
            for microbatch_metrics in metrics_micro_batch:
                lm_loss = microbatch_metrics.get("lm_loss")
                if torch.is_tensor(lm_loss) and lm_loss.shape == (1, ):
                    microbatch_metrics["lm_loss"] = lm_loss.squeeze(0)

        metric_prefix = "eval" if forward_only else "pretrain"
        metrics = self.collate_microbatch_metrics(metric_prefix, metrics_micro_batch)
        metrics[f"{metric_prefix}/seq_length"] = max_seq_length
        aux_metrics = self._collect_aux_metrics(num_microbatches)
        if metrics:
            metrics.update({f"{metric_prefix}/{k}": v for k, v in aux_metrics.items()})

        return metrics

    def _rm_train_func(self, seq_length: int):
        """Forward+loss for Bradley-Terry reward-model training (output_scalar).

        Reuses the scalar value head: ``model(...)`` returns ``[B, S, 1]``; the
        ``rm_bt`` loss pools the last valid token and ranks chosen vs rejected.
        """
        def fwd_output_and_loss_func(seq_length, data_iterator, model):
            assert not self.policy_config.ppo_pack_seq, \
                "reward model training (v1) does not support ppo_pack_seq"
            batches: List[Dict[str, Any]] = next(data_iterator)
            unwrapped_model = unwrap_model(model)
            batch, fwd_kwargs = self.prepare_data.rm_train(
                batches,
                seq_length,
                self.tokenizer.pad_token_id,
                comput_attn_mask=self.training_config.comput_attn_mask,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self.vocab_size,
            )

            model_output = model(**fwd_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            assert isinstance(model_output, torch.Tensor)

            def loss_func(model_output):
                loss_fn = get_loss_fn("mcore", "rm_bt")
                loss_input = FinetuneLossInput(
                    logits=model_output,
                    batch=batch,
                    unwrapped_model=unwrapped_model,
                )
                return loss_fn(self.config, loss_input)

            return model_output, loss_func

        return partial(fwd_output_and_loss_func, seq_length)

    def _calc_dynamic_mbs(self, len_batch, max_seq_length):
        dynamic_mbs_target_seqlen = self.policy_config.dynamic_mbs_target_seqlen
        if dynamic_mbs_target_seqlen is None or dynamic_mbs_target_seqlen < self.config.training.seq_length:
            dynamic_mbs_target_seqlen = self.config.training.seq_length
        dynamic_mbs = dynamic_mbs_target_seqlen // max_seq_length
        while dynamic_mbs >= 1:
            if len_batch % dynamic_mbs == 0:
                break
            dynamic_mbs -= 1
        assert dynamic_mbs >= 1
        return dynamic_mbs

    def _finetune_step_default_fwd_bwd(
        self, batch: List[Dict[str, Any]], num_microbatches: int, forward_only: bool
    ) -> Tuple[List[Dict], int, Optional[int], int]:
        """Default finetune forward-backward: uniform seq_length for all microbatches."""
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        # DPO expands each preference pair to [chosen, rejected] before this
        # function, so its effective sample count and microbatch size are doubled.
        gbs_factor = 2 if isinstance(self.config, DpoConfig) else 1
        if training_config.check_gbs_consistency and not self.dist_config.dynamic_context_parallel:
            assert training_config.train_gbs * gbs_factor == len(
                batch
            ) * dp_size, f"{training_config.train_gbs=} {len(batch)=} {dp_size=}"

        if isinstance(self.config, DpoConfig):
            max_seq_length = self._get_dpo_forward_seq_length(batch)
        if isinstance(self.config, EmbeddingConfig):
            max_seq_length = max([e['tokens'].shape[-1] for e in batch])
        elif self.config.debug.experimental_pad_to_max_length:
            max_seq_length = self.config.training.seq_length
        elif isinstance(self.config, RewardConfig):
            # RM 样本此时还没有 'tokens'，只有 chosen_tokens/rejected_tokens（rm_train 才融合）。
            pad = self.training_config.pad_to_mulitiple_of
            max_token_len = max(
                max(e["chosen_tokens"].shape[-1], e["rejected_tokens"].shape[-1]) for e in batch
            )
            max_seq_length = ((max_token_len + pad - 1) // pad) * pad
            max_seq_length = get_max_seqlen_within_dp(max_seq_length)
            max_seq_length = min(max_seq_length, training_config.seq_length)
        else:
            max_seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
            max_seq_length = get_max_seqlen_within_dp(max_seq_length)
            max_seq_length = min(max_seq_length, training_config.seq_length)

        if training_config.loss_func in ["square_averaging_cross_entropy"]:
            update_square_averaging_token_len(batch, max_seq_length)

        dynamic_mbs = 1
        if training_config.use_dynamic_mbs:
            dynamic_mbs = self._calc_dynamic_mbs(len(batch), max_seq_length)
            num_microbatches = len(batch) // dynamic_mbs

        data_iter = get_iterator_k_split_list(
            batch, num_microbatches, vpp_size=self.dist_config.virtual_pipeline_model_parallel_size
        )
        pipeline_micro_batch_size = divide(len(batch), num_microbatches)
        expected_micro_batch_size = (
            dynamic_mbs if training_config.use_dynamic_mbs else self.training_config.train_mbs *
            gbs_factor
        )
        # Embedding 的数据集比较特殊
        if not isinstance(self.config, EmbeddingConfig):
            assert pipeline_micro_batch_size == expected_micro_batch_size, (
                f"{pipeline_micro_batch_size=} {expected_micro_batch_size=} "
                f"{len(batch)=} {num_microbatches=} {gbs_factor=}"
            )
        if isinstance(self.config, RewardConfig):
            forward_step_func = self._rm_train_func(max_seq_length)
        else:
            forward_step_func = self._finetune_func(num_microbatches, max_seq_length)
        fwd_bwd_function = get_forward_backward_func()
        metrics_micro_batch = fwd_bwd_function(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=max_seq_length,
            decoder_seq_length=max_seq_length,
            micro_batch_size=pipeline_micro_batch_size,
        )
        return (
            metrics_micro_batch,
            max_seq_length,
            dynamic_mbs if training_config.use_dynamic_mbs else None,
            num_microbatches,
        )

    def _finetune_step_smart_pad_fwd_bwd(
        self, batch: List[Dict[str, Any]], forward_only: bool
    ) -> Tuple[List[Dict], int, Optional[int], int]:
        """Smart-pad finetune forward-backward: group by seqlen, fwd_bwd per group.

        This ensures each group uses its own seq_length for CP splitting, matching
        the teacher's smart_pad_compute_logits behavior.
        """
        training_config = self.config.training
        train_mbs = self.training_config.train_mbs

        # Use same grouping logic as smart_pad_compute_logits
        smart_pad_helper = CatedSmartPadInferHelper(batch, train_mbs)
        # NOTE: Cap seqlen at `training_config.seq_length` to match `smart_pad_compute_logits`.
        get_seqlen_func = lambda sample: min(sample["tokens"].shape[-1], training_config.seq_length)
        smart_pad_helper.gen_row_based_batches()
        smart_pad_helper.gen_extend_batches(get_seqlen_func)
        smart_pad_helper.gen_sorted_batches()
        smart_pad_helper.gen_smart_pad_batches(training_config.pad_to_mulitiple_of)

        # Total sample count across all seqlen groups, used for the
        # microbatch_loss_reweight computation below. Each batch_id corresponds
        # to train_mbs samples (that's how gen_extend_batches groups them).
        total_samples = sum(
            len(batch_ids) * train_mbs for batch_ids in smart_pad_helper.seqlen_batch_ids.values()
        )

        fwd_bwd_function = get_forward_backward_func()
        all_metrics = []
        last_max_seq_length = 0
        total_num_microbatches = 0

        for seqlen, batch_ids in smart_pad_helper.seqlen_batch_ids.items():
            seqlen_batches = [smart_pad_helper.extend_batches[bid] for bid in batch_ids]
            flat_samples = [sample for batch in seqlen_batches for sample in batch]
            num_group_samples = len(flat_samples)

            max_seq_length = min(seqlen, training_config.seq_length)
            max_seq_length = get_max_seqlen_within_dp(max_seq_length)
            last_max_seq_length = max(last_max_seq_length, max_seq_length)

            # Dynamic MBS: for shorter seqlens, pack more samples per microbatch to reduce
            # forward pass count. _calc_dynamic_mbs guarantees num_group_samples % dynamic_mbs == 0
            # by decrementing until divisible, so no samples are dropped.
            dynamic_mbs = train_mbs
            if training_config.use_dynamic_mbs:
                dynamic_mbs = self._calc_dynamic_mbs(num_group_samples, max_seq_length)
            group_num_microbatches = num_group_samples // dynamic_mbs
            assert group_num_microbatches * dynamic_mbs == num_group_samples, (
                f"Sample count mismatch: {group_num_microbatches=} * {dynamic_mbs=} != {num_group_samples=}"
            )

            # Regroup flat samples into microbatches of size dynamic_mbs
            regrouped_batches = [
                flat_samples[i * dynamic_mbs:(i + 1) * dynamic_mbs]
                for i in range(group_num_microbatches)
            ]

            # Gradient scale correction:
            # Megatron internally divides loss by group_num_microbatches (per fwd_bwd call).
            # We want the global gradient to equal mean over all samples, i.e. each sample
            # contributes 1/total_samples. So this group's total contribution should be
            # num_group_samples/total_samples. Megatron gives 1/group_num_microbatches per
            # microbatch (which sums to 1.0 for the group). Multiplying by
            # microbatch_loss_reweight makes the group contribute
            # num_group_samples/total_samples instead.
            microbatch_loss_reweight = num_group_samples / total_samples

            data_iter = iter(regrouped_batches)
            metrics_micro_batch = fwd_bwd_function(
                forward_step_func=self._finetune_func(
                    group_num_microbatches,
                    max_seq_length,
                    microbatch_loss_reweight=microbatch_loss_reweight
                ),
                data_iterator=data_iter,
                model=self.model,
                num_microbatches=group_num_microbatches,
                forward_only=forward_only,
                seq_length=max_seq_length,
                decoder_seq_length=max_seq_length,
                micro_batch_size=dynamic_mbs,
            )
            all_metrics.extend(metrics_micro_batch)
            total_num_microbatches += group_num_microbatches
            # Smart-pad issues many `fwd_bwd_function` calls per training step (one per
            # seqlen bucket), so explicitly clear memory to avoid fragmentation.
            clear_memory()
        # hard to return dynamic_mbs here since it varies for each group
        return all_metrics, last_max_seq_length, None, total_num_microbatches

    def _qat_step_context(self):
        """Weight + attn activation QAT for one finetune fwd/bwd step."""
        if qat_parameters_context is None:
            return nullcontext()
        # Prefer fp4 over fp8 when both are set (matches HF DSV4 MoE QAT order).
        if getattr(self.policy_config, "fp4_qat", False):
            qat_type = "fp4"
        elif getattr(self.policy_config, "fp8_qat", False):
            qat_type = "fp8"
        else:
            qat_type = None
        return qat_parameters_context(self.model, qat_type=qat_type)

    def _finetune_step(
        self, batch: List[Dict[str, Any]], num_microbatches: int, forward_only: bool = False
    ):
        training_config = self.config.training
        assert not (
            self.dist_config.dynamic_context_parallel and self.should_dump_metrics
        ), "SFT dump metrics with dynamic_context_parallel is not implemented yet"

        if isinstance(self.config, RewardConfig):
            assert not self.policy_config.smart_pad_train, \
                "reward model training (v1) does not support smart_pad_train"
            assert not self.dist_config.dynamic_context_parallel, \
                "reward model training (v1) does not support dynamic_context_parallel"
            assert not training_config.use_dynamic_mbs, \
                "reward model training (v1) does not support use_dynamic_mbs"

        # Type-select Linear/GroupedLinear weights, in-place simulate_qat for this step's forwards
        with self._qat_step_context():
            if self.policy_config.smart_pad_train:
                assert not self.dist_config.dynamic_context_parallel
                metrics_micro_batch, max_seq_length, dynamic_mbs, actual_num_microbatches = (
                    self._finetune_step_smart_pad_fwd_bwd(batch, forward_only)
                )
            else:
                metrics_micro_batch, max_seq_length, dynamic_mbs, actual_num_microbatches = (
                    self._finetune_step_default_fwd_bwd(batch, num_microbatches, forward_only)
                )

        metric_prefix = "finetune"
        if forward_only:
            metric_prefix = "eval"
        metrics = {}
        dump_metrics_micro_batch = self._extract_dump_metrics_micro_batch(metrics_micro_batch)
        if is_pipeline_last_stage() and len(metrics_micro_batch) > 0:
            token_level_accumulated = {}
            scalar_accumulated = {}

            for key in metrics_micro_batch[0].keys():
                if key.startswith("dump/"):
                    continue
                values = torch.stack([loss_reduced[key] for loss_reduced in metrics_micro_batch])
                if values.dim() == 2 and values.shape[1] == 2:
                    token_level_accumulated[key] = values.sum(dim=0)
                else:
                    scalar_accumulated[key] = values.mean()

            if token_level_accumulated:
                tk_keys = sorted(token_level_accumulated.keys())
                all_vals = torch.stack([token_level_accumulated[k] for k in tk_keys])
                torch.distributed.all_reduce(
                    all_vals, group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                for i, k in enumerate(tk_keys):
                    token_level_accumulated[k] = all_vals[i]

            if is_last_rank():
                metrics = {f"{metric_prefix}/seq_length": max_seq_length}
                for key, val in token_level_accumulated.items():
                    metrics[f"{metric_prefix}/{key}"] = (val[0] / val[1].clamp(min=1)).cpu().item()
                for key, val in scalar_accumulated.items():
                    metrics[f"{metric_prefix}/{key}"] = val.cpu().item()
            if dynamic_mbs is not None:
                metrics[f"{metric_prefix}/dynamic_mbs"] = dynamic_mbs

        # Surface aux metrics (e.g. MTP per-layer loss tracked inside the model's
        # process_mtp_loss) in the SFT/eval path too. Must run on all ranks since
        # it does collective reductions and clears the loss trackers each step.
        aux_metrics = self._collect_aux_metrics(actual_num_microbatches)
        if metrics:
            metrics.update({f"{metric_prefix}/{k}": v for k, v in aux_metrics.items()})

        obj_list = [metrics]
        torch.distributed.broadcast_object_list(
            obj_list, get_last_rank(cpu_group()), group=cpu_group()
        )
        metrics = obj_list[0]

        self._broadcast_dumped_metrics(metrics, dump_metrics_micro_batch)

        return metrics

    def get_logits_or_hidden_state_only_func(
        self, seq_len, inference_only=True, return_logits=True
    ):
        def logits_output_only_func(seq_len, dataloader_iter, model):
            batches: List[Dict[str, Any]] = next(dataloader_iter)
            # log for batch_get_*_logprobs
            if self.total_iters > 0 and torch.distributed.get_rank() == 0:
                self.batch_iters += 1
                log(f"{self.batch_log_str} {self.batch_iters:8d}/{self.total_iters:8d}", rank=0)

            model_fwd_args = self.prepare_data.model_forward_only(
                batches,
                seq_len,
                self.tokenizer.pad_token_id,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self.vocab_size,
            )

            target = model_fwd_args.pop("target")
            output_tensor = model(**model_fwd_args)

            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            if (not return_logits and mpu.is_pipeline_last_stage()):
                assert self.training_config.use_linear_ce
                assert isinstance(output_tensor, dict)
                hidden_states = output_tensor["hidden_states"]
                output_layer = output_tensor["output_layer"]
                if getattr(output_layer, "sequence_parallel", False):
                    hidden_states = (
                        tensor_parallel.gather_from_sequence_parallel_region(
                            hidden_states,
                            tensor_parallel_output_grad=False,
                        )
                    )
                output_tensor = hidden_states.transpose(0, 1).contiguous()
            else:
                assert isinstance(output_tensor, torch.Tensor)

            def id_func(output_tensor, non_loss_data=True):
                # TODO(@nrwu): 检查 sp 情况下，此处 output tensor shape 是否应该是 [b, s, v//tp] ？
                return output_tensor

            return output_tensor, id_func

        return partial(logits_output_only_func, seq_len)

    @torch.no_grad()
    def _compute_logits_or_hidden_states_impl(
        self,
        model,
        batches_list: List[Dict[str, Any]],
        batch_log_str: str,
        return_logits: bool = True,
    ):
        total_samples = len(batches_list)
        self.batch_iters = 0
        self.batch_log_str = batch_log_str
        if not return_logits:
            assert self.training_config.use_linear_ce
            num_microbatches = divide(total_samples, self.forward_only_mbs)
            calc_batches_list = batches_list
            self.total_iters = total_samples // self.forward_only_mbs
        elif self.config.distill.enable_data_with_alpha:
            num_w_alpha = sum(1 for data in batches_list if data.get('offpd_loss_alpha', 0) > 0)
            num_w_alpah_ceil = (
                num_w_alpha + self.forward_only_mbs - 1
            ) // self.forward_only_mbs * self.forward_only_mbs
            num_microbatches = divide(num_w_alpah_ceil, self.forward_only_mbs)
            # 前面已经把要计算 kl 的数据放前了
            calc_batches_list = batches_list[:num_w_alpah_ceil]
            self.total_iters = (num_w_alpah_ceil) // self.forward_only_mbs
        else:
            num_microbatches = divide(total_samples, self.forward_only_mbs)
            calc_batches_list = batches_list
            self.total_iters = total_samples // self.forward_only_mbs

        seq_length = get_batches_max_seqlen(batches_list, self.training_config.pad_to_mulitiple_of)
        seq_length = get_max_seqlen_within_dp(seq_length)
        max_seq_length = min(seq_length, self.training_config.seq_length)

        batch_iter = get_iterator_k_split_list(calc_batches_list, num_microbatches)

        target_logits_dtype = (
            torch.bfloat16 if self.config.training.teacher_logits_dtype == "bf16" else torch.float32
        )
        num_b_per_execution = self.config.training.num_batches_per_execution
        num_chunks = (num_microbatches + num_b_per_execution - 1) // num_b_per_execution

        fwd_bwd_function = get_forward_backward_func()

        outputs = None
        if mpu.is_pipeline_last_stage():
            assert max_seq_length % mpu.get_context_parallel_world_size(
            ) == 0, f"{max_seq_length=} {mpu.get_context_parallel_world_size()=}"
            seq_length_shard = max_seq_length // mpu.get_context_parallel_world_size()

            if not return_logits:
                outputs = []
            else:
                assert self.vocab_size % mpu.get_tensor_model_parallel_world_size(
                ) == 0, f"{self.vocab_size=} {mpu.get_tensor_model_parallel_world_size()=}"
                vacab_size_shard = self.vocab_size // mpu.get_tensor_model_parallel_world_size()
                if self.logits_cpu_buffer is None or seq_length_shard > self.logits_cpu_buffer.shape[
                    1]:
                    self.logits_cpu_buffer = torch.empty(
                        (total_samples, seq_length_shard, vacab_size_shard),
                        dtype=target_logits_dtype,
                        device="cpu",
                        pin_memory=True
                    )
                offset = 0

        for ni in range(num_chunks):
            act_gas = min(num_b_per_execution, num_microbatches - ni * num_b_per_execution)
            output_list = fwd_bwd_function(
                forward_step_func=self.get_logits_or_hidden_state_only_func(
                    max_seq_length,
                    inference_only=True,
                    return_logits=return_logits,
                ),
                data_iterator=batch_iter,
                model=model,
                num_microbatches=act_gas,
                forward_only=True,
                seq_length=max_seq_length,
                micro_batch_size=self.forward_only_mbs,
                collect_non_loss_data=True,
                decoder_seq_length=max_seq_length,
            )
            if mpu.is_pipeline_last_stage():
                assert len(output_list) > 0
                assert len(output_list) * output_list[0].shape[
                    0
                ] == act_gas * self.forward_only_mbs, f"Expected {act_gas=}  {self.forward_only_mbs=} {len(output_list)=} {output_list[0].shape=}"

                if not return_logits:
                    for hidden_states in output_list:
                        outputs.extend(hidden_states.unbind(0))
                else:
                    for output in output_list:
                        assert output.shape == output_list[0].shape
                        b = output.shape[0]
                        cpu_slice = self.logits_cpu_buffer[offset:offset + b, :seq_length_shard]
                        cpu_slice.copy_(output, non_blocking=True)
                        offset += b
            else:
                assert len(output_list) == 0

        if mpu.is_pipeline_last_stage():
            if return_logits:
                torch.cuda.synchronize()
                outputs = [
                    lg.squeeze(0) for i, lg in
                    enumerate(self.logits_cpu_buffer[:, :seq_length_shard, :].chunk(total_samples))
                ]

            assert len(
                outputs
            ) == total_samples, f"Expected {total_samples} outputs but got {len(outputs)}"
        clear_memory()
        return outputs

    @torch.no_grad()
    def _compute_values(self, model, batches_list: List[Dict[str, Any]], batch_log_str: str):
        """ value model """
        self.batch_iters = 0
        self.total_iters = len(batches_list) // self.forward_only_mbs
        self.batch_log_str = batch_log_str

        total_samples = len(batches_list)
        seq_length = get_batches_max_seqlen(batches_list, self.training_config.pad_to_mulitiple_of)
        seq_length = get_max_seqlen_within_ep(seq_length)
        num_microbatches = divide(total_samples, self.forward_only_mbs)
        batch_iter = get_iterator_k_split_list(batches_list, num_microbatches)
        fwd_bwd_function = get_forward_backward_func()
        values_list = fwd_bwd_function(
            forward_step_func=self.get_logits_or_hidden_state_only_func(
                seq_length, inference_only=True
            ),  # reuse get_logits_or_hidden_state_only_func
            data_iterator=batch_iter,
            model=model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=self.forward_only_mbs,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )
        values = torch.cat(values_list).squeeze(-1) if len(values_list) > 0 else None
        if values is not None and mpu.get_context_parallel_world_size() > 1:
            values = all_gather_from_context_parallel_region(values)

        # [bs, seq_len - 1]
        values = values[:, :-1].clone() if values is not None else None

        # Broadcast it from last PP stage to everything else.
        values = BroadcastUtils.broadcast_2d_tensor_within_pp(values)
        assert values.dtype == torch.float32, f'{values.dtype=}'

        assert values.shape[0] == total_samples
        values = [values.squeeze(0) for values in values.cpu().chunk(total_samples)]
        clear_memory()
        return values

    def value_model_forward_step(self, seq_length: int):
        def fwd_output_and_loss_func(seq_length, data_iterator, model):
            batches: List[Dict[str, Any]] = next(data_iterator)
            unwrapped_model = unwrap_model(model)
            batch, fwd_kwargs = self.prepare_data.ppo_value_train(
                batches,
                seq_length,
                self.tokenizer.pad_token_id,
                ppo_pack_seq=self.policy_config.ppo_pack_seq,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self.vocab_size,
            )

            for key in ["mask", "returns", "values"]:
                assert key in batch, f'{key} not in batch'

            if not self.policy_config.ppo_pack_seq:
                values = model(**fwd_kwargs)
            else:
                values = gptmodel_pack_foward(unwrapped_model, batch, fwd_kwargs, self.config)

            if isinstance(values, tuple):
                values = values[0]
            assert isinstance(values, torch.Tensor)

            def loss_func(values):
                # [bs, seq_len - 1]
                values = values.squeeze(dim=-1)
                if mpu.get_context_parallel_world_size(
                ) > 1 and not self.policy_config.ppo_pack_seq:
                    values = all_gather_from_context_parallel_region(values)
                values = values[:, :-1]
                fn = get_loss_fn("mcore", "ppo_value_loss")
                old_values = batch["values"]
                returns = batch["returns"]
                mask = batch["mask"]
                return fn(self.ppo_config, old_values, values, returns, mask)
                # TODO(hessianliu): fill this

            return values, loss_func

        return partial(fwd_output_and_loss_func, seq_length)

    def _update_value_model(self, batch: List[Dict[str, Any]], num_microbatches: int):
        policy_config = self.policy_config

        seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
        if policy_config.dynamic_mbs_target_seqlen is not None:
            seq_length = get_max_seqlen_within_dp(seq_length)
        else:
            seq_length = get_max_seqlen_within_ep(seq_length)

        dynamic_num_microbatches = 0
        dynamic_mbs = 0
        batch_size = len(batch)

        if policy_config.dynamic_mbs_target_seqlen is not None:
            dynamic_mbs = policy_config.dynamic_mbs_target_seqlen // seq_length * self.training_config.train_mbs
            if dynamic_mbs == 0:
                dynamic_mbs = 1
            dynamic_mbs = min(policy_config.dynamic_mbs_limit, dynamic_mbs)
            while dynamic_mbs >= 1:
                if batch_size % dynamic_mbs == 0:
                    dynamic_num_microbatches = batch_size // dynamic_mbs
                    break
                else:
                    dynamic_mbs -= 1
            if dynamic_num_microbatches > 0:
                assert dynamic_num_microbatches * dynamic_mbs == batch_size, \
                    f"{dynamic_mbs=} {dynamic_num_microbatches=} {batch_size=} mismatch!"

        enable_dynamic_mbs = True if dynamic_num_microbatches > 0 else False
        log(
            f"[TRAIN] {seq_length=} {batch_size=} {num_microbatches=} {enable_dynamic_mbs=} {dynamic_num_microbatches=} {dynamic_mbs=}",
            rank=0
        )

        data_iter = get_iterator_k_split_list(
            batch, dynamic_num_microbatches if enable_dynamic_mbs else num_microbatches
        )

        fwd_bwd_function = get_forward_backward_func()

        metrics_micro_batch = fwd_bwd_function(
            forward_step_func=self.value_model_forward_step(seq_length),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=dynamic_num_microbatches if enable_dynamic_mbs else num_microbatches,
            forward_only=False,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=dynamic_mbs if enable_dynamic_mbs else self.training_config.train_mbs,
        )

        metrics = {}
        if is_pipeline_last_stage() and len(metrics_micro_batch) > 0:
            token_level_accumulated = {}
            scalar_accumulated = {}

            for key in metrics_micro_batch[0].keys():
                values = torch.stack([loss_reduced[key] for loss_reduced in metrics_micro_batch])
                if values.dim() == 2 and values.shape[1] == 2:
                    token_level_accumulated[key] = values.sum(dim=0)
                else:
                    scalar_accumulated[key] = values.mean()

            if token_level_accumulated:
                tk_keys = sorted(token_level_accumulated.keys())
                all_vals = torch.stack([token_level_accumulated[k] for k in tk_keys])
                torch.distributed.all_reduce(all_vals, group=mpu.get_data_parallel_group())
                for i, k in enumerate(tk_keys):
                    token_level_accumulated[k] = all_vals[i]

            if is_last_rank():
                metrics = {"value/seq_length": seq_length}
                for key, val in token_level_accumulated.items():
                    metrics[f"value/{key}"] = (val[0] / val[1].clamp(min=1)).cpu().item()
                for key, val in scalar_accumulated.items():
                    metrics[f"value/{key}"] = val.cpu().item()

        obj_list = [metrics]
        torch.distributed.broadcast_object_list(
            obj_list, get_last_rank(cpu_group()), group=cpu_group()
        )
        metrics = obj_list[0]

        return metrics

    def _collect_aux_metrics(self, num_microbatches):
        """Collect aux metrics."""
        total_loss_dict = {}
        # 1、 collect moe metric
        mcore_config = self.get_mcore_config()
        # collect MoE metrics.
        if mcore_config.num_moe_experts is not None and get_moe_metrics_tracker is not None:
            moe_loss_scale = 1 / num_microbatches
            track_names = []
            if "aux_loss" in mcore_config.moe_router_load_balancing_type:
                track_names.append("load_balancing_loss")
            if "seq_aux_loss" in mcore_config.moe_router_load_balancing_type:
                track_names.append("seq_load_balancing_loss")
            if "global_aux_loss" in mcore_config.moe_router_load_balancing_type:
                track_names.append("global_load_balancing_loss")
            if mcore_config.moe_z_loss_coeff is not None:
                track_names.append("z_loss")

            hybrid_layer_pattern = None
            try:
                hybrid_layer_pattern = get_attr_wrapped_model(self.model[0], "hybrid_layer_pattern")
            except Exception as e:
                pass

            if hybrid_layer_pattern is not None:
                from operator import itemgetter

                from megatron.core.ssm.mamba_hybrid_layer_allocation import (
                    Symbols,
                    get_hybrid_layer_counts,
                )

                layers = itemgetter(Symbols.MOE)(get_hybrid_layer_counts(hybrid_layer_pattern))
            else:
                layers = mcore_config.num_layers

            model_pg_collection = get_attr_wrapped_model(self.model[0], "pg_collection")
            moe_log_string = get_moe_metrics_tracker().report(
                loss_scale=moe_loss_scale,
                iteration=None,
                writer=None,
                wandb_writer=None,
                per_layer_logging=mcore_config.moe_per_layer_logging,
                force_initialize=True,
                track_names=track_names,
                num_layers=layers,
                moe_layer_freq=mcore_config.moe_layer_freq,
                mtp_num_layers=mcore_config.mtp_num_layers,
                pg_collection=model_pg_collection,
                total_loss_dict=total_loss_dict,
            )

            if getattr(
                mcore_config, 'log_moe_overload_factor', False
            ) and get_moe_overload_factor_tracker is not None:

                class _CaptureWriter:
                    def add_scalar(self, name, value, iteration):
                        t = torch.tensor(value, dtype=torch.float32)
                        if name in total_loss_dict:
                            total_loss_dict[name] += t
                        else:
                            total_loss_dict[name] = t

                get_moe_overload_factor_tracker().report(
                    iteration=None,
                    writer=_CaptureWriter(),
                    wandb_writer=None,
                    per_layer_logging=mcore_config.moe_per_layer_logging,
                )

        # 2、 collect MTP metrics.
        if mcore_config.mtp_num_layers is not None and MTPLossLoggingHelper is not None:
            mtp_loss_scale = 1 / num_microbatches
            MTPLossLoggingHelper.track_mtp_metrics(
                mtp_loss_scale, None, None, None, total_loss_dict
            )

        # 3、Track sparse attention indexer loss
        if DSAIndexerLossLoggingHelper is not None and mcore_config.dsa_indexer_loss_coeff is not None and mcore_config.dsa_indexer_loss_coeff > 0:
            indexer_loss_scale = 1 / num_microbatches
            DSAIndexerLossLoggingHelper.track_indexer_metrics(
                loss_scale=indexer_loss_scale,
                iteration=None,
                writer=None,
                wandb_writer=None,
                total_loss_dict=total_loss_dict,
            )
        total_loss_dict = {k: v.cpu() for k, v in total_loss_dict.items()}
        return total_loss_dict

    @torch.no_grad()
    def _embedding_forward_only(
        self, batch: List[Dict[str, Any]], num_microbatches: int, step: int
    ):
        """Default finetune forward-backward: uniform seq_length for all microbatches."""
        assert isinstance(self.config, EmbeddingConfig)

        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        # DPO expands each preference pair to [chosen, rejected] before this
        # function, so its effective sample count and microbatch size are doubled.
        gbs_factor = 2 if isinstance(self.config, DpoConfig) else 1
        if training_config.check_gbs_consistency and not self.dist_config.dynamic_context_parallel:
            assert training_config.train_gbs * gbs_factor == len(
                batch
            ) * dp_size, f"{training_config.train_gbs=} {len(batch)=} {dp_size=}"

        max_seq_length = max([e['tokens'].shape[-1] for e in batch])

        dynamic_mbs = 1
        if training_config.use_dynamic_mbs:
            dynamic_mbs = self._calc_dynamic_mbs(len(batch), max_seq_length)
            num_microbatches = len(batch) // dynamic_mbs

        data_iter = get_iterator_k_split_list(batch, num_microbatches)
        pipeline_micro_batch_size = divide(len(batch), num_microbatches)
        expected_micro_batch_size = (
            dynamic_mbs if training_config.use_dynamic_mbs else self.training_config.train_mbs *
            gbs_factor
        )
        # Embedding / pack_seq：每 microbatch 是 1 条 packed THD，不是 train_mbs。
        if not isinstance(self.config, EmbeddingConfig):
            assert pipeline_micro_batch_size == expected_micro_batch_size, (
                f"{pipeline_micro_batch_size=} {expected_micro_batch_size=} "
                f"{len(batch)=} {num_microbatches=} {gbs_factor=}"
            )
        forward_step_func = self._embedding_forward_func(max_seq_length)
        fwd_bwd_function = get_forward_backward_func()
        metrics_micro_batch = fwd_bwd_function(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=max_seq_length,
            decoder_seq_length=max_seq_length,
            micro_batch_size=pipeline_micro_batch_size,
        )

        return metrics_micro_batch

    @torch.no_grad()
    # 下面是 finetune 相关才用到的函数
    def _embedding_forward_func(
        self,
        seq_length: int,
        microbatch_loss_reweight: Optional[float] = None,
    ):
        def fwd_output_and_loss_func(seq_length, microbatch_loss_reweight, data_iterator, model):
            batches: List[Dict[str, Any]] = next(data_iterator)
            unwrapped_model = unwrap_model(model)

            if self.dist_config.dynamic_context_parallel:
                prepare_data_func = self.prepare_data.sft_train_with_dynamic_cp
            else:
                prepare_data_func = self.prepare_data.sft_train
            batch, fwd_kwargs = prepare_data_func(
                batches,
                seq_length,
                self.tokenizer.pad_token_id,
                comput_attn_mask=self.training_config.comput_attn_mask,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                input_teacher_hidden_states=getattr(
                    self.config.training, "enable_teacher_kl_loss", False
                ),
                vocab_size=self.vocab_size,
            )

            fp32_output = not self.training_config.return_hidden_states_for_ce
            if not self.policy_config.ppo_pack_seq:
                model_output = model(**fwd_kwargs, fp32_output=fp32_output)
            else:
                assert not self.training_config.return_hidden_states_for_ce, "暂不支持"
                model_output = gptmodel_pack_foward(unwrapped_model, batch, fwd_kwargs, self.config)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            if self.training_config.return_hidden_states_for_ce and mpu.is_pipeline_last_stage():
                assert isinstance(model_output, dict)
                assert model_output["hidden_states"].dtype == torch.bfloat16
                model_output['batch'] = batch
            else:
                assert isinstance(model_output, torch.Tensor)

            return model_output, None

        return partial(fwd_output_and_loss_func, seq_length, microbatch_loss_reweight)


class MetricMixin:
    def reduce_optimizer_stats_across_model_parallel_group(
        self,
        update_successful: bool,
        grad_norm: Optional[float],
        num_zeros_in_grad: Optional[float],
        post_clip_grad_norm: Optional[float],
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
        """One Gloo MP all-gather for optimizer/logging stats, then AND / MAX.

        Payload keeps original types:
        ``[bool, Optional[float], Optional[float], Optional[float]]``.
        """
        mp_group = get_model_parallel_group_gloo()
        local = [update_successful, grad_norm, num_zeros_in_grad, post_clip_grad_norm]
        gathered: List[Optional[list]] = [None] * dist.get_world_size(group=mp_group)
        dist.all_gather_object(gathered, local, group=mp_group)

        cols = list(zip(*gathered))
        update_successful = all(cols[0])

        def _max_or_none(vals) -> Optional[float]:
            present = [v for v in vals if v is not None]
            return max(present) if present else None

        return (
            update_successful,
            _max_or_none(cols[1]),
            _max_or_none(cols[2]),
            _max_or_none(cols[3]),
        )

    def reduce_metrics_across_data_parallel_group(self, metrics: Dict[str, Any]):
        """Gloo DP all-gather 后按 key 规约成可上报的标量。

        应在 actor 热路径**最后**调用（timers / MFU 等都加完之后）。
        调用后新增到 ``metrics`` 的字段是当前 rank 的本地值，不会参与规约。

        规约规则（先把各 DP rank 的同名 value 拼成 list，再按 key 规约）：
        _check_metric 检查 val 的合法性，要求 val 内元素属性要一致
        **key 是禁止与 "dump/" 开头的**

        - 全是 ``None`` → ``None``（文本日志会跳过）
        - tensor 且 shape 为 ``[*, 2]``（``[sum, count]``）→
          ``sum(sums) / max(sum(counts), 1)``（token 加权均值）
        - 其它 tensor → stack 后 ``tolist``，再走下面的后缀规则
        - key 以 ``_max`` 结尾 → ``max``
        - key 以 ``_min`` 结尾 → ``min``
        - key 以 ``_sum`` 结尾 → ``sum``
        - 其它 → ``mean``
        """
        def _to_cpu(val: Any) -> Any:
            if isinstance(val, torch.Tensor):
                return val.detach().cpu()
            if isinstance(val, list):
                return [_to_cpu(v) for v in val]
            return val

        def _reduce_metrics(key: str, val: List[Any]) -> Optional[float]:
            if val[0] is None:
                return None

            if isinstance(val[0], torch.Tensor):
                val = torch.stack(val)
                # token-weighted: list of [sum, count] → shape [num_ranks, 2]
                if val.dim() == 2 and val.shape[1] == 2:
                    val = val.sum(dim=0)
                    return (val[0] / val[1].clamp(min=1)).item()
                # stack of 0-D scalars → 1-D; fall through to suffix reduce
                assert val.dim(
                ) == 1, f"unsupported tensor metric shape for {key=}: {tuple(stacked.shape)}; only 0-D scalars or [*, 2] (sum, count) are allowed"
                val = val.tolist()

            if key.endswith('_max'):
                return np.max(val)
            elif key.endswith('_min'):
                return np.min(val)
            elif key.endswith('_sum'):
                return np.sum(val)

            return np.mean(val)

        def _check_metric(val: list[Any]) -> bool:
            """Validate a merged metric value list before reduce.

            Rules:
            - ``val`` is a non-empty list
            - every element is ``int``/``float``, or every element is ``Tensor``
              (no mixing)
            - tensors are CPU-only, share the same ``shape`` and ``dtype``
            - each tensor has dim 0 or 1 so ``torch.stack(val).dim()`` is 1 or 2
            """
            if not isinstance(val, list) or len(val) == 0:
                return False

            first = val[0]
            if first is None:
                return all(v is None for v in val)

            if isinstance(first, torch.Tensor):
                if first.is_cuda or first.dim() not in (0, 1):
                    return False
                if first.dim() == 1 and tuple(first.shape) != (2, ):
                    return False
                ref_shape = tuple(first.shape)
                ref_dtype = first.dtype
                for v in val:
                    if not isinstance(v, torch.Tensor):
                        return False
                    if (
                        v.is_cuda or v.dim() not in (0, 1) or tuple(v.shape) != ref_shape or
                        v.dtype != ref_dtype
                    ):
                        return False
                return True

            if isinstance(first, (int, float)):
                return all(isinstance(v, (int, float)) for v in val)

            return False

        dp_group = mpu.get_data_parallel_group_gloo()
        local = {key: _to_cpu(val) for key, val in metrics.items()}

        gathered: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size(group=dp_group)
        dist.all_gather_object(gathered, local, group=dp_group)

        merged: Dict[str, list] = {}
        for rank_metrics in gathered:
            assert rank_metrics is not None
            for key, val in rank_metrics.items():
                if not isinstance(val, list):
                    val = [val]
                merged.setdefault(key, []).extend(val)

        reduce_metrics = {}
        for key, val in merged.items():
            assert _check_metric(val), f"invalid metric values for {key=}: {val=}"
            reduce_metrics[key] = _reduce_metrics(key, val)
        # NOTE(guanyouhe): 理论上这里要向所有的 rank 广播 bast-rank 的 reduce_metrics
        # 但打印与上报什么的都用 last-rank，所以这里不广播了
        return reduce_metrics
