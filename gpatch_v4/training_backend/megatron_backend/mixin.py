import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
import torch.distributed as dist
from einops import rearrange
from transformers import AutoConfig, AutoTokenizer

from megatron.core import mpu
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

from gpatch_v4.configs.config import DpoConfig, OnPolicyDistillConfig, RewardConfig
from gpatch_v4.configs.transformer_config import merge_core_transformer_config
from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
from gpatch_v4.core.parallel_state import (
    cpu_group,
    get_last_rank,
    is_last_rank,
    is_mp_and_cp_head,
    is_mp_head,
)
from gpatch_v4.core.smart_pad_helper import CatedSmartPadInferHelper, _sample_idx_key
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.loss_factory import (
    FinetuneLossInput,
    PolicyLossInput,
    get_policy_loss_fn,
)
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    bridge_save_hf,
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
from gpatch_v4.utils.training_utils import (
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
    logprobs_from_linear_ce,
    masked_mean,
    update_square_averaging_token_len,
)


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
        if self.training_config.use_linear_ce:
            if override_transformer_config is None:
                override_transformer_config = {}
            override_transformer_config['cross_entropy_loss_fusion'] = True
            override_transformer_config['cross_entropy_fusion_impl'] = 'linear'
            logging_rank0(
                "use_linear_ce enabled: inject cross_entropy_loss_fusion=True, "
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
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
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
            ddp_config_dict = {"use_distributed_optimizer": True}
            ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
            ddp_config.finalize()

        with profile_memory_and_time(f"get {model_type} from megatron_bridge", rank=0):
            model = self.provider.provide_distributed_model(
                wrap_with_ddp=wrap_with_ddp,
                ddp_config=ddp_config,
            )

            tf_config = get_model_config(model[0] if isinstance(model, list) else model)

            if load_weights_from_bridge:
                logging_rank0(f"loading {model_type} weights from megatron_bridge {hf_model_path=}")
                allowed_mismatched_params = []
                bridge.load_hf_weights(
                    model, hf_model_path, allowed_mismatched_params=allowed_mismatched_params
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
            model = bridge.get_model(
                bf16=True,
                wrap_with_ddp=wrap_with_ddp and not post_wrap_with_ddp,
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
    def save_checkpoint(self, global_step: int):
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
        )

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

    def get_cur_cp_tp_index(self, index):
        config = self.get_mcore_config()
        if config.context_parallel_size > 1:
            index = get_tensor_on_this_cp_rank(index, seq_dim=0)
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            index = index.view(config.tensor_model_parallel_size,
                               -1)[mpu.get_tensor_model_parallel_rank()]
        return index

    @torch.no_grad()
    def prepare_for_router_replay(self, batches: List[Dict[str, Any]], seqlen: int):
        config = self.get_mcore_config()
        num_layer = get_num_layers_to_build(config)
        offset = get_transformer_layer_offset(config)
        full_index = torch.arange(seqlen, device="cuda").long()
        routed_experts = []
        for batch in batches:
            # seq, layer, experts
            routed_experts_batch = batch["routed_experts"]
            # real seqlen of the request before padding - 1, last token is not calculated by inference engine (hence no router), neither by train engine for loss
            seq_batch = routed_experts_batch.shape[0] - 1
            # index = full_index[:seq_batch] + some kind of random router padding
            index = (
                full_index + (full_index // seq_batch) * 6 * mpu.get_data_parallel_rank()
            ) % seq_batch
            # support cp and sequence_parallel
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

    def maybe_prepare_for_router_replay(self, batches: List[Dict[str, Any]], seqlen: int):
        if self.get_router_replay_manager().enabled:
            self.prepare_for_router_replay(batches, seqlen)

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


class ForwardStepMixin(RouterReplayMixin):
    @property
    def calc_per_token_loss(self) -> bool:
        """Whether per-token gradient normalization is enabled."""
        model = self.model[0] if isinstance(self.model, list) else self.model
        return get_model_config(model).calculate_per_token_loss

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
        当 ``use_linear_ce`` 启用时，直接从 hidden states 计算 logprobs，
        避免物化完整 logits 张量，大幅减少显存占用。
        """
        if self.training_config.use_linear_ce:
            assert not compute_topk and gather_target_ids_key is None, (
                "linear_ce 与 compute_topk/gather_target_ids_key 不兼容，因为不生成完整 logits"
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
                model_fwd_args = self.prepare_data.model_forward_only(
                    batches,
                    seq_len,
                    self.tokenizer.pad_token_id,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self.vocab_size,
                )
                target = model_fwd_args.pop("target")

                use_linear_ce = self.training_config.use_linear_ce
                fp32_output = not use_linear_ce
                model_output = model(**model_fwd_args, fp32_output=fp32_output)

                if self.config.training.forward_clear_memory and \
                   self.batch_iters % self.config.training.forward_clear_memory_interval == 0:
                    logging_rank0(f"clear memory in forward only {self.batch_iters=}")
                    clear_memory()

                if isinstance(model_output, tuple):
                    model_output = model_output[0]
                if use_linear_ce and mpu.is_pipeline_last_stage():
                    assert isinstance(model_output, dict)
                    assert model_output["hidden_states"].dtype == torch.bfloat16
                    linear_ce_output = model_output
                    model_output = model_output["hidden_states"]
                else:
                    assert isinstance(model_output, torch.Tensor)

                def id_func(model_output, non_loss_data=True):
                    if use_linear_ce:
                        logprobs = logprobs_from_linear_ce(
                            linear_ce_backend=self.training_config.linear_ce_backend,
                            linear_ce_output=linear_ce_output,
                            target=target,
                            ignore_cp=False,
                            pre_shifted=False,
                        )
                    else:
                        # TODO(@nrwu): 检查 sp 情况下，此处 output tensor shape 是否应该是 [b, s, v//tp] ？
                        if compute_topk or gather_target_ids_key is not None:
                            output_tensor_for_topk = model_output.clone()
                        logprobs = from_parallel_logits_to_logprobs(
                            vocab_parallel_logits=model_output,
                            target=target,
                            inference_only=inference_only
                        )
                    if not compute_topk and gather_target_ids_key is None:
                        return logprobs

                    result = {"logprobs": logprobs}
                    if compute_topk:
                        topk_logprobs, topk_token_ids = from_parallel_logits_to_topk_logprobs(
                            vocab_parallel_logits=output_tensor_for_topk,
                            topk=getattr(self.ppo_config, "log_prob_top_k", 0),
                        )
                        result["topk_logprobs"] = topk_logprobs[:, :-1].contiguous()
                        result["topk_ids"] = topk_token_ids[:, :-1].to(torch.int32).contiguous()
                    if gather_target_ids_key is not None:
                        # 将各 sample 的 [S_i-1, K] ids pad 到 [B, seq_len, K]，超出部分填 0。
                        per_sample = [b[gather_target_ids_key] for b in batches]
                        k_dim = per_sample[0].shape[-1]
                        padded_ids = torch.zeros(
                            (model_output.shape[0], seq_len, k_dim),
                            dtype=torch.long,
                            device=model_output.device,
                        )
                        for i, ids in enumerate(per_sample):
                            li = min(ids.shape[0], seq_len)
                            padded_ids[i, :li] = ids[:li].to(model_output.device, dtype=torch.long)
                        gather_lp = from_parallel_logits_to_opd_topk_logprobs(
                            vocab_parallel_logits=output_tensor_for_topk,
                            target_ids=padded_ids,
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
            mb_data = next(dataloader_iter)

            self.batch_iters += 1
            if self.total_iters > 0 and torch.distributed.get_rank() == 0:
                log(f"{self.batch_log_str} {self.batch_iters:8d}/{self.total_iters:8d}", rank=0)

            mb_copy = dict(mb_data)
            batch, fwd_kwargs = self.prepare_data.grpo_train_with_dynamic_cp(
                [mb_copy],
                seqlen=seqlen,
                pad_token_id=self.tokenizer.pad_token_id,
                ppo_pack_seq=self.policy_config.ppo_pack_seq,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self.vocab_size,
            )

            use_linear_ce = self.training_config.use_linear_ce
            fp32_output = not use_linear_ce
            model_output = model(**fwd_kwargs, fp32_output=fp32_output)
            if isinstance(model_output, tuple):
                model_output = model_output[0]

            if use_linear_ce and mpu.is_pipeline_last_stage():
                assert isinstance(model_output, dict)
                linear_ce_output = model_output
                model_output = model_output["hidden_states"]
            else:
                if mpu.is_pipeline_last_stage():
                    assert isinstance(model_output, torch.Tensor)

            def id_func(model_output, non_loss_data=True):
                target = batch["target"]
                if use_linear_ce:
                    logprobs = logprobs_from_linear_ce(
                        linear_ce_backend=self.training_config.linear_ce_backend,
                        linear_ce_output=linear_ce_output,
                        target=target,
                        ignore_cp=True,
                        pre_shifted=True,
                    )
                else:
                    logprobs = from_parallel_logits_to_logprobs(
                        vocab_parallel_logits=model_output.float(),
                        target=target,
                        ignore_cp=True,
                        pre_shifted=True,
                    )

                # Reassemble from CP shards if local_cp_size > 1.
                lcp = batch.get("local_cp_size")
                lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
                if lcp_val > 1:
                    cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=lcp_val)
                    cp_rank = cp_group.rank()
                    total_tokens = mb_data["tokens"].size(0)
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

    @torch.no_grad()
    def compute_logprobs(
        self,
        model,
        batches_list: List[Dict[str, Any]],
        batch_log_str: str,
        compute_topk: bool = False,
        gather_target_ids_key: str = None,
    ):
        """Run a forward-only pass and gather per-sample log-probs.

        设置 top-K 相关参数时，每个 sample 返回 dict（含 3D 字段）；否则返回 2D tensor。
        """
        self.batch_iters = 0
        self.total_iters = len(batches_list) // self.forward_only_mbs
        self.batch_log_str = batch_log_str

        total_samples = len(batches_list)
        seq_length = get_batches_max_seqlen(batches_list, self.training_config.pad_to_mulitiple_of)
        seq_length = get_max_seqlen_within_ep(seq_length)
        num_microbatches = divide(total_samples, self.forward_only_mbs)
        batch_iter = get_iterator_k_split_list(batches_list, num_microbatches)

        fwd_bwd_function = get_forward_backward_func()
        fwd_results = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(
                seq_length,
                inference_only=True,
                compute_topk=compute_topk,
                gather_target_ids_key=gather_target_ids_key,
            ),
            data_iterator=batch_iter,
            model=model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=self.forward_only_mbs,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )

        if not compute_topk and gather_target_ids_key is None:
            # Label-based logps path: tensor-cat → 2D PP broadcast → per-sample chunk.
            logprobs = torch.cat(fwd_results) if len(fwd_results) > 0 else None

            # Broadcast it from last PP stage to everything else.
            logprobs = BroadcastUtils.broadcast_2d_tensor_within_pp(logprobs)
            assert logprobs.dtype == torch.float32, f'{logprobs.dtype=}'

            assert logprobs.shape[0] == total_samples
            logprobs = [logprob.squeeze(0) for logprob in logprobs.cpu().chunk(total_samples)]
            clear_memory()
            return logprobs

        # Topk path: 在 PP last stage 组装 per-sample dict，broadcast 到其他 stage。
        if mpu.is_pipeline_last_stage() and len(fwd_results) > 0:
            cat_fields: Dict[str, torch.Tensor] = {}
            for k in fwd_results[0].keys():
                cat_fields[k] = torch.cat([d[k] for d in fwd_results], dim=0)
            assert cat_fields["logprobs"].shape[0] == total_samples, (
                f'{cat_fields["logprobs"].shape[0]=} {total_samples=}'
            )
            cat_fields["logprobs"] = cat_fields["logprobs"].float().cpu()
            for k in list(cat_fields.keys()):
                if k != "logprobs":
                    cat_fields[k] = cat_fields[k].cpu()
            per_sample_list = [
                {
                    k: v[i]
                    for k, v in cat_fields.items()
                } for i in range(total_samples)
            ]
        else:
            per_sample_list = []

        per_sample_list = BroadcastUtils.broadcast_object_within_pp(per_sample_list)
        assert len(per_sample_list) == total_samples, \
            f"len(per_sample_list) expect {total_samples}, but get {len(per_sample_list)}"
        clear_memory()
        return per_sample_list

    def _smart_pad_forward_step(self, batch_iter, num_microbatches, micro_batch_size, seq_length):
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

        Returns
        -------
        list
            Forward results (logprobs) per micro-batch.
        """
        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(seq_length, inference_only=True),
            data_iterator=batch_iter,
            model=self._smart_pad_current_model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
            decoder_seq_length=seq_length,
        )
        clear_memory()
        return output_tensor

    def _smart_pad_forward_step_logits(
        self, batch_iter, num_microbatches, micro_batch_size, seq_length
    ):
        """Like _smart_pad_forward_step but writes per-sample logits into logits_cpu_buffer."""
        assert not self.training_config.use_linear_ce, (
            "fused-CE (use_linear_ce) is incompatible with the full-logits smart-pad path "
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
    ):
        """Compute logprobs using smart pad to group samples by seqlen.

        Parameters
        ----------
        model : module
        batches_list : list of dict
            Expanded per-sample dicts.
        batch_log_str : str
            Log prefix for progress.

        Returns
        -------
        list of Tensor
            Per-sample logprobs tensors on CPU.
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
            forward_step_wrapped_func=self._smart_pad_forward_step,
            dynamic_mbs_target_seqlen=dynamic_mbs_target_seqlen,
            dynamic_mbs_limit=dynamic_mbs_limit,
            update_total_iters_callback=lambda total_steps:
            setattr(self, 'total_iters', total_steps),
        )

        logprobs_list = smart_pad_helper.get_rowed_based_forward_results(is_row_based_rets=True)

        flatten_logprobs_list = []
        if mpu.is_pipeline_last_stage():
            for per_forward_step_results in logprobs_list:
                for logprob in per_forward_step_results:
                    flatten_logprobs_list.append(logprob.cpu())

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
                self.maybe_prepare_for_router_replay(batches, seq_length)

                batch, fwd_kwargs = prepare_data_func(
                    batches,
                    seq_length,
                    self.tokenizer.pad_token_id,
                    ppo_pack_seq=self.policy_config.ppo_pack_seq,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self.vocab_size,
                )

                for key in ["mask", "advantages", "prev_log_probs", "target"]:
                    assert key in batch

                use_linear_ce = self.training_config.use_linear_ce
                fp32_output = not use_linear_ce
                if self.policy_config.ppo_pack_seq and not dyn_cp:
                    assert not self.training_config.use_linear_ce, "暂不支持"
                    model_output = gptmodel_pack_foward(unwrapped_model, batch, fwd_kwargs)
                else:
                    model_output = model(**fwd_kwargs, fp32_output=fp32_output)

                if isinstance(model_output, tuple):
                    model_output = model_output[0]
                if use_linear_ce and mpu.is_pipeline_last_stage():
                    assert isinstance(model_output, dict)
                    assert model_output["hidden_states"].dtype == torch.bfloat16
                    linear_ce_output = model_output
                    model_output = model_output["hidden_states"]
                else:
                    assert isinstance(model_output, torch.Tensor)

                def loss_func(model_output):
                    mask = batch["mask"]
                    advantages = batch["advantages"]
                    prev_log_probs = batch["prev_log_probs"]
                    ref_log_probs = batch.get("ref_log_probs", None)
                    teacher_log_probs = batch.get("teacher_log_probs", None)
                    rollout_log_probs = batch.get("rollout_log_probs", None)
                    prev_topk_logprobs = batch.get("prev_topk_logprobs", None)
                    opd_topk_ids = batch.get("opd_topk_ids", None)
                    assert prev_log_probs.dtype == torch.float32
                    target = batch["target"]
                    advantages = advantages.float()
                    ignore_cp = self.policy_config.ppo_pack_seq or dyn_cp

                    if use_linear_ce:
                        curr_log_probs, scaled_entropy, per_token_entropy = (
                            logprobs_from_linear_ce(
                                linear_ce_backend=self.training_config.linear_ce_backend,
                                linear_ce_output=linear_ce_output,
                                target=target,
                                mask=mask,
                                pre_shifted=dyn_cp,
                                ignore_cp=ignore_cp,
                                return_entropy=True,
                            )
                        )
                        im_end_metrics = {}
                        curr_topk_logprobs = None
                        dumped_topk_logprobs = None
                        dumped_topk_token_ids = None
                    else:
                        parallel_logits = model_output.float()
                        parallel_logits_clone = parallel_logits.clone()

                        curr_log_probs = from_parallel_logits_to_logprobs(
                            vocab_parallel_logits=parallel_logits,
                            target=target,
                            ignore_cp=ignore_cp,
                            pre_shifted=dyn_cp,
                        )

                        # Top-K path: gather current model's log-probs at the
                        # rollout-time student top-K ids (with gradient).
                        curr_topk_logprobs = None
                        if prev_topk_logprobs is not None and opd_topk_ids is not None:
                            prev_topk_logprobs = prev_topk_logprobs.float()
                            target_topk_ids = torch.nn.functional.pad(
                                opd_topk_ids.long(), (0, 0, 0, 1), value=0
                            )
                            curr_topk_logprobs = from_parallel_logits_to_opd_topk_logprobs(
                                vocab_parallel_logits=parallel_logits_clone,
                                target_ids=target_topk_ids,
                                ignore_cp=self.policy_config.ppo_pack_seq,
                            )[:, :-1].contiguous()

                        im_end_metrics = self.get_im_end_metrics(
                            parallel_logits=parallel_logits_clone,
                            target=target,
                            response_mask=mask,
                            ignore_cp=self.policy_config.ppo_pack_seq,
                        )

                        scaled_entropy, per_token_entropy = vocab_parallel_entropy(
                            parallel_logits_clone, mask, ignore_cp=ignore_cp, pre_shifted=dyn_cp
                        )

                        dumped_topk_logprobs = None
                        dumped_topk_token_ids = None
                        if self.should_dump_metrics and self.config.training.dump_metrics_logprobs_topk > 0:
                            dumped_topk_logprobs, dumped_topk_token_ids = from_parallel_logits_to_topk_logprobs(
                                vocab_parallel_logits=parallel_logits_clone,
                                topk=self.config.training.dump_metrics_logprobs_topk,
                            )
                            dumped_topk_logprobs = dumped_topk_logprobs.to(
                                dtype=torch.bfloat16, device="cpu"
                            )
                            dumped_topk_token_ids = dumped_topk_token_ids.to(
                                dtype=torch.int32, device="cpu"
                            )

                    loss_input = PolicyLossInput(
                        advantages=advantages,
                        prev_log_probs=prev_log_probs,
                        ref_log_probs=ref_log_probs,
                        curr_log_probs=curr_log_probs,
                        response_mask=mask,
                        scaled_entropy=scaled_entropy,
                        rollout_log_probs=rollout_log_probs,
                        per_token_entropy=per_token_entropy,
                        parallel_logits=model_output if not use_linear_ce else None,
                        sample_mask=batch.get("sample_mask", None),
                        global_retention_ratio=batch.get("global_retention_ratio", None),
                        entropy_aux_figures=batch.get("entropy_aux_figures", None),
                        teacher_log_probs=teacher_log_probs,
                        dumped_topk_logprobs=dumped_topk_logprobs,
                        dumped_topk_token_ids=dumped_topk_token_ids,
                        should_dump_metrics=self.should_dump_metrics and not use_linear_ce,
                        prev_topk_logprobs=prev_topk_logprobs,
                        curr_topk_logprobs=curr_topk_logprobs,
                        cu_seqlens_padded=batch.get("cu_seqlens_padded", None),
                        local_cp_size=batch.get("local_cp_size", 1),
                    )

                    policy_loss_fn = get_policy_loss_fn(self.ppo_config.loss_func)
                    bwd_loss, metrics_dict = policy_loss_fn(self.config, loss_input)
                    metrics_dict.update(im_end_metrics)

                    if dyn_cp:
                        # Each microbatch's tokens stay sharded across its local
                        # CP group, so average the scalar report metrics over that
                        # group (gradients are handled by finalize_model_grads).
                        lcp = batch["local_cp_size"]
                        lcp_val = lcp.item() if isinstance(lcp, torch.Tensor) else int(lcp)
                        if lcp_val > 1:
                            dyn_cp_group = mpu.get_dynamic_data_context_parallel_groups(
                                group_size=lcp_val
                            )
                            for _v in metrics_dict.values():
                                if isinstance(_v, torch.Tensor) and _v.dim() == 0:
                                    torch.distributed.all_reduce(
                                        _v, group=dyn_cp_group, op=torch.distributed.ReduceOp.AVG
                                    )

                    if self.calc_per_token_loss:
                        total_tokens = mask.sum()
                        loss_sum = bwd_loss * total_tokens

                        cp_size = mpu.get_context_parallel_world_size()
                        if cp_size > 1 and not dyn_cp:
                            per_rank_tokens = total_tokens / cp_size
                        else:
                            per_rank_tokens = total_tokens

                        return (loss_sum, per_rank_tokens.to(torch.int), metrics_dict)

                    return (bwd_loss, metrics_dict)

                return model_output, loss_func

        return partial(fwd_output_and_loss_func, seq_length)

    def _broadcast_dumped_metrics(self, metrics, metrics_micro_batch, dump_metrics_keys):
        """Broadcast per-sample dump metrics from the last pipeline stage to all PP ranks."""
        if not self.should_dump_metrics:
            return
        if is_pipeline_last_stage():
            dumped_loss_fn_metrics = [
                {
                    k: sample[k][idx]
                    for k in dump_metrics_keys if k in sample and sample[k] is not None
                } for sample in metrics_micro_batch
                for idx in range(sample[dump_metrics_keys[0]].shape[0])
            ]
        else:
            dumped_loss_fn_metrics = []
        dumped_loss_fn_metrics = [dumped_loss_fn_metrics]
        torch.distributed.broadcast_object_list(
            dumped_loss_fn_metrics,
            src=get_pipeline_model_parallel_last_rank(),
            group=get_pipeline_model_parallel_group()
        )
        metrics["dumped_loss_fn_metrics"] = dumped_loss_fn_metrics[0]

    def _update_policy(self, batch: List[Dict[str, Any]], num_microbatches: int):
        policy_config = self.policy_config
        dyn_cp = self.dist_config.dynamic_context_parallel
        batch_size = len(batch)
        dynamic_num_microbatches = 0
        dynamic_mbs = 0
        dyn_cp_max_local_cp = None

        if dyn_cp:
            assert not self.should_dump_metrics, (
                "Dynamic CP is not compatible with ppo_dump_metrics_interval > 0"
            )
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

        metrics = {}
        dump_metrics_keys = [
            "curr_logprobs",
            "per_token_entropy",
            "topk_logprobs",
            "topk_token_ids",
            "ppo_ratio_unclamped",
            "is_ppo_ratio_clamped",
            "mask",
        ]
        if is_pipeline_last_stage() and len(metrics_micro_batch) > 0:
            token_level_accumulated = {}
            scalar_accumulated = {}

            for key in metrics_micro_batch[0].keys():
                if key in dump_metrics_keys:
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
                    token_level_accumulated[key] = values.sum(dim=0)
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

        aux_metrics = self._collect_aux_metrics(actual_num_microbatches)
        metrics.update(aux_metrics)
        obj_list = [metrics]
        torch.distributed.broadcast_object_list(
            obj_list, get_last_rank(cpu_group()), group=cpu_group()
        )
        metrics = obj_list[0]

        self._broadcast_dumped_metrics(metrics, metrics_micro_batch, dump_metrics_keys)

        return metrics

    # 下面是 finetune 相关才用到的函数
    def _finetune_func(
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
                input_teacher_logits=getattr(self.config.training, "enable_teacher_kl_loss", False),
                vocab_size=self.vocab_size,
            )

            fp32_output = not self.training_config.use_linear_ce
            if not self.policy_config.ppo_pack_seq:
                model_output = model(**fwd_kwargs, fp32_output=fp32_output)
            else:
                assert not self.training_config.use_linear_ce, "暂不支持"
                model_output = gptmodel_pack_foward(unwrapped_model, batch, fwd_kwargs)

            if isinstance(model_output, tuple):
                model_output = model_output[0]
            if self.training_config.use_linear_ce and mpu.is_pipeline_last_stage():
                assert isinstance(model_output, dict)
                assert model_output["hidden_states"].dtype == torch.bfloat16
            else:
                assert isinstance(model_output, torch.Tensor)

            def loss_func(model_output):
                loss_fn = get_policy_loss_fn(self.training_config.loss_func)
                batch["should_dump_metrics"] = self.should_dump_metrics
                loss_input = FinetuneLossInput(
                    logits=model_output if isinstance(model_output, torch.Tensor) else None,
                    batch=batch,
                    unwrapped_model=unwrapped_model,
                    skip_cp_loss_reduce=self.calc_per_token_loss,
                    linear_ce_input=model_output if isinstance(model_output, dict) else None,
                    cp_group=batch.get("cp_group", None),
                )
                result = loss_fn(self.config, loss_input)
                if microbatch_loss_reweight is not None:
                    # Reweight the per-microbatch loss so that, after Megatron
                    # divides by group_num_microbatches internally, the effective
                    # per-microbatch contribution is 1/total_num_microbatches.
                    result = (result[0] * microbatch_loss_reweight, ) + result[1:]
                batch.pop("should_dump_metrics", None)
                return result

            return model_output, loss_func

        return partial(fwd_output_and_loss_func, seq_length, microbatch_loss_reweight)

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
                loss_fn = get_policy_loss_fn("rm_bt")
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
        # dpo batch 是 train_gbs 的两倍，因为每个 dpo 样本会出两个数据。reward 在
        # 外部按 pair 计数（每条 batch 元素带 chosen/rejected 两个 key），在 rm_train
        # 内部才拆成 2 行，所以这里 gbs_factor 维持 1。
        gbs_factor = int(isinstance(self.config, DpoConfig)) + 1
        if training_config.check_gbs_consistency and not self.dist_config.dynamic_context_parallel:
            assert training_config.train_gbs * gbs_factor == len(
                batch
            ) * dp_size, f"{training_config.train_gbs=} {len(batch)=} {dp_size=}"

        if self.config.debug.experimental_pad_to_max_length:
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

        data_iter = get_iterator_k_split_list(batch, num_microbatches)
        if isinstance(self.config, RewardConfig):
            forward_step_func = self._rm_train_func(max_seq_length)
        else:
            forward_step_func = self._finetune_func(max_seq_length)
        fwd_bwd_function = get_forward_backward_func()
        metrics_micro_batch = fwd_bwd_function(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=max_seq_length,
            decoder_seq_length=max_seq_length,
            micro_batch_size=dynamic_mbs
            if training_config.use_dynamic_mbs else self.training_config.train_mbs,
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
                    max_seq_length, microbatch_loss_reweight=microbatch_loss_reweight
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

    def _finetune_step(
        self, batch: List[Dict[str, Any]], num_microbatches: int, forward_only: bool = False
    ):
        training_config = self.config.training

        if isinstance(self.config, RewardConfig):
            assert not self.policy_config.smart_pad_train, \
                "reward model training (v1) does not support smart_pad_train"
            assert not self.dist_config.dynamic_context_parallel, \
                "reward model training (v1) does not support dynamic_context_parallel"
            assert not training_config.use_dynamic_mbs, \
                "reward model training (v1) does not support use_dynamic_mbs"

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
        dump_metrics_keys = ["mask", "per_token_entropy", "topk_logprobs", "topk_token_ids"]
        if is_pipeline_last_stage() and len(metrics_micro_batch) > 0:
            token_level_accumulated = {}
            scalar_accumulated = {}

            for key in metrics_micro_batch[0].keys():
                if key in dump_metrics_keys:
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

        self._broadcast_dumped_metrics(metrics, metrics_micro_batch, dump_metrics_keys)

        return metrics

    def get_logits_output_only_func(self, seq_len, inference_only=True):
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
            assert isinstance(output_tensor, torch.Tensor)

            def id_func(output_tensor, non_loss_data=True):
                # TODO(@nrwu): 检查 sp 情况下，此处 output tensor shape 是否应该是 [b, s, v//tp] ？
                return output_tensor

            return output_tensor, id_func

        return partial(logits_output_only_func, seq_len)

    @torch.no_grad()
    def _compute_logits(self, model, batches_list: List[Dict[str, Any]], batch_log_str: str):
        total_samples = len(batches_list)
        self.batch_iters = 0
        self.batch_log_str = batch_log_str
        if self.config.distill.enable_data_with_alpha:
            num_w_alpha = sum(1 for data in batches_list if data.get('offpd_loss_alpha', 0) > 0)
            num_w_alpah_ceil = (
                num_w_alpha + self.forward_only_mbs - 1
            ) // self.forward_only_mbs * self.forward_only_mbs
            num_microbatches = divide(num_w_alpah_ceil, self.forward_only_mbs)
            # 前面已经把要计算 kl 的数据放前了
            calc_batches_list = batches_list[:num_w_alpah_ceil]
            self.total_iters = (num_w_alpah_ceil) // self.forward_only_mbs
        else:
            num_w_alpha = total_samples
            num_microbatches = divide(total_samples, self.forward_only_mbs)
            calc_batches_list = batches_list
            self.total_iters = total_samples // self.forward_only_mbs

        seq_length = get_batches_max_seqlen(batches_list, self.training_config.pad_to_mulitiple_of)
        seq_length = get_max_seqlen_within_dp(seq_length)
        max_seq_length = min(seq_length, self.training_config.seq_length)

        batch_iter = get_iterator_k_split_list(calc_batches_list, num_microbatches)

        target_logits_dtype = torch.bfloat16 if self.config.training.teacher_logits_dtype == "bf16" else torch.float32
        num_b_per_execution = self.config.training.num_batches_per_execution
        num_chunks = (num_microbatches + num_b_per_execution - 1) // num_b_per_execution

        fwd_bwd_function = get_forward_backward_func()

        if mpu.is_pipeline_last_stage():
            assert self.vocab_size % mpu.get_tensor_model_parallel_world_size(
            ) == 0, f"{self.vocab_size=} {mpu.get_tensor_model_parallel_world_size()=}"
            assert max_seq_length % mpu.get_context_parallel_world_size(
            ) == 0, f"{max_seq_length=} {mpu.get_context_parallel_world_size()=}"
            vacab_size_shard = self.vocab_size // mpu.get_tensor_model_parallel_world_size()
            seq_length_shard = max_seq_length // mpu.get_context_parallel_world_size()

            if self.logits_cpu_buffer is None or seq_length_shard > self.logits_cpu_buffer.shape[1]:
                self.logits_cpu_buffer = torch.empty(
                    (total_samples, seq_length_shard, vacab_size_shard),
                    dtype=target_logits_dtype,
                    device="cpu",
                    pin_memory=True
                )
            offset = 0

        for ni in range(num_chunks):
            act_gas = min(num_b_per_execution, num_microbatches - ni * num_b_per_execution)
            logits_list = fwd_bwd_function(
                forward_step_func=self.get_logits_output_only_func(
                    max_seq_length, inference_only=True
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

            logits = None
            if mpu.is_pipeline_last_stage():
                assert len(logits_list) > 0
                assert len(logits_list) * logits_list[0].shape[
                    0
                ] == act_gas * self.forward_only_mbs, f"Expected {act_gas=}  {self.forward_only_mbs=} {len(logits_list)=} {logits_list[0].shape=}"

                for logits in logits_list:
                    assert logits.shape == logits_list[0].shape
                    b = logits.shape[0]
                    cpu_slice = self.logits_cpu_buffer[offset:offset + b, :seq_length_shard]
                    cpu_slice.copy_(logits, non_blocking=True)
                    offset += b
            else:
                assert len(logits_list) == 0

        if mpu.is_pipeline_last_stage():
            torch.cuda.synchronize()
            logits = [
                lg.squeeze(0) for i, lg in
                enumerate(self.logits_cpu_buffer[:, :seq_length_shard, :].chunk(total_samples))
            ]

            assert len(
                logits
            ) == total_samples, f"Expected {total_samples} logits but got {len(logits)}"
        clear_memory()
        return logits

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
            forward_step_func=self.get_logits_output_only_func(
                seq_length, inference_only=True
            ),  # reuse get_logits_output_only_func
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
                values = gptmodel_pack_foward(unwrapped_model, batch, fwd_kwargs)

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
                fn = get_policy_loss_fn("ppo_value_loss")
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

            #TODO(hessianliu): collect data in log_moe_overload_factor
            """
            if getattr(mcore_config, 'log_moe_overload_factor', False):
                overload_log_string = get_moe_overload_factor_tracker().report(
                    iteration=iteration,
                    writer=None,
                    wandb_writer=None,
                    per_layer_logging=mcore_config.moe_per_layer_logging,
                )
            """

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
