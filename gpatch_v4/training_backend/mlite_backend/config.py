from megatron.lite.model import resolve_model_type_from_hf
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import (
    OptimizerConfig as MegatronLiteOptimizerConfig,
)
from megatron.lite.runtime.contracts.config import ParallelConfig

_DENSE_QWEN35_MODEL_TYPES = {"qwen3_5"}
_MOE_QWEN35_MODEL_TYPES = {"qwen3_5_moe"}
_WELM_MODEL_TYPES = {"welmv4_moe"}


def _config_field(config, name):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _validate_mlite_hf_config(engine) -> None:
    hf_config = getattr(engine, "hf_config", None)
    if hf_config is None:
        return

    model_type = _config_field(hf_config, "model_type")
    has_dense_type = model_type in _DENSE_QWEN35_MODEL_TYPES
    has_moe_type = model_type in _MOE_QWEN35_MODEL_TYPES
    has_welm_type = model_type in _WELM_MODEL_TYPES
    model_arch = engine.policy_config.model_arch
    if model_arch == "qwen3_5" and (has_moe_type or has_welm_type):
        raise ValueError(f"policy.model_arch='qwen3_5' does not match HF model_type={model_type!r}")
    if model_arch == "qwen3_5_moe" and (has_dense_type or has_welm_type):
        raise ValueError(
            f"policy.model_arch='qwen3_5_moe' does not match HF model_type={model_type!r}"
        )
    if model_arch == "welmv4_moe" and (not has_welm_type):
        raise ValueError(
            "policy.model_arch='welmv4_moe' requires "
            f"HF model_type='welmv4_moe', got {model_type!r}"
        )


def _mount_vision_model(engine) -> bool:
    hf_config = getattr(engine, "hf_config", None)
    if hf_config is None:
        return False
    return _config_field(hf_config, "vision_config") is not None


def validate_mlite_config(engine) -> None:
    training = engine.training_config
    policy = engine.policy_config
    checkpoint = engine.checkpoint_config
    dist_config = engine.dist_config

    if training.training_backend != "mlite":
        raise ValueError("MliteEngine requires training_backend='mlite'")
    if training.loss_func != "cross_entropy":
        raise NotImplementedError("mlite finetune supports cross_entropy only")
    if policy.model_arch not in ("qwen3_5", "qwen3_5_moe", "welmv4_moe"):
        raise NotImplementedError(
            "mlite finetune currently supports qwen3_5, qwen3_5_moe, "
            "and welmv4_moe only"
        )
    is_welm = policy.model_arch == "welmv4_moe"
    if policy.model_arch == "qwen3_5" and dist_config.expert_model_parallel_size != 1:
        raise NotImplementedError("mlite qwen3_5 dense requires ep=1")
    if not policy.without_ref:
        raise NotImplementedError("mlite finetune requires policy.without_ref=True")
    if policy.without_optim:
        raise NotImplementedError("mlite finetune requires an optimizer")
    if dist_config.tensor_model_parallel_size != 1:
        raise NotImplementedError("mlite 建议 tp=1")
    if dist_config.expert_tensor_parallel_size != 1:
        raise NotImplementedError("mlite finetune currently requires etp=1")
    if is_welm and dist_config.pipeline_model_parallel_size != 1:
        raise NotImplementedError("mlite WELM finetune currently requires pp=1")
    if dist_config.dynamic_context_parallel:
        if dist_config.max_seqlen_per_dp_cp_rank is None:
            raise ValueError("mlite dynamic CP requires dist_config.max_seqlen_per_dp_cp_rank")
        if dist_config.dynamic_cp_scheduler_type != "default":
            raise NotImplementedError(
                "mlite dynamic CP only supports dynamic_cp_scheduler_type='default'"
            )
        min_cp = dist_config.min_dynamic_context_parallel_size
        if min_cp < 1 or (min_cp & (min_cp - 1)) != 0:
            raise ValueError(
                "mlite dynamic CP requires min_dynamic_context_parallel_size "
                f"to be a positive power of two, got {min_cp}"
            )
    if dist_config.virtual_pipeline_model_parallel_size not in (None, 1):
        raise NotImplementedError("mlite finetune does not support virtual PP")
    if training.use_dynamic_mbs or any(
        value is not None for value in (
            policy.dynamic_mbs_target_seqlen,
            policy.dynamic_mbs_limit,
            policy.dynamic_mbs_target_seqlen_fwd_only,
            policy.dynamic_mbs_limit_fwd_only,
        )
    ):
        raise NotImplementedError("mlite finetune does not support dynamic microbatches")
    if policy.smart_pad_train or policy.smart_pad_infer:
        raise NotImplementedError("mlite finetune does not support smart padding")
    if policy.lora.enabled():
        raise NotImplementedError("mlite finetune does not support LoRA")
    if training.enable_mtp or training.online_mtp_sft:
        raise NotImplementedError("mlite finetune does not support MTP")
    if training.moe_router_replay:
        raise NotImplementedError("mlite finetune does not support router replay")
    if training.ppo_dump_metrics_interval > 0:
        raise NotImplementedError("mlite finetune does not support metric dumps")
    if policy.override_transformer_config:
        raise NotImplementedError("mlite finetune does not accept mcore transformer overrides")
    if is_welm:
        if checkpoint.use_dist_checkpointing is not False:
            raise NotImplementedError(
                "mlite WELM finetune currently requires rank-local checkpointing"
            )
        if checkpoint.no_load_optim or checkpoint.no_save_optim:
            raise NotImplementedError(
                "mlite WELM rank-local resume requires optimizer state load and save"
            )
        if (checkpoint.save_total_limit is not None or checkpoint.save_retain_interval is not None):
            raise NotImplementedError("mlite WELM checkpoint retention is not implemented")
        if checkpoint.async_save:
            raise NotImplementedError("mlite WELM rank-local checkpointing is synchronous")
    elif checkpoint.use_dist_checkpointing is not True:
        raise NotImplementedError("mlite finetune requires distributed checkpointing")
    if checkpoint.convert_mcore_to_hf_offline or checkpoint.skip_save_mcore_model:
        raise NotImplementedError(
            "mlite checkpoint does not support offline HF conversion or "
            "skipping training-checkpoint model weights"
        )


def resolve_model_name(hf_config) -> str:
    return resolve_model_type_from_hf(hf_config)


def _build_runtime_plugins(dist_config) -> dict:
    if not dist_config.dynamic_context_parallel:
        return {}
    return {
        "dynamic_context_parallel":
            {
                "enabled": True,
                "max_seqlen_per_dp_cp_rank": dist_config.max_seqlen_per_dp_cp_rank,
                "min_context_parallel_size": dist_config.min_dynamic_context_parallel_size,
                "require_full_cp_size_coverage": False,
            },
    }


def build_mlite_config(engine) -> MegatronLiteConfig:
    _validate_mlite_hf_config(engine)
    dist_config = engine.dist_config
    attention_backend = engine.training_config.attention_backend
    ep_backend = engine.policy_config.ep_backend
    if ep_backend not in ("eager", "deepep"):
        raise ValueError(f"mlite ep_backend must be 'eager' or 'deepep', got {ep_backend!r}")
    impl_cfg = {
        "optimizer": "fsdp2",
        # "optimizer": "dist_opt",
        "use_thd": True,
        "use_deepep": ep_backend == "deepep",
        "cross_entropy_fusion": engine.training_config.use_linear_ce,
        "recompute": ["full"] if engine.training_config.recompute else [],
        "mtp_enable": False,
        "mtp_enable_train": False,
        "mount_vision_model": _mount_vision_model(engine),
        "freeze_vision_model": engine.training_config.freeze_vit,
        "freeze_vision_projector": engine.training_config.freeze_projector,
    }  #TODO：参数化控制，现在写的比较粗暴
    runtime_plugins = _build_runtime_plugins(dist_config)
    if runtime_plugins:
        impl_cfg["runtime_plugins"] = runtime_plugins
    if dist_config.dynamic_context_parallel:
        impl_cfg["cp_attention_backend"] = "all_gather"
    return MegatronLiteConfig(
        model_name=resolve_model_name(engine.hf_config),
        impl="lite",
        hf_path=engine.policy_config.hf_model_path,
        parallel=ParallelConfig(
            tp=dist_config.tensor_model_parallel_size,
            etp=dist_config.expert_tensor_parallel_size,
            ep=dist_config.expert_model_parallel_size,
            pp=dist_config.pipeline_model_parallel_size,
            vpp=dist_config.virtual_pipeline_model_parallel_size or 1,
            cp=dist_config.context_parallel_size,
        ),
        optimizer=build_mlite_optimizer_config(engine),
        attention_backend_override=(None if attention_backend == "auto" else attention_backend),
        router_aux_loss_coef=0.0,
        load_hf_weights=True,
        impl_cfg=impl_cfg,
    )


def build_mlite_optimizer_config(engine) -> MegatronLiteOptimizerConfig:
    optimizer = engine.config.optimizer
    override = optimizer.override_optimizer_config or {}
    is_welm = engine.policy_config.model_arch == "welmv4_moe"
    if is_welm:
        supported_overrides = {"adam_eps", "eps"}
    else:
        supported_overrides = {
            "adam_eps",
            "decoupled_weight_decay",
            "eps",
            "offload_fraction",
            "optimizer_cpu_offload",
            "optimizer_offload_fraction",
            "use_precision_aware_optimizer",
        }
    unsupported_overrides = set(override) - supported_overrides
    if unsupported_overrides:
        if is_welm:
            raise NotImplementedError(
                "mlite WELM keeps optimizer state on GPU and does not support "
                f"optimizer overrides: {sorted(unsupported_overrides)}"
            )
        raise NotImplementedError(
            "mlite does not support optimizer overrides: "
            f"{sorted(unsupported_overrides)}"
        )
    offload_fraction = None
    if not is_welm:
        offload_fraction = override.get(
            "offload_fraction",
            override.get("optimizer_offload_fraction"),
        )
        if offload_fraction is None and override.get("optimizer_cpu_offload"):
            offload_fraction = 1.0
    return MegatronLiteOptimizerConfig(
        optimizer=normalize_optimizer_name(optimizer.optimizer_type),
        lr=optimizer.lr,
        min_lr=optimizer.min_lr,
        clip_grad=optimizer.max_grad_norm,
        weight_decay=optimizer.weight_decay,
        lr_warmup_steps_ratio=optimizer.lr_warmup_step_frac,
        total_training_steps=engine.training_config.total_training_step or 0,
        lr_warmup_steps=optimizer.lr_warmup_steps,
        lr_warmup_init=optimizer.lr_warmup_init,
        lr_decay_steps=optimizer.lr_decay_steps,
        lr_decay_style=optimizer.lr_decay_style.replace("_", "-"),
        weight_decay_incr_style=optimizer.weight_decay_incr_style,
        lr_wsd_decay_style=optimizer.lr_wsd_decay_style,
        lr_wsd_decay_steps=optimizer.lr_wsd_decay_steps,
        use_checkpoint_opt_param_scheduler=optimizer.use_checkpoint_opt_param_scheduler,
        adam_beta1=optimizer.adam_beta1,
        adam_beta2=optimizer.adam_beta2,
        adam_eps=override.get(
            "adam_eps",
            override.get("eps", optimizer.adam_epsilon),
        ),
        offload_fraction=offload_fraction,
        use_precision_aware_optimizer=(
            None if is_welm else override.get("use_precision_aware_optimizer")
        ),
        decoupled_weight_decay=(None if is_welm else override.get("decoupled_weight_decay")),
    )


def normalize_optimizer_name(name: str) -> str:
    lower = str(name).lower()
    if "adam" in lower:
        return "adam"
    raise ValueError(f"MliteEngine only supports Adam-style optimizers, got {name!r}")
