import collections
import math
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Union

from gpatch_v4.configs.self_heal_config import GeminiSelfHealConfig
from gpatch_v4.configs.utils import MappingProtocol


def _hf_text_config_for_moe_router_replay(hf_config: Any) -> Any:
    # Omni nests text under thinker_config; VL uses text_config; LLM is top-level.
    if hasattr(hf_config, "thinker_config") and hasattr(hf_config.thinker_config, "text_config"):
        return hf_config.thinker_config.text_config
    return getattr(hf_config, "text_config", hf_config)


@dataclass
class TrainingConfig(MappingProtocol):
    """Base training configuration shared by all trainers.

    Attributes
    ----------
    training_backend : str
        ``"mcore"`` (Megatron-Core) or ``"fsdp2"`` (PyTorch FSDP2).
    train_gbs : int or None
        ``None`` → auto from ``train_mbs`` and world size.
    train_mbs : int
        Train micro batch size per GPU.
    gradient_accumulation_steps : int or None
        a.k.a. ``train_gas``; ``None`` → auto.
    exit_step : int or None
        1-indexed; ``-1`` disables early exit.
    seed : int
    seq_length : int
        Maximum sequence length.
    recompute : bool
        Activation recomputation (gradient checkpointing).
    total_training_step : int or None
        ``None`` → derived from epochs and data size.
    total_eval_step : int
        ``0`` → no evaluation.
    num_train_epoches : int
    apply_deterministic_mode : bool
    disable_flash_attn_3 : bool
        Force TE to skip FA3 (falls back to FA2). Deterministic mode +
        ``attention_backend=flash`` sets this automatically.
    disable_flash_attn_4 : bool
        Force TE to skip FA4 (falls back to FA2). RTX PRO 5000 (sm_120)
        sets this automatically; FA4 packed-THD training is unstable there.
    use_fast_tokenizer : bool
        Use the HF fast tokenizer.
    save_interval : int
        Save every ``n`` steps (ppo_step in RL training).
    eval_interval : int
        Evaluate every ``n`` steps; ``0`` disables.
    eval_before_train : bool
    eval_rollout_gbs : int
    eval_rollout_mbs : int
    pad_to_mulitiple_of : int
        Pad token sequences to a multiple of this value.
    data_parallel_random_init : bool
    comput_attn_mask : bool
        Whether to compute attention mask explicitly.
    auto_load_from_save_ckpt : bool
        Auto resume from latest saved checkpoint.
    allow_tf32 : bool
        ``torch.backends.cuda.matmul.allow_tf32``.
    use_torch_autocast : bool
        Deprecated.
    max_train_step_waiting_time : float or None
        Wall-clock seconds; ``None`` → no limit.
    max_restart_attempts : int or None
        ``0`` → no restart, launch once.
    skip_train_step : bool or None
        Debug switch.
    attention_backend : str or None
        ``"flash"`` / ``"fused"`` / ``"unfused"`` / ``"local"`` / ``"auto"``.
    build_from_mbridge : bool
        ``True`` → mbridge; ``False`` → megatron_bridge.
    moe_pad_with_random_tokens : bool
    freeze_moe_router : bool
    freeze_moe_shared_experts : bool
    freeze_llm : bool
    freeze_vit : bool
    freeze_projector : bool
    freeze_audio : bool
    freeze_qformer : bool
    moe_token_dispatcher_type : str
        ``"allgather"`` / ``"alltoall"`` / ``"flex"``.
    moe_router_load_balancing_type : str
        ``"none"`` disables load balancing.
    moe_balance_loss_coef : float
        FSDP2 switch-style MoE load-balancing loss coefficient; ``0`` disables.
    freeze_router_weight : bool
        freeze MoE router weight (``requires_grad=False``).
    freeze_csa_indexer : bool
        ``True`` (default) freezes DeepSeek-V4 CSA Lightning Indexer params
        (``requires_grad=False``) so AdamW/Muon weight decay cannot shrink
        them when there is no indexer KL loss / top-k gradient.
    freeze_router_correction_bias : bool
        ``True`` (default) freezes ``e_score_correction_bias``;
        ``False`` enables the per-step loss-free load-balancing update.
    router_correction_bias_update_speed : float
        per-step magnitude of the ``e_score_correction_bias``
        update (see https://arxiv.org/abs/2408.15664); active only when
        ``freeze_router_correction_bias=False``; default ``1e-3``.
    router_correction_bias_use_abs_update : bool
        use absolute update for ``e_score_correction_bias``
        as the paper does; otherwise, use the log-based update.
        default ``True``.
    offload_process_group : bool
    use_dynamic_mbs : bool
        Requires ``train_mbs == 1``.
    ppo_dump_metrics_interval : int
        Dump every ``n`` PPO steps; ``-1`` disables. Not supported when
        ``DistConfig.dynamic_context_parallel=True``.
    ppo_dump_metrics_dir : str
    ppo_dump_per_token_entropy : bool
    dump_metrics_logprobs_topk : int
        Dump top-k logprobs ``[b, s, topk]``; ``0`` disables.
    ppo_dump_gradient : bool
        When dumping metrics, also dump ``∂bwd_loss/∂actor_loss`` and
        ``∂bwd_loss/∂curr_log_probs`` (new loss path only). Default ``False``.
    im_end_metrics_enable : bool
        Compute EOS probability/top-k diagnostics during actor train forward.
    ppo_dump_moe_topk : int
        Dump top-k MoE experts ``[b, s, topk]``; ``0`` disables.
    check_gbs_consistency : bool
        Assert actual batch size equals ``train_gbs * dp_size`` before each step.
    """
    # TODO add training dtype
    training_backend: str = field(
        default="mcore",
        metadata={"help": "Strategy to use for training."},
    )
    train_gbs: Optional[int] = field(default=None, metadata={"help": "Train global batch size."})
    train_mbs: int = field(default=1, metadata={"help": "Train micro batch size."})
    # TODO rename it train_gas later
    gradient_accumulation_steps: Optional[int] = field(
        default=None, metadata={"help": "train_gas, number of gradient accumulation steps."}
    )
    exit_step: Optional[int] = field(
        default=-1, metadata={"help": "Whether to exit at a specific step, start from 1"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    seq_length: int = field(
        default=2048,
        metadata={"help": "Sequence length."},
    )
    recompute: bool = field(
        default=False,
        metadata={"help": "Whether to recompute."},
    )
    total_training_step: Optional[int] = field(
        default=None, metadata={"help": "Total training steps."}
    )
    total_eval_step: int = field(default=0, metadata={"help": "Total eval steps."})
    num_train_epoches: Union[int, float] = field(
        default=1,
        metadata={"help": "Number of epochs for training."},
    )
    apply_deterministic_mode: bool = field(
        default=False,
        metadata={"help": "Whether to apply deterministic mode."},
    )
    disable_flash_attn_3: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable TransformerEngine FlashAttention 3 (fall back to FA2). "
                "Set automatically when apply_deterministic_mode and "
                "attention_backend=flash."
            ),
        },
    )
    disable_flash_attn_4: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable TransformerEngine FlashAttention 4 (fall back to FA2). "
                "Set automatically on NVIDIA RTX PRO 5000."
            ),
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use fast tokenizer."},
    )
    save_interval: int = field(
        default=100,
        metadata={"help": "Whether to save model at each n steps (ppo_step when training rl)"},
    )
    eval_interval: int = field(
        default=0,
        metadata={"help": "Whether to eval model at each n steps (ppo_step when training rl)"},
    )
    eval_rollout_gbs: Optional[int] = field(
        default=None,
        metadata={"help": "Global batch size for evaluation rollouts."},
    )
    eval_rollout_mbs: Optional[int] = field(
        default=None,
        metadata={"help": "Micro batch size for evaluation rollouts."},
    )
    eval_before_train: bool = field(
        default=False,
        metadata={"help": "Whether eval before train"},
    )
    pad_to_mulitiple_of: int = field(default=512, metadata={"help": "Pad to multiple of."})
    data_parallel_random_init: bool = field(
        default=False,
        metadata={"help": "Whether to use data parallel random init."},
    )
    comput_attn_mask: bool = field(
        default=True,
        metadata={"help": "Whether to compute attention mask."},
    )
    auto_load_from_save_ckpt: bool = field(
        default=False, metadata={"help": "Whether to auto load from save ckpt"}
    )
    allow_tf32: bool = field(
        default=False,
        metadata={"help": "torch.backends.cuda.matmul.allow_tf32"},
    )
    use_torch_autocast: bool = field(
        default=False,
        metadata={"help": "use torch autocast, which is deprecated"},
    )
    max_train_step_waiting_time: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum every train step waiting time. None means no timelimit."}
    )
    max_restart_attempts: Optional[int] = field(
        default=0,
        metadata={"help": "Maximum restart attempts. 0 means no restart, only launch once"}
    )
    node_replacer_cls: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Fully qualified class path of a NodeReplacer implementation "
                "(e.g. 'gpatch_v4.orches.node_replacer.MockNodeReplacer'). "
                "None means no node replacement on failure — just restart in place."
        }
    )
    enable_self_heal: bool = field(
        default=False,
        metadata={"help": "Enable GPU fault self-healing (auto node replacement + restart)."},
    )
    self_heal_config: Optional[GeminiSelfHealConfig] = field(
        default=None,
        metadata={"help": "GeminiSelfHealConfig overrides (recovery_max_wait, ray_port, etc.)."},
    )
    skip_train_step: Optional[bool] = field(
        default=False, metadata={"help": "Whether to skip the train step."}
    )
    attention_backend: Optional[str] = field(
        default="auto", metadata={"help": "Attention backend to use"}
    )
    build_from_mbridge: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to build from vanilla bridge. If True, build from mbridge otherwise build from megatron_bridge"
        }
    )
    moe_pad_with_random_tokens: bool = field(
        default=False, metadata={"help": "Whether to pad with random tokens"}
    )
    # moe freeze, 将这些挪出去
    freeze_moe_router: bool = field(
        default=False, metadata={"help": "Whether to freeze MoE routers"}
    )
    freeze_moe_shared_experts: bool = field(
        default=False, metadata={"help": "Whether to freeze MoE shared experts"}
    )
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze language model"})
    freeze_vit: bool = field(default=False, metadata={"help": "Whether to freeze vision model"})
    freeze_projector: bool = field(
        default=False, metadata={"help": "Whether to freeze vision projection module"}
    )
    freeze_audio: bool = field(default=False, metadata={"help": "Whether to freeze audio model"})
    freeze_qformer: bool = field(
        default=False, metadata={"help": "Whether to freeze audio qformer module"}
    )
    moe_token_dispatcher_type: str = field(
        default="alltoall", metadata={"help": "Token dispatcher type"}
    )
    moe_router_load_balancing_type: str = field(
        default="none", metadata={"help": "Router load balancing type"}
    )
    moe_balance_loss_coef: float = field(
        default=0.0,
        metadata={"help": "FSDP2 switch-style MoE load-balancing loss coefficient; 0 disables."},
    )
    freeze_router_weight: bool = field(
        default=False,
        metadata={"help": "freeze MoE router weight (requires_grad=False)."},
    )
    freeze_csa_indexer: bool = field(
        default=True,
        metadata={
            "help":
                "True (default) freezes DeepSeek-V4 CSA Lightning Indexer params "
                "(requires_grad=False) so weight decay cannot shrink them when "
                "there is no indexer KL / top-k gradient. Set False to keep them "
                "trainable (e.g. after wiring indexer KL loss)."
        },
    )
    freeze_router_correction_bias: bool = field(
        default=True,
        metadata={
            "help":
                "True (default) freezes e_score_correction_bias; "
                "False enables the per-step loss-free load-balancing bias update."
        },
    )
    router_correction_bias_update_speed: float = field(
        default=1e-3,
        metadata={
            "help":
                "per-step e_score_correction_bias update magnitude "
                "(see https://arxiv.org/abs/2408.15664); active only when freeze_router_correction_bias=False."
        },
    )
    router_correction_bias_use_abs_update: bool = field(
        default=True,
        metadata={
            "help":
                "use absolute update for ``e_score_correction_bias`` "
                "as the paper does; otherwise, use the log-based update."
        },
    )

    offload_process_group: bool = field(
        default=False, metadata={"help": "Whether to offload process group"}
    )

    use_dynamic_mbs: bool = field(
        default=False, metadata={"help": "use dynamic mbs, train_mbs must be 1"}
    )

    # dump metrics
    ppo_dump_metrics_interval: int = field(
        default=-1,
        metadata={
            "help":
                "Dump train metrics every n ppo steps. GRPO + dyn-CP reverse-reroutes "
                "1D per-token fields; SFT dump, MoE top-k, and vocab top-k with dyn-CP "
                "are unsupported."
        }
    )
    ppo_dump_metrics_dir: str = field(
        default="", metadata={"help": "directory to save dumpped metrics"}
    )
    ppo_dump_per_token_entropy: bool = field(
        default=False, metadata={"help": "Whether to dump per-token entropy"}
    )

    dump_metrics_logprobs_topk: int = field(
        default=0, metadata={"help": "Dump topk logits[b,s,v], choose topk ->[b,s,topk]"}
    )
    ppo_dump_gradient: bool = field(
        default=False,
        metadata={
            "help":
                "When dumping metrics (ppo_dump_metrics_interval > 0), also dump "
                "token-level ∂bwd_loss/∂actor_loss and ∂bwd_loss/∂curr_log_probs. "
                "Only effective with ppo.use_legacy_loss=False."
        },
    )
    im_end_metrics_enable: bool = field(
        default=False,
        metadata={"help": "Report EOS probability/top-k diagnostics from actor train forward."}
    )
    ppo_dump_moe_topk: int = field(
        default=0,
        metadata={"help": "Dump topk moe experts[b,s,topk], 0 means disable dump moe topk"}
    )
    enable_thinking: bool = field(default=False, metadata={"help": "Whether to enable thinking"})
    preserve_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to preserve <think>...</think> in historical assistant turns"},
    )
    ignore_thinking_flag: bool = field(
        default=False, metadata={"help": "Whether to ignore thinking, means no set it"}
    )
    moe_router_replay: bool = field(default=False, metadata={"help": "Whether to enable r3."})
    moe_router_replay_num_layers: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Full sampler layer count used by MoE router replay. The parent "
                "config resolves it during initialization when omitted."
        },
    )
    moe_router_replay_topk: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Experts selected per token for MoE router replay. The parent config "
                "resolves it from the HF config during initialization when omitted."
        },
    )
    calc_mfu_freq: Optional[int] = field(
        default=None, metadata={"help": "Calculate mfu frequency."}
    )
    image_grid_thw_name: str = field(
        default="image_grid_thw",
        metadata={"help": "The name of image_grid_thw in the batch data."}
    )
    check_gbs_consistency: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to assert that the actual batch size equals "
                "train_gbs * dp_size before each training step."
        },
    )
    enable_mtp: bool = field(default=False, metadata={"help": "Whether to build mtp model."})
    enable_dspark: bool = field(
        default=False,
        metadata={
            "help":
                "Load and save DeepSeek-V4 DSpark draft weights stored under mtp.*. "
                "Training is gated by online_train_dspark."
        },
    )
    online_train_dspark: bool = field(
        default=False,
        metadata={
            "help": "Whether to train DSpark (prepare + loss). "
                    "Requires enable_dspark=True."
        },
    )
    dspark_num_anchors: Optional[int] = field(
        default=None,
        metadata={"help": "Sampled DSpark anchor blocks per sequence."},
    )
    dspark_ce_loss_alpha: float = field(default=0.1)
    dspark_l1_loss_alpha: float = field(default=0.9)
    dspark_confidence_loss_alpha: float = field(default=1.0)
    dspark_loss_scaling_factor: float = field(default=1.0)
    dspark_loss_decay_gamma: float = field(default=4.0)
    online_mtp_sft: bool = field(
        default=False,
        metadata={"help": "Whether to tune the mtp layers (during rl or sft)."},
    )
    mtp_loss_scaling_factor: float = field(
        default=0.1,
        metadata={"help": "Scaling factor for auxiliary MTP loss."},
    )
    docker_image_tag: str = field(default="", metadata={"help": "Docker image tag."})
    use_linear_ce: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to use linear cross entropy loss. cross_entropy_loss_fusion=True and cross_entropy_fusion_impl=linear"
        }
    )
    linear_ce_backend: str = field(
        default="separate",
        metadata={
            "help":
                "Backward method for linear cross entropy. Options: "
                "'separate' (full d_logits buffer, native dtype, fastest & best memory, default), "
                "'fuse_mn' (single fused kernel, no intermediate d_logits but needs fp32 d_hidden/d_weight), "
                "'split_n' (split d_logits along vocab dim, loop over splits, slowest)."
        },
    )
    ce_compaction: bool = field(
        default=False,
        metadata={"help": "Compact masked tokens before MCore output projection and CE."},
    )

    def resolve_moe_router_replay_shape(
        self,
        hf_model_path: str,
        model_override_args: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple[int, int]]:
        """Fill the MoE router replay layout before actor creation."""
        if not self.moe_router_replay:
            return None

        num_layers = self.moe_router_replay_num_layers
        moe_router_topk = self.moe_router_replay_topk
        assert (num_layers is None) == (moe_router_topk is None), (
            "moe_router_replay_num_layers and moe_router_replay_topk must be "
            "configured together"
        )

        if num_layers is None:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                hf_model_path,
                trust_remote_code=True,
            )
            hf_text_config = _hf_text_config_for_moe_router_replay(hf_config)
            override_num_layers = (model_override_args or {}).get("num_hidden_layers", None)
            num_layers = (
                int(override_num_layers)
                if override_num_layers is not None else int(hf_text_config.num_hidden_layers)
            )
            moe_router_topk = getattr(hf_text_config, "num_experts_per_tok", None)
            assert moe_router_topk is not None, (
                "moe_router_replay requires num_experts_per_tok on the policy "
                "HF config or its text_config"
            )

        num_layers = int(num_layers)
        moe_router_topk = int(moe_router_topk)
        assert num_layers > 0 and moe_router_topk > 0, (
            f"invalid MoE router replay shape: {(num_layers, moe_router_topk)}"
        )
        self.moe_router_replay_num_layers = num_layers
        self.moe_router_replay_topk = moe_router_topk
        return num_layers, moe_router_topk

    @property
    def return_hidden_states_for_ce(self) -> bool:
        return self.use_linear_ce or self.ce_compaction

    def __post_init__(self):
        assert self.training_backend in ["mcore", "fsdp2", "mlite"]
        assert self.attention_backend in ["flash", "fused", "unfused", "local", "auto"]
        assert self.moe_token_dispatcher_type in ['allgather', 'alltoall', 'flex']
        assert not (self.enable_mtp and
                    self.enable_dspark), ("enable_mtp and enable_dspark are mutually exclusive")
        if self.enable_dspark:
            assert self.training_backend == "fsdp2"
        if self.online_train_dspark:
            assert self.enable_dspark, ("online_train_dspark requires enable_dspark=True")
            assert self.loss_func == "cross_entropy"
            assert self.dspark_num_anchors is not None and self.dspark_num_anchors > 0
            assert self.dspark_loss_decay_gamma > 0
            assert min(
                self.dspark_ce_loss_alpha,
                self.dspark_l1_loss_alpha,
                self.dspark_confidence_loss_alpha,
                self.dspark_loss_scaling_factor,
            ) >= 0
            assert not self.use_linear_ce, (
                "DSpark requires dense logits for Markov, L1, and confidence losses"
            )
            assert self.freeze_router_correction_bias, (
                "DSpark P0 does not update router correction-bias buffers"
            )
        assert self.linear_ce_backend in ["fuse_mn", "separate", "split_n"], (
            f"Unknown linear_ce_backend: '{self.linear_ce_backend}'. "
            f"Choose from: ['fuse_mn', 'separate', 'split_n']"
        )
        if self.return_hidden_states_for_ce:
            assert self.dump_metrics_logprobs_topk == 0, (
                "use_linear_ce/ce_compaction is incompatible with "
                "dump_metrics_logprobs_topk: Linear CE and compact CE do not materialize "
                "dense logits"
            )
            assert not self.im_end_metrics_enable, (
                "use_linear_ce/ce_compaction is incompatible with "
                "im_end_metrics_enable: Linear CE and compact CE have no dense logits"
            )
        if self.ppo_dump_metrics_interval > 0:
            assert self.ppo_dump_metrics_dir
        if self.eval_before_train:
            assert self.eval_interval > 0
        if self.use_dynamic_mbs:
            assert self.train_mbs == 1
        if self.eval_rollout_gbs is not None:
            assert self.eval_rollout_gbs > 0
            assert self.eval_rollout_mbs is not None
            assert self.eval_rollout_mbs > 0
        else:
            assert self.eval_rollout_mbs is None
        if self.node_replacer_cls is not None:
            assert self.max_restart_attempts > 0, (
                "node_replacer_cls is set but max_restart_attempts=0 — "
                "the replacer will never be invoked"
            )

        if self.enable_self_heal:
            import os
            assert os.environ.get("__SYS_TASK_RUNTIME_ID__"), (
                "enable_self_heal=True but not running on Gemini platform "
                "(__SYS_TASK_RUNTIME_ID__ missing)"
            )
            assert os.environ.get("__SYS_JOB_INSTANCE_SIGNATURE__"), (
                "enable_self_heal=True but not running on Gemini platform "
                "(__SYS_JOB_INSTANCE_SIGNATURE__ missing)"
            )
            self.node_replacer_cls = ("gpatch_v4.orches.gemini_self_heal.GeminiNodeReplacer")
            self.auto_load_from_save_ckpt = True
            if (self.max_restart_attempts or 0) == 0:
                self.max_restart_attempts = 3
            if self.self_heal_config is None:
                self.self_heal_config = GeminiSelfHealConfig()

        try:
            with open("/home/wepsdl/gcore_image_version.txt", "r") as f:
                self.docker_image_tag = f.read().strip()
        except:
            pass

    def chat_template_thinking_kwargs(self) -> dict:
        """Build thinking-related kwargs for ``apply_chat_template``.

        Returns
        -------
        dict
            Empty when ``ignore_thinking_flag`` is True (omit the arg, like
            verl); otherwise ``{"enable_thinking": <bool>}``.
        """
        if self.ignore_thinking_flag:
            return {}
        return {"enable_thinking": self.enable_thinking}


@dataclass
class FinetuneTrainingConfig(TrainingConfig):
    """Training configuration for supervised fine-tuning, extends :class:`TrainingConfig`.

    Attributes
    ----------
    train_step_per_epoch : int or None
        Number of training steps per epoch. ``None`` means determined by data size.
    prefetch_num_gb : int or None
        Number of global batches to prefetch.
    loss_func : str
        Loss function to use. One of ``"cross_entropy"``, ``"ce_with_kl"``, ``"custom"``,
        ``"dpo"``, ``"square_averaging_cross_entropy"``.
    loss_func_py_path : str or None
        Path to the loss function.
    loss_func_py_name : str or None
        Name of the loss function.
    cross_entropy_loss_fusion : bool
        Whether to fuse cross entropy loss computation for performance.
    cross_entropy_fusion_impl : str
        Implementation of cross entropy fusion: ``"native"`` or ``"te"`` (TransformerEngine).
    sort_batched : bool
        Whether to sort samples within a batch by length for packing efficiency.
    forward_clear_memory : bool
        Whether to periodically clear memory during forward-only passes.
    forward_clear_memory_interval : int
        Interval (in microbatches) to clear memory during forward-only passes.
    manual_gc : bool
        Disable automatic cyclic collection and collect at fixed training-step intervals.
    manual_gc_interval : int
        Number of completed training steps between manual collections.
    """
    train_step_per_epoch: Optional[int] = field(
        default=None, metadata={"help": "Number of steps per epoch."}
    )
    prefetch_num_gb: Optional[int] = field(
        default=1, metadata={"help": "Number of prefetch steps."}
    )
    loss_func: str = field(default="cross_entropy", metadata={"help": "Loss function."})
    loss_func_py_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the loss function."}
    )
    loss_func_py_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the loss function."}
    )
    cross_entropy_loss_fusion: bool = field(
        default=False, metadata={"help": "Whether to fuse cross entropy loss."}
    )
    cross_entropy_fusion_impl: str = field(
        default="native", metadata={"help": "Implementation of cross entropy fusion."}
    )
    sort_batched: bool = field(default=False, metadata={"help": "Whether to sort the batched."})
    forward_clear_memory: bool = field(
        default=False,
        metadata={"help": "Whether to periodically clear memory during forward-only passes."}
    )
    forward_clear_memory_interval: int = field(
        default=4,
        metadata={"help": "Interval (in microbatches) to clear memory during forward-only passes."}
    )
    manual_gc: bool = field(
        default=False,
        metadata={"help": "Use fixed-interval Python cyclic garbage collection."},
    )
    manual_gc_interval: int = field(
        default=20,
        metadata={"help": "Training steps between manual garbage collections."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.manual_gc and self.manual_gc_interval <= 0:
            raise ValueError("manual_gc_interval must be positive when manual_gc=True.")

        assert self.loss_func in [
            "cross_entropy", "ce_with_kl", "custom", "dpo", "rm_bt",
            "square_averaging_cross_entropy", "grad_cache_loss"
        ], f"Invalid loss function: {self.loss_func}"
        if self.loss_func == "custom":
            assert self.loss_func_py_path is not None and self.loss_func_py_name is not None, "Custom loss function must be provided"
        if self.ce_compaction:
            # TODO: support FSDP2 (SFT/GRPO/DSV4) via lm_head bypass gather before kernel.
            assert self.training_backend == "mcore", (
                "ce_compaction currently supports MCore SFT only"
            )
            assert not self.apply_deterministic_mode, (
                "ce_compaction has not been validated with apply_deterministic_mode=True"
            )
            assert self.loss_func in ["cross_entropy", "square_averaging_cross_entropy"
                                     ], ("ce_compaction supports cross_entropy losses only")
        assert self.cross_entropy_fusion_impl in [
            "native", "te", "linear"
        ], f"Invalid cross entropy fusion implementation: {self.cross_entropy_fusion_impl}"
        if self.ce_compaction and not self.use_linear_ce:
            assert (
                not self.cross_entropy_loss_fusion or
                self.cross_entropy_fusion_impl in ["native", "te"]
            ), (
                "non-Linear CE compaction requires cross_entropy_fusion_impl='native' or 'te' "
                "when cross_entropy_loss_fusion=True"
            )


@dataclass
class EmbeddingTrainingConfig(FinetuneTrainingConfig):
    """Training configuration for embedding SFT.

    Attributes
    ----------
    use_gbs_embedding_in_loss : bool
        If True, gather embeddings across DP and compute the contrastive
        loss at global batch size. If False, compute loss on local
        micro-batch embeddings only.
    """
    use_gbs_embedding_in_loss: bool = field(
        default=False, metadata={"help": "Use gbs embedding in loss"}
    )
    gradcache_loss_py_path: Optional[str] = field(
        default=None, metadata={"help": "Path to calc grad cache loss."}
    )
    gradcache_loss_py_name: Optional[str] = field(
        default=None, metadata={"help": "Name to calc grad cache loss."}
    )


@dataclass
class T2iSftTrainingConfig(FinetuneTrainingConfig):
    """Training configuration for text-to-image SFT, extends :class:`FinetuneTrainingConfig`.

    Attributes
    ----------
    width : int
        Width of the generated image in pixels.
    height : int
        Height of the generated image in pixels.
    """
    width: int = field(
        default=1024,
        metadata={"help": "Width of the image."},
    )
    height: int = field(
        default=1024,
        metadata={"help": "Height of the image."},
    )


@dataclass
class OffPolicyDistillTrainingConfig(FinetuneTrainingConfig):
    """Training configuration for off-policy distillation, extends :class:`FinetuneTrainingConfig`.

    Attributes
    ----------
    enable_teacher_kl_loss : bool
        Whether to enable KL divergence loss from the teacher model.
    setup_teacher_in_independent_topo : bool
        Whether to set up the teacher model in an independent topology (separate GPU group).
    enable_teacher_rollout : bool
        Whether to use the teacher model for rollout generation.
    num_batches_per_execution : int
        Number of micro-batches per forward execution.
    teacher_logits_dtype : str
        Data type of teacher logits: ``"bf16"`` or ``"fp32"``.
    """
    enable_teacher_kl_loss: bool = field(
        default=True, metadata={"help": "Whether to enable KL loss from teacher."}
    )
    setup_teacher_in_independent_topo: bool = field(
        default=False, metadata={"help": "Whether to setup teacher in independent topo"}
    )
    enable_teacher_rollout: bool = field(default=True, metadata={"help": "Whether to use sampler."})
    num_batches_per_execution: int = field(
        default=8, metadata={"help": "Number of batches per execution."}
    )
    teacher_logits_dtype: str = field(
        default="bf16", metadata={"help": "Data type of teacher logits."}
    )

    def __post_init__(self):
        super().__post_init__()
        assert not self.ce_compaction, (
            "ce_compaction currently supports SFT only, not off-policy distillation"
        )
        assert self.teacher_logits_dtype in [
            "bf16", "fp32"
        ], f"Invalid teacher logits dtype: {self.teacher_logits_dtype}"


@dataclass
class CustomActor(MappingProtocol):
    """Registration info for a custom Ray actor.

    Attributes
    ----------
    name : str
        Registered name of the custom actor class.
    cls_path : str
        Fully qualified class path, e.g. ``"my_module.MyActor"``.
    """
    name: str = field(metadata={"help": "register name of the actor class."})
    cls_path: str = field(metadata={"help": "cls path like xx.xxx.class_name"})


@dataclass
class DynamicSamplingConfig(MappingProtocol):
    """Configuration for dynamic rollout sampling.

    Each PPO step keeps sampling until ``target`` valid prompt groups are
    collected.  A group is one prompt with ``sampling_repeat_n`` responses.

    Sampling size per wave::

        num_prompts = ceil(gap * ema_expansion_ratio * oversampling_ratio)

    then rounded up to a multiple of ``rollout_mbs``.  ``gap`` is how many
    valid groups are still missing.  The first wave starts from
    ``ema_expansion_ratio = init_expansion_ratio``.  At most
    ``max_refill_times`` extra waves are allowed after the first.

    EMA expansion ratio, aggregated across DP ranks after the step::

        current_ratio = min(total_groups / num_valid_groups, max_expansion_ratio)
        # if num_valid_groups == 0: current_ratio = max_expansion_ratio
        ema = ema_decay * ema + (1 - ema_decay) * current_ratio

    Attributes
    ----------
    oversampling_ratio : float
        Extra sampling on top of the EMA estimate.  Must be ``>= 1``.
    ema_decay : float
        EMA coefficient in ``(0, 1)``.  Closer to 1 means slower updates.
    init_expansion_ratio : float
        Initial ``ema_expansion_ratio``.  Must be ``>= 1``.
    max_expansion_ratio : float
        Cap on both ``current_ratio`` and the EMA value.
    max_refill_times : int
        Extra sampling waves after the first.  ``0`` means one wave only.
        After the last wave, remaining slots are padded with invalid
        groups (``sample_mask=False``).  Fails if even invalid groups are
        not enough to fill the target.
    filter_py_path, filter_fn_name : str or None
        Optional custom group filter.  Must be set together or both omitted.
    epoch_mode : str
        ``"fixed_ppo_steps"`` or ``"consumed_data_epochs"``.
        ``consumed_data_epochs`` stops the train loop after a PPO step
        in which ``num_train_epoches`` of prompts have been consumed.
        The finishing step may wrap into the next epoch to complete refill.
    """
    oversampling_ratio: float = field(default=1.2)
    ema_decay: float = field(default=0.9)
    init_expansion_ratio: float = field(default=1.0)
    max_expansion_ratio: float = field(default=10.0)
    max_refill_times: int = field(default=3)
    filter_py_path: Optional[str] = field(default=None)
    filter_fn_name: Optional[str] = field(default=None)
    epoch_mode: str = field(default="fixed_ppo_steps")

    def __post_init__(self):
        assert self.oversampling_ratio >= 1.0
        assert 0.0 < self.ema_decay < 1.0
        assert self.init_expansion_ratio >= 1.0
        assert self.max_expansion_ratio >= self.init_expansion_ratio
        assert self.max_refill_times >= 0
        assert (self.filter_py_path is None) == (self.filter_fn_name is None)
        assert self.epoch_mode in ["fixed_ppo_steps", "consumed_data_epochs"]


@dataclass
class RLTrainingConfig(TrainingConfig):
    """Training configuration for reinforcement learning, extends :class:`TrainingConfig`.

    Attributes
    ----------
    total_ppo_step : int or None
        Total number of PPO steps. ``None`` means determined by epochs and data size.
    ppo_step_per_epoch : int or None
        Number of PPO steps per epoch.
    rollout_gbs : int
        Rollout global batch size (number of prompts per PPO step).
    rollout_mbs : int
        Rollout micro batch size. Must be ``1``.
    sampling_repeat_n : int
        Number of response samples to generate per prompt.
    sampling_keep_n : int
        Number of response samples to keep per prompt after filtering.
    eval_sampling_repeat_n : int
        Number of samples per prompt during evaluation.
    sampling_keeping_strategy : str
        Strategy to filter kept samples on the legacy actor path:
        ``"all"``, ``"test"``, ``"best-and-worst"``, or a custom name.
    dynamic_batch_rollout_filter_strategy : list of str
        Ordered sample-level filters for ``dynamic_batch_train``. Built-ins:
        ``best-and-worst``, ``sample-mask``, ``valid_group``. Empty keeps all
        samples; optional ``ppo_filter_samplings_*`` custom hook runs last.
    ppo_filter_samplings_path : str or None
        Path to a custom filter function. Legacy actor path expects
        ``(config, rollout_batches, sampling_repeat_n, sampling_keep_n)``.
        With ``dynamic_batch_train``, expects
        ``(config, samples) -> list[sample]``.
    ppo_filter_samplings_name : str or None
        Name of the filter sampling function.
    metrics_report : list of str or None
        Names of metrics to report.
    ppo_max_epochs_2 : int
        Number of epochs to train on each rollout batch.
    use_bt_rm_reward : bool
        Whether to use batch reward model.
    use_gen_rm_reward : bool
        Whether to use generative reward model.
    early_swap_model : bool
        Whether to swap model weights to inference engine early (overlap communication and compute).
    async_rollout : bool
        Whether to enable the single-controller async rollout pipeline.
    rollout_max_staleness : int
        Max PPO step staleness for async rollout.
        Only meaningful when ``async_rollout`` is enabled.
        ``s > 0`` means each sampler-weight window trains ``s`` PPO steps
        while one extra batch is prefetched under the previous weights and
        carried across ``update_weights`` to keep the GPU fed.  That carry-
        over batch may drift up to ``s`` policy-gradient steps between its
        generation and its consumption.  ``s = 0`` degenerates to fully
        synchronous training (no carry-over).  ``save_interval`` saves
        land on window boundaries (multiples of ``max(s, 1)``); the final
        step always gets a save.
    rollout_over_dispatch_ratio : float
        Tail-batching over-fire ratio.  ``1.0`` disables the feature and
        is behaviourally identical to no tail-batching.  ``> 1.0`` fires
        ``ceil(target_mb * ratio)`` microbatches and aborts whichever are
        still running once ``target_mb`` worth of rollout batches have
        been collected, trading a systematic bias against long samples
        for a shorter rollout tail.  Mutually exclusive with
        ``rollout_ordered_collection``.
    rollout_reuse_unused_prompts : bool
        Whether prompts whose generation was aborted are buffered and
        reused later.  Requires ``rollout_over_dispatch_ratio > 1.0``
        because only ``TailBatchingDataSource`` owns the buffer.
    tail_batching_discard_py_path : str or None
        Path to a ``.py`` file whose function is called once per
        ``begin_step`` with the microbatches the previous window dropped,
        as ``fn(samples) -> None`` where ``samples`` are the raw batched
        prompts.  Runs before attr-cache release and independently of
        ``rollout_reuse_unused_prompts``.  Requires
        ``tail_batching_discard_fn_name``.
    tail_batching_discard_fn_name : str or None
        Function name within ``tail_batching_discard_py_path``.
    rollout_ordered_collection : bool
        When ``False`` (default), single-controller rollout collection
        consumes the ready queue in first-finished order.  When ``True``,
        collection waits for the requested original PPO step and preserves
        original microbatch order.
    num_agent_loop_workers : int
        Number of AgentLoopActor Ray actors for the distributed rollout
        pipeline.  Only used when ``async_rollout`` is enabled.
    agent_loop_actor_cls : str or None
        Fully qualified class path of a custom ``BaseAgentLoopActor``
        subclass.  When set, replaces the built-in ``AgentLoopActor``
        in the async rollout pipeline.
    reflect_prompt : str or None
        User message appended for the reflection second turn in
        ``TwoTurnReflectAgentLoopActor``.  Ignored by the default actor.
    reflect_top_k : int or None
        Number of best / worst samples to select for the reflection
        second turn.  When set, ``rb_multiplier`` is auto-computed
        as ``1 + 2 * reflect_top_k``.
    rb_multiplier : int or None
        Number of rollout batches produced per microbatch.
        Auto-computed from ``reflect_top_k`` when not set.
    custom_actors : list of CustomActor or None
        Custom Ray actors to register for the training pipeline.
    """
    total_ppo_step: Optional[int] = field(
        default=None, metadata={"help": "Total number of ppo steps."}
    )
    ppo_step_per_epoch: Optional[int] = field(
        default=None, metadata={"help": "Number of ppo steps per epoch."}
    )
    dynamic_sampling: DynamicSamplingConfig = field(default_factory=DynamicSamplingConfig)
    rollout_gbs: int = field(default=8, metadata={"help": "Rollout global batch size."})
    rollout_mbs: int = field(default=1, metadata={"help": "Rollout micro batch size. must be 1"})
    sampling_repeat_n: int = field(
        default=8, metadata={"help": "Number of times to repeat the rollout."}
    )
    sampling_keep_n: int = field(
        default=8, metadata={"help": "Number of times to keep the rollout."}
    )
    eval_sampling_repeat_n: int = field(
        default=1, metadata={"help": "Number of times to repeat the eval rollout."}
    )
    sampling_keeping_strategy: str = field(
        default="all",
        metadata={
            "help":
                "Legacy actor filter strategy. Built-in: "
                "[all, test, best-and-worst]. "
                "Custom names require ppo_filter_samplings_path. "
                "Ignored when dynamic_batch_train=True."
        }
    )
    dynamic_batch_rollout_filter_strategy: List[str] = field(
        default_factory=list,
        metadata={
            "help":
                "Ordered dynamic-batch sample filters. Built-in: "
                "[best-and-worst, sample-mask, valid_group]. "
                "Empty list keeps all samples. Custom hook from "
                "ppo_filter_samplings_path runs after these strategies."
        },
    )
    ppo_filter_samplings_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Path to custom filter. Legacy: "
                "(config, rollout_batches, sampling_repeat_n, sampling_keep_n). "
                "dynamic_batch_train: (config, samples) -> list[sample]."
        },
    )
    ppo_filter_samplings_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the filter sampling function."}
    )
    filter_sampling_stage: Optional[str] = field(
        default='pre',
        metadata={
            "help":
                "When to apply sample filtering: 'pre' filters before logprobs/advantage "
                "computation (saves compute), 'post' filters after (allows filtering on "
                "computed advantages/logprobs, but recomputes metrics)."
        },
    )

    metrics_report: Optional[List[str]] = field(default=None, metadata={"help": "Metrics"})
    ppo_max_epochs_2: int = field(
        default=1, metadata={"help": "Number of epochs rollout batches training"}
    )
    use_bt_rm_reward: bool = field(default=False, metadata={"help": "Whether to use bt reward."})
    use_gen_rm_reward: bool = field(
        default=False, metadata={"help": "Whether to use gen rm reward."}
    )
    use_external_reward: bool = field(
        default=False, metadata={"help": "Whether to use external reward."}
    )
    stream_external_reward: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to fire each rollout micro-batch's external reward as soon as "
                "that micro-batch finishes generating, instead of one batched call after "
                "all generation completes. Honoured on the sync and agent-loop paths. "
                "Ignored unless the reward sets supports_concurrent_calls=True (see "
                "BaseExternalReward), and on the agent-loop path when use_gen_rm_reward "
                "is on."
        }
    )
    early_swap_model: bool = field(default=False, metadata={"help": "Whether to early swap model."})
    async_rollout: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to enable single-controller async rollout. "
                "Requires placement_type='disaggregated'."
        }
    )
    single_controller: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to enable single-controller colocate training. "
                "Requires placement_type='colocate'. "
                "Driver-side fire/collect loop with phased sampler/gen_rm wake/sleep."
        }
    )
    dynamic_batch_train: bool = field(
        default=False,
        metadata={
            "help":
                "Use the dynamic-batch single-controller GRPO train path. "
                "Train-step groups are fixed before filtering."
        },
    )
    custom_convert_samples_to_train_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Path to the dynamic-batch train-data conversion hook. "
                "Hook receives a flat list[sample] with _train_step_id already set."
        },
    )
    custom_convert_samples_to_train_data_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Function name to import from custom_convert_samples_to_train_data_path."
        },
    )
    custom_compute_rollout_metrics_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the controller-side rollout metrics hook."},
    )
    custom_compute_rollout_metrics_name: Optional[str] = field(
        default=None,
        metadata={"help": "Function name to import from custom_compute_rollout_metrics_path."},
    )
    rollout_max_staleness: int = field(
        default=0,
        metadata={
            "help":
                "Max PPO step staleness for async rollout. "
                "Only valid when async_rollout=True. "
                "s > 0 = train s steps per sampler-weight window; 1 extra "
                "batch is prefetched under the previous weights and carried "
                "across update_weights (up to s policy-step drift). "
                "s = 0 = fully synchronous (no carry-over)."
        }
    )
    rollout_over_dispatch_ratio: float = field(
        default=1.0,
        metadata={
            "help":
                "Tail-batching over-fire ratio. The single-controller "
                "trainer fires ceil(target_mb * ratio) microbatches per "
                "fire window but only collects target_mb worth of "
                "rollout batches. ratio=1.0 disables tail-batching, "
                "ratio>1.0 enables over-dispatch."
        }
    )
    rollout_reuse_unused_prompts: bool = field(
        default=False,
        metadata={
            "help":
                "When True, prompts from over-fired microbatches that "
                "were not used for rollout are buffered and reused in "
                "subsequent steps via TailBatchingDataSource. "
                "When False, unused prompts are discarded."
        }
    )
    tail_batching_discard_py_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Path to a .py file whose function is called once per "
                "begin_step with the microbatches the previous window "
                "dropped, as fn(samples) -> None. Runs before "
                "attr-cache release and independently of "
                "rollout_reuse_unused_prompts. Requires "
                "tail_batching_discard_fn_name."
        }
    )
    tail_batching_discard_fn_name: Optional[str] = field(
        default=None, metadata={"help": "Function name within tail_batching_discard_py_path."}
    )
    num_agent_loop_workers: int = field(
        default=4,
        metadata={
            "help":
                "Number of AgentLoopActor Ray actors for the distributed "
                "rollout pipeline. Used when async_rollout=True or "
                "single_controller=True. "
                "Workers are round-robin scheduled across cluster nodes."
        }
    )
    agent_loop_actor_cls: Optional[str] = field(
        default='gpatch_v4.rollout_generator.async_rollout.agent_loop_actor.AgentLoopActor',
        metadata={
            "help":
                "Fully qualified class path of a custom BaseAgentLoopActor "
                "subclass, e.g. 'my_module.MyAgentLoopActor'. "
                "When set, replaces the built-in AgentLoopActor in the "
                "async rollout pipeline."
        }
    )
    load_aware_sampler_routing: bool = field(
        default=False,
        metadata={
            "help":
                "When True, sampler requests are routed to the cluster with "
                "the fewest waiting requests instead of round-robin. "
                "If all clusters are busy, the actor sleeps before retrying."
        }
    )
    load_aware_sampler_dispatch_stagger_s: float = field(
        default=0.0,
        metadata={
            "help":
                "Per-slot stagger interval before querying sampler load when "
                "load-aware sampler routing is enabled. This spreads bursty "
                "generate dispatches so later requests can observe earlier "
                "requests in backend load metrics."
        }
    )
    rollout_ordered_collection: bool = field(
        default=False,
        metadata={
            "help":
                "When True, single-controller rollout collection consumes "
                "microbatches by original PPO step and microbatch order. "
                "Default False keeps first-finished queue consumption."
        }
    )
    reflect_prompt: Optional[str] = field(
        default="请重新审视你的回答，仔细检查一下你的推理过程和最终答案。",
        metadata={
            "help":
                "User message appended for the reflection second turn in "
                "TwoTurnReflectAgentLoopActor. Ignored by the default actor."
        }
    )
    reflect_top_k: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Number of best/worst samples to select for the reflection "
                "second turn. When set, rb_multiplier is auto-computed as "
                "1 + 2*reflect_top_k."
        }
    )
    rb_multiplier: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Number of rollout batches produced per microbatch. "
                "Auto-computed from reflect_top_k when not set."
        }
    )
    custom_actors: Optional[List[CustomActor]] = field(
        default=None, metadata={"help": "custom actors to be registered"}
    )
    forward_clear_memory: bool = field(
        default=False,
        metadata={"help": "Whether to periodically clear memory during forward-only passes."}
    )
    forward_clear_memory_interval: int = field(
        default=4,
        metadata={"help": "Interval (in microbatches) to clear memory during forward-only passes."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.ce_compaction:
            # TODO: support FSDP2 (SFT/GRPO/DSV4) via lm_head bypass gather before kernel.
            assert self.training_backend == "mcore", (
                "ce_compaction currently supports MCore GRPO only"
            )
            assert not self.apply_deterministic_mode, (
                "ce_compaction has not been validated with apply_deterministic_mode=True"
            )
        assert self.rb_multiplier is None, f"{self.rb_multiplier=} should not be set manually"
        if self.reflect_top_k is not None:
            assert self.agent_loop_actor_cls is not None, f"{self.agent_loop_actor_cls=} must be set when {self.reflect_top_k=} is set"
            self.rb_multiplier = 1 + 2 * self.reflect_top_k
        else:
            self.rb_multiplier = 1

        _BUILTIN_STRATEGIES = ['all', 'test', 'best-and-worst']
        if self.sampling_keeping_strategy not in _BUILTIN_STRATEGIES:
            assert self.ppo_filter_samplings_path is not None and self.ppo_filter_samplings_name is not None, (
                f"ppo_filter_samplings_path must be specified for "
                f"non-builtin strategy '{self.sampling_keeping_strategy}'"
            )
        assert self.filter_sampling_stage in [
            "pre", "post"
        ], (f"filter_sampling_stage must be 'pre' or 'post', got '{self.filter_sampling_stage}'")
        assert self.rollout_max_staleness >= 0, (
            f"rollout_max_staleness must be >= 0, got {self.rollout_max_staleness}"
        )
        if self.async_rollout:
            assert self.rollout_max_staleness >= 0, (
                "async_rollout requires rollout_max_staleness >= 0"
            )
        else:
            assert self.rollout_max_staleness == 0, (
                "rollout_max_staleness is only valid when async_rollout=True"
            )
        if self.rollout_ordered_collection:
            assert self.async_rollout or self.single_controller, (
                "rollout_ordered_collection is only valid for single-controller "
                "async or colocate rollout"
            )
        if self.dynamic_batch_train:
            assert self.single_controller
            assert self.training_backend == "mcore"
            assert self.filter_sampling_stage == "pre"
            assert not self.ppo_filter_samplings_path or self.ppo_filter_samplings_name
            assert (
                not self.custom_convert_samples_to_train_data_path or
                self.custom_convert_samples_to_train_data_name
            )
            assert (
                not self.custom_compute_rollout_metrics_path or
                self.custom_compute_rollout_metrics_name
            )
            _DYNAMIC_BATCH_FILTERS = {'best-and-worst', 'sample-mask', 'valid_group'}
            unknown = [
                name for name in self.dynamic_batch_rollout_filter_strategy
                if name not in _DYNAMIC_BATCH_FILTERS
            ]
            assert not unknown, (
                f"unknown dynamic_batch_rollout_filter_strategy entries {unknown}; "
                f"built-ins: {sorted(_DYNAMIC_BATCH_FILTERS)}"
            )
        assert self.rollout_over_dispatch_ratio >= 1.0, (
            f"rollout_over_dispatch_ratio must be >= 1.0, "
            f"got {self.rollout_over_dispatch_ratio}"
        )
        tail_batching = self.rollout_over_dispatch_ratio > 1.0
        # Ordered collection pins microbatch order; tail-batching drops whichever
        # microbatches finish last.  The two cannot both hold.
        assert not (tail_batching and self.rollout_ordered_collection), (
            "rollout_over_dispatch_ratio > 1.0 is incompatible with "
            "rollout_ordered_collection"
        )
        # Only TailBatchingDataSource owns the reuse buffer, and the controller
        # builds it solely when the ratio exceeds 1.0.
        assert not (self.rollout_reuse_unused_prompts and not tail_batching), (
            "rollout_reuse_unused_prompts requires "
            "rollout_over_dispatch_ratio > 1.0"
        )
        assert (self.tail_batching_discard_py_path
                is None) == (self.tail_batching_discard_fn_name is None), (
                    "tail_batching_discard_py_path and tail_batching_discard_fn_name "
                    "must be set together"
                )
        assert self.tail_batching_discard_py_path is None or tail_batching, (
            "tail_batching_discard_py_path requires "
            "rollout_over_dispatch_ratio > 1.0"
        )


@dataclass
class T2iRlTrainingConfig(RLTrainingConfig):
    """Training configuration for text-to-image RL, extends :class:`RLTrainingConfig`.

    Attributes
    ----------
    rollout_model_mbs : int or None
        Concurrency for the T2I diffusion model during generation.
        Defaults to ``rollout_mbs * sampling_repeat_n``.
    train_gbs_wo_timestep : int or None
        Train global batch size without timestep dimension.
    train_gas_wo_timestep : int or None
        Gradient accumulation steps without timestep dimension.
    width : int
        Width of the generated image in pixels.
    height : int
        Height of the generated image in pixels.
    max_prompt_length : int
        Maximum length of image caption (only for specific tasks).
    time : int or None
        Length of the video in frames.
    timestep_fraction : float
        Fraction of diffusion timesteps to sample for training.
    eta : float or None
        Noise eta for the flow step.
    shift : float
        Shift for the timestep scheduler.
    sampling_steps : int
        Number of diffusion sampling steps per image.
    enable_cfg : bool
        Whether to enable classifier-free guidance.
    guidance_scale : float
        Guidance scale for classifier-free guidance.
    init_same_noise : bool
        Whether to initialize the same noise for all samples in a rollout.
    disable_cfg_uncond_grad : bool
        Whether to skip gradient computation for the unconditional CFG branch.
    t2i_cfg_logps_impl_v2 : bool
        Activate new log-probability implementation for T2I CFG.
    """
    rollout_model_mbs: Optional[int] = field(
        default=None,
        metadata={
            "help": 'generate 时 t2i diffus model 的 concurrency，默认是 rollout_mbs * sampling_repeat_n'
        }
    )
    train_gbs_wo_timestep: Optional[int] = field(
        default=None, metadata={"help": "Train global batch size."}
    )
    train_gas_wo_timestep: Optional[int] = field(
        default=None, metadata={"help": "train_gas, number of gradient accumulation steps."}
    )
    width: int = field(
        default=1024,
        metadata={"help": "Width of the image."},
    )
    height: int = field(
        default=1024,
        metadata={"help": "Height of the image."},
    )
    # max_prompt_length is only for Oteam44
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "The max length of image caption. Only for Oteam44"},
    )
    time: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the video."},
    )
    timestep_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of the timestep."},
    )
    # TODO change name of eta and shift
    eta: Optional[float] = field(
        default=None,
        metadata={"help": "Noise eta for the flux_step."},
    )
    shift: float = field(
        default=1.0,
        metadata={"help": "Shift for timestep scheduler."},
    )
    sampling_steps: int = field(
        default=16,
        metadata={"help": "Number of sampling steps per images."},
    )
    enable_cfg: bool = field(
        default=False,
        metadata={"help": "Whether to enable CFG."},
    )
    guidance_scale: float = field(
        default=1.0,
        metadata={"help": "Guidance scale for the sampler."},
    )
    init_same_noise: bool = field(
        default=False,
        metadata={"help": "Whether to initialize the same noise for the rollout."},
    )
    disable_cfg_uncond_grad: bool = field(
        default=False,
        metadata={"help": "Whether to skip train cfg compute."},
    )
    # TODO 如果靠谱，删掉开关保留代码；如果不靠谱，删除掉代码。
    t2i_cfg_logps_impl_v2: bool = field(default=False, metadata={'help': '激活新的 logps 的 impl'})

    def __post_init__(self) -> None:
        super().__post_init__()
        assert not self.ce_compaction, ("ce_compaction currently supports text GRPO only")


@dataclass
class DpoTrainingConfig(FinetuneTrainingConfig):
    """Training configuration for DPO, extends :class:`FinetuneTrainingConfig`.

    Attributes
    ----------
    dpo_beta : float
        DPO beta parameter controlling the strength of the preference constraint.
    dpo_label_smoothing : float
        Label smoothing coefficient for DPO loss.
    dpo_loss_type : str
        DPO loss type. Currently only ``"sigmoid"`` is supported.
    dpo_ftx_gamma : float
        Gamma coefficient for the auxiliary FTX (fine-tuning cross-entropy) loss.
    """
    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "dpo beta."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "dpo label smoothing."},
    )
    dpo_loss_type: str = field(default="sigmoid", metadata={"help": "dpo loss type."})
    dpo_ftx_gamma: float = field(default=0.0, metadata={"help": "dpo FTX loss gamma."})

    def __post_init__(self):
        super().__post_init__()
        assert self.dpo_loss_type in [
            "sigmoid"
        ], f"not support this loss type: {self.dpo_loss_type}"


@dataclass
class RewardTrainingConfig(FinetuneTrainingConfig):
    """Training configuration for Bradley-Terry reward-model training.

    Attributes
    ----------
    build_reward_head : bool
        Replace the LM output layer with a scalar value head
        (``LinearForLastLayer`` -> 1) so the model emits a per-token scalar.
        Sequence-level reward is then pooled at the last valid token.
    """
    build_reward_head: bool = field(
        default=True, metadata={"help": "Build a scalar reward head instead of the LM head."}
    )
    loss_func: str = field(default="rm_bt", metadata={"help": "Loss function."})

    def __post_init__(self):
        super().__post_init__()
        assert self.loss_func == "rm_bt", \
            f"reward model training requires loss_func='rm_bt', got {self.loss_func}"
        assert self.build_reward_head, "reward model training requires build_reward_head=True"
