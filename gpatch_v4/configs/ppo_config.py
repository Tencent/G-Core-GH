from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class PpoConfig(MappingProtocol):
    """PPO / GRPO algorithm configuration.

    Attributes
    ----------
    advantage_type : str
        ``"grpo"`` / ``"ppo"`` / ``"on_policy_distill"`` / ``"gdpo"`` /
        ``"gdpo_sample_bn"`` / ``"custom"``.
    loss_func : str
        ``"grpo"`` / ``"steer"`` / ``"gspo"`` / ``"fipo"`` / ``"sapo"`` /
        ``"cispo"`` / ``"vespo"``.
    use_legacy_loss : bool
        If True (default), use ``loss_factory`` policy losses. If False, use
        ``gpatch_v4.training_backend.loss`` (grpo/steer/cispo/gspo/sapo/vespo
        only). ``"vespo"`` exists only on the non-legacy path.
    ppo_value_truncate_head : bool
        When aligning full-token values or per-token rewards to next-token
        log probabilities, truncate the first entry instead of the last.
    ppo_initial_policy_kl_penalty : float
    ppo_discount_factor : float
        GAE gamma.
    ppo_gae_lambda : float
    ppo_entropy_bonus : float
    ppo_ratio_eps : float
        PPO clipping epsilon for the probability ratio.
    ppo_dual_clip_ratio_c : float or None
        Must be > 1.0. See `arXiv:1912.09729 <https://arxiv.org/pdf/1912.09729>`_.
    ppo_use_absolute_kl : bool
    grpo_advantage_epsilon : float
        Numerical stability epsilon.
    ppo_logps_ratio_clamp : float or None
        Clamp log-prob ratio in ``(-x, +x)``.
    ppo_clip_ratio_low : float or None
        DAPO clip-higher low ratio. Overrides ``ppo_ratio_eps`` if set.
    ppo_clip_ratio_high : float or None
        DAPO clip-higher high ratio. Overrides ``ppo_ratio_eps`` if set.
    ppo_clamp_kl_val : float or None
        Clamp KL value in ``(-x, +x)``.
    ppo_value_clip : float or None
        Clamp value-function increment in ``(-x, +x)``.
    advantage_clip : float or None
        Symmetric advantage clipping bound ``[-x, +x]``.
    advantage_clip_lower_bound : float or None
    advantage_clip_upper_bound : float or None
    whiten_advantages: bool
        It takes effect in the reinforcement / identity stage.
    grpo_kl_loss_beta : float
    enable_off_policy_correction : bool
    off_policy_correction_level : str
        ``"token"`` / ``"sequence"`` / ``"geometric"``.
    off_policy_correction_mode : str
        ``"truncate"`` (TIS) / ``"mask"`` (MIS) / ``"icepop"`` / ``"kpop"`` / ``"clip"`` (CIS).
        ``"icepop"`` zeros out-of-range weights but keeps response_mask
        unchanged (denominator counts all valid tokens), matching verl's IcePop.
        ``"kpop"`` masks tokens where bidirectional binary KL between proximal
        and behavior policies exceeds ``off_policy_correction_upper_bound``.
    off_policy_correction_upper_bound : float
        Also used as KPop binary KL threshold φ when mode="kpop".
    off_policy_correction_lower_bound : float or None
    off_policy_correction_veto_threshold : float or None
        If any token ratio < this value, zero the entire sequence weight.
    fipo_decay_rate : float
        Future-KL decay rate τ; γ = 2^(-1/τ). Larger τ → slower decay.
    fipo_chunk_size : int
        Block size for chunked Future-KL matmul.
    fipo_clip_ratio : float
        Clipping range. ``fipo_clip_high_only=True`` → ``[1.0, 1+ratio]``;
        else ``[1-ratio, 1+ratio]``.
    fipo_clip_high_only : bool
    fipo_safety_thresh : float
        Negative-advantage tokens with IS ratio above this are capped to
        ``[0.8, 1.0]`` to prevent over-penalisation.
    fipo_correction_aware_filter : bool
        When *True*, the sequence-level dual-clip filter excludes tokens
        already zeroed by off-policy correction (e.g. icepop).
    ppo_entropy_regularization_type : str or None
        Entropy regularization type: ``"clip-cov"`` or ``"kl-cov"``. *None* to
        disable. See `arXiv:2505.22617 <https://arxiv.org/abs/2505.22617>`_.
    ppo_clip_cov_ratio : float
        [clip-cov] Fraction of valid response tokens to zero-out.
    ppo_clip_cov_ub : float
        [clip-cov] Upper bound for covariance filtering.
    ppo_clip_cov_lb : float
        [clip-cov] Lower bound for covariance filtering.
    ppo_kl_cov_ratio : float
        [kl-cov] Fraction of top-k covariance tokens to apply KL penalty.
    ppo_kl_cov_coef : float
        [kl-cov] Coefficient for KL penalty on high-covariance tokens.
    ppo_entropy_global_cov : bool
        Compute the advantage-logprob covariance over the global train_gbs
        (across DP) rather than per-micro-batch. *False* keeps per-mb behavior.
    sapo_tau_pos : float
        SAPO soft-gate temperature for positive-advantage tokens.
    sapo_tau_neg : float
        SAPO soft-gate temperature for negative-advantage tokens (``> τ_pos``
        recommended). See `arXiv:2511.20347 <https://arxiv.org/pdf/2511.20347>`_.
    vespo_c1_pos : float
        VESPO kernel exponent ``c1`` for ``A >= 0`` sequences; damps ``W < 1``.
        ``>= 1`` keeps the variance bound non-trivial.
    vespo_c2_pos : float
        VESPO kernel decay ``c2`` for ``A >= 0`` sequences; damps ``W > 1``.
    vespo_c1_neg : float
        VESPO kernel exponent ``c1`` for ``A < 0`` sequences.
    vespo_c2_neg : float
        VESPO kernel decay ``c2`` for ``A < 0`` sequences.
    use_original_logprob : bool
        If True (default), sampler returns pre-temperature (raw) logprobs and
        actor/ref recompute with temperature=1.0. If False, sampler returns
        post-temperature / processed logprobs (verl default) and actor/ref
        recompute with ``generate_params.temperature`` so IS ratios are on
        the same scale. SGLang: ``SGLANG_RETURN_ORIGINAL_LOGPROB``. vLLM:
        ``logprobs_mode=raw_logprobs`` vs ``processed_logprobs``.
    """
    advantage_type: str = field(default="grpo", metadata={"help": "Whether to use advantage."})
    # use_grpo: bool = field(default=True, metadata={"help": "Whether to use grpo."})
    loss_func: str = field(
        default="grpo",
        metadata={"help": "Loss function. [grpo, steer, gspo, fipo, sapo, cispo, vespo]"}
    )
    use_legacy_loss: bool = field(
        default=True,
        metadata={
            "help":
                "If True, use loss_factory policy losses. If False, use "
                "gpatch_v4.training_backend.loss (grpo/steer/cispo/gspo/sapo only)."
        },
    )
    loss_func_py_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the loss function."}
    )
    loss_func_py_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the loss function."}
    )
    custom_advantage_py_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Path to a custom advantage function. Legacy actor: "
                "(AdvantageContext) -> AdvantageResult. dynamic_batch_train: "
                "(config, samples) -> (advantages, returns, init_policy_kl)."
        },
    )
    custom_advantage_py_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the advantage function to import from custom_advantage_py_path."
        },
    )
    custom_post_advantage_py_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Name of the post-advantage function to import from custom_post_advantage_py_path."
        },
    )
    custom_reward_normalize_py_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Path to a dynamic-batch Controller module for custom reward "
                "normalization: (config, samples) -> metrics."
        },
    )
    custom_reward_normalize_py_name: Optional[str] = field(
        default=None,
        metadata={"help": "Function name to import from custom_reward_normalize_py_path."},
    )

    ppo_value_truncate_head: bool = field(
        default=False,
        metadata={
            "help": "Whether to truncate the head of full-token values and per-token rewards."
        },
    )
    ppo_initial_policy_kl_penalty: float = field(
        default=0.0, metadata={"help": "Initial policy kl penalty."}
    )
    ppo_discount_factor: float = field(
        default=1.0, metadata={"help": "Ppo discount factor, ppo GAE gamma."}
    )
    ppo_gae_lambda: float = field(default=0.95, metadata={"help": "Ppo gae lambda."})
    ppo_entropy_bonus: float = field(default=0.0, metadata={"help": "Ppo entropy bonus."})
    use_adaptive_entropy: bool = field(
        default=False,
        metadata={
            "help":
                "TRL-style adaptive entropy control (Skywork-OR1). When True, "
                "ppo_entropy_bonus is the initial coefficient; after each train "
                "step the coefficient and last_world_entropy are stored in "
                "PpoFeatureStore via set/get. Requires feature_store_enable=True."
        },
    )
    entropy_target: float = field(
        default=0.2,
        metadata={
            "help":
                "Target mean per-token entropy (nats) for adaptive entropy. "
                "Bonus is applied only when last_world_entropy <= this value. "
                "Default 0.2 matches TRL and only engages near collapse; tune "
                "to early-training entropy for normal runs."
        },
    )
    entropy_coef_delta: float = field(
        default=0.005,
        metadata={"help": "Adaptive entropy coefficient step size per train step."},
    )
    entropy_coef_min: float = field(
        default=0.0,
        metadata={"help": "Lower bound for adaptive entropy_coef."},
    )
    entropy_coef_max: float = field(
        default=1.0,
        metadata={"help": "Upper bound for adaptive entropy_coef."},
    )
    steer_token_weight_min: float = field(
        default=0.8,
        metadata={
            "help":
                (
                    "Minimum STEER token weight. Applies only when loss_func='steer'; "
                    "larger estimated entropy changes receive weights in "
                    "[steer_token_weight_min, 1]."
                )
        },
    )
    steer_policy_method: str = field(
        default="grpo",
        metadata={
            "help":
                (
                    "Base actor surrogate reweighted by STEER when loss_func='steer'. "
                    "Options: grpo, cispo, gspo, sapo."
                )
        },
    )
    ppo_ratio_eps: float = field(default=0.2, metadata={"help": "Ppo ratio eps."})
    ppo_dual_clip_ratio_c: Optional[float] = field(
        default=None,
        metadata={
            "help":
                "Dual-clip PPO, should be greater than 1.0, detail in: https://arxiv.org/pdf/1912.09729"
        }
    )
    ppo_use_absolute_kl: bool = field(
        default=False, metadata={"help": "Whether to use absolute kl."}
    )
    grpo_advantage_epsilon: float = field(default=1e-8, metadata={"help": "Advantage epsilon."})
    ppo_logps_ratio_clamp: Optional[float] = field(
        default=None, metadata={"help": "Clamp logps ratio in (-x, +x) for numerical stability"}
    )
    ppo_clip_ratio_low: Optional[float] = field(
        default=None,
        metadata={
            "help":
                "Dapo clip-higher low ratio. if set, ppo-ratio-eps will be replaced, ref: "
                "https://github.com/volcengine/verl/blob/main/recipe/dapo/README.md#separated-clip-epsilons---clip-higher"
        }
    )
    ppo_clip_ratio_high: Optional[float] = field(
        default=None,
        metadata={"help": "Dapo clip-higher high ratio. if set, ppo-ratio-eps will be replaced"}
    )
    ppo_clamp_kl_val: Optional[float] = field(
        default=None, metadata={"help": "Clamp kl value in (-x, +x) for numerical stability"}
    )

    ppo_value_clip: Optional[float] = field(
        default=None, metadata={"help": "Clamp value incr in (-x, +x) for numerical stability"}
    )
    advantage_clip: Optional[float] = field(
        default=None, metadata={"help": "Clip advantage value in (-x, +x)."}
    )
    advantage_clip_lower_bound: Optional[float] = field(
        default=None,
        metadata={"help": "Explicit lower bound for advantage clipping."},
    )
    advantage_clip_upper_bound: Optional[float] = field(
        default=None,
        metadata={"help": "Explicit upper bound for advantage clipping."},
    )
    whiten_advantages: bool = field(default=False, metadata={"help": "Whiten advantages"})
    reinforce_gamma: float = field(
        default=1.0, metadata={"help": "Gamma parameter for advantage calculation"}
    )

    grpo_kl_loss_beta: float = field(default=1e-3, metadata={"help": "Grpo kl loss beta."})
    norm_adv_by_std_in_grpo: bool = field(
        default=True,
        metadata={
            "help":
                "If True (default, original GRPO), divide group-centered advantages by "
                "group std. If False (Dr.GRPO / verl default for AudioRL), only subtract "
                "the group mean."
        },
    )
    # Importance Sampling
    enable_off_policy_correction: bool = False
    # Aggregation level for importance sampling weights:
    # token: per-token
    # sequence: product over tokens
    # geometric: geometric mean
    off_policy_correction_level: str = "token"
    # Handling mode for IS weights:
    # truncate: cap to upper bound, TIS
    # mask: zero outside [lower, upper] & modify mask denominator, MIS
    # icepop: zero outside [lower, upper] & keep mask unchanged (verl IcePop)
    # kpop: mask tokens where bidirectional binary KL > upper_bound (KPop)
    # clip: clip to [lower, upper], CIS
    off_policy_correction_mode: str = "truncate"
    off_policy_correction_upper_bound: float = 2.0
    off_policy_correction_lower_bound: Optional[float] = None
    # KPop binary KL clamping epsilon to avoid log(0) in Bernoulli KL computation.
    off_policy_correction_kpop_eps: float = 1e-6
    # Per-token veto threshold. If any token ratio < this, zero the entire sequence weight, the sequences won't have gradient
    # Note: float number must be written with dot e.g. 1.0e-4, not 1e-4
    off_policy_correction_veto_threshold: Optional[float] = None
    critic_model_warmup_steps: int = 0
    mask_negative_samples: bool = False

    # FIPO (Future-KL Influenced Policy Optimization) configuration
    # ref: https://arxiv.org/abs/2603.19835
    fipo_decay_rate: float = field(
        default=128.0, metadata={"help": "FIPO Future-KL decay rate τ. γ = 2^(-1/τ)."}
    )
    fipo_chunk_size: int = field(
        default=128, metadata={"help": "Block size for chunked Future-KL computation."}
    )
    fipo_clip_ratio: float = field(
        default=0.2, metadata={"help": "Clipping range for FIPO influence weights."}
    )
    fipo_clip_high_only: bool = field(
        default=False,
        metadata={"help": "If True, only clip upper bound of influence weights [1.0, 1+ratio]."}
    )
    fipo_safety_thresh: float = field(
        default=3.0,
        metadata={"help": "Safety threshold for negative-adv tokens with high IS ratio."}
    )
    fipo_correction_aware_filter: bool = field(
        default=False,
        metadata={
            "help":
                "When True, FIPO's sequence-level dual-clip filter excludes tokens "
                "already zeroed by off-policy correction (e.g. icepop mode)."
        },
    )

    # Entropy Regularization (clip-cov / kl-cov)
    # ref: https://arxiv.org/abs/2505.22617
    ppo_entropy_regularization_type: Optional[str] = field(
        default=None,
        metadata={"help": "Entropy regularization type: 'clip-cov' or 'kl-cov'. None to disable."}
    )
    ppo_clip_cov_ratio: float = field(
        default=2e-4,
        metadata={"help": "[clip-cov] Fraction of valid response tokens to zero-out."}
    )
    ppo_clip_cov_ub: float = field(
        default=5.0, metadata={"help": "[clip-cov] Upper bound for covariance filtering."}
    )
    ppo_clip_cov_lb: float = field(
        default=1.0, metadata={"help": "[clip-cov] Lower bound for covariance filtering."}
    )
    ppo_kl_cov_ratio: float = field(
        default=2e-4,
        metadata={"help": "[kl-cov] Fraction of top-k covariance tokens to apply KL penalty."}
    )
    ppo_kl_cov_coef: float = field(
        default=0.1,
        metadata={"help": "[kl-cov] Coefficient for KL penalty on high-covariance tokens."}
    )
    ppo_entropy_global_cov: bool = field(
        default=True,
        metadata={
            "help":
                "[clip-cov / kl-cov] Compute the advantage-logprob covariance over the "
                "global train_gbs (across DP) instead of per-micro-batch. When True, "
                "centering uses the global mean and kl-cov selects the true global top-rho% "
                "via a global threshold. Default True keeps the global behavior."
        }
    )
    # SAPO (Soft Adaptive Policy Optimization) configuration
    # ref: https://arxiv.org/pdf/2511.20347
    sapo_tau_pos: float = field(
        default=1.0,
        metadata={"help": "SAPO soft-gate temperature for positive-advantage tokens (τ_pos)."},
    )
    sapo_tau_neg: float = field(
        default=1.05,
        metadata={
            "help":
                "SAPO soft-gate temperature for negative-advantage tokens (τ_neg). "
                "Paper recommends τ_neg > τ_pos so negative-token gradients decay faster."
        },
    )

    # VESPO (Variational sEquence-level Soft Policy Optimization) configuration
    # ref: https://arxiv.org/abs/2602.10693
    vespo_c1_pos: float = field(
        default=2.0,
        metadata={"help": "VESPO kernel exponent c1 for positive-advantage sequences."},
    )
    vespo_c2_pos: float = field(
        default=3.0,
        metadata={"help": "VESPO kernel decay c2 for positive-advantage sequences."},
    )
    vespo_c1_neg: float = field(
        default=3.0,
        metadata={"help": "VESPO kernel exponent c1 for negative-advantage sequences."},
    )
    vespo_c2_neg: float = field(
        default=2.0,
        metadata={"help": "VESPO kernel decay c2 for negative-advantage sequences."},
    )

    gdpo_reward_weights: dict[str, Any] = field(default_factory=dict)

    skip_prev_logps: bool = field(
        default=False,
        metadata={
            "help":
                "Skip the prev_logps forward pass in on-policy training. "
                "When True, PPO ratio uses curr_log_probs - curr_log_probs.detach() "
                "(always 1.0 but with correct gradient). Requires strict on-policy "
                "condition: train_gbs == rollout_gbs * sampling_keep_n and "
                "ppo_max_epochs_2 == 1."
        },
    )
    use_original_logprob: bool = field(
        default=True,
        metadata={
            "help":
                "If True (default), sampler returns pre-temperature (raw) logprobs "
                "and actor/ref recompute with temperature=1.0. If False, sampler "
                "returns post-temperature / processed logprobs (verl default) and "
                "actor/ref recompute with generate_params.temperature so IS ratios "
                "are on the same scale. SGLang: SGLANG_RETURN_ORIGINAL_LOGPROB; "
                "vLLM: logprobs_mode raw_logprobs vs processed_logprobs."
        },
    )

    # When True, GRPO actors init PpoFeatureStore and run ppo_step_interval finalize.
    feature_store_enable: bool = field(
        default=False,
        metadata={
            "help":
                "Enable process-local PpoFeatureStore (interval finalize, sidecar save/load, "
                "extra/* metrics). Custom loss/advantage that call get_ppo_feature_store() "
                "require this to be True."
        },
    )
    post_compute_logprobs: str = field(
        default="none",
        metadata={
            "help":
                "Named hook after compute_logprobs writes prev/ref logprobs into rollout "
                "batches. Builtin: 'none' (no-op). Custom names require "
                "post_compute_logprobs_py_path / post_compute_logprobs_py_name."
        },
    )
    post_compute_logprobs_py_path: Optional[str] = field(
        default=None,
        metadata={"help": "Absolute path to a .py file for a custom post_compute_logprobs hook."},
    )
    post_compute_logprobs_py_name: Optional[str] = field(
        default=None,
        metadata={"help": "Callable name to import from post_compute_logprobs_py_path."},
    )

    def __post_init__(self):
        builtin_types = {
            "grpo",
            "reinforce",
            "ppo",
            "on_policy_distill",
            "g_opd",
            "custom",
            "gdpo",
            "gdpo_sample_bn",
            "group_gdpo",
            "group_gdpo_sample_bn",
            "identity",
        }
        if self.advantage_type not in builtin_types:
            assert (
                self.custom_advantage_py_path is not None and
                self.custom_advantage_py_name is not None
            ), (
                f"Non-builtin advantage_type '{self.advantage_type}' requires "
                "custom_advantage_py_path and custom_advantage_py_name."
            )
        if self.custom_reward_normalize_py_path is not None:
            assert self.custom_reward_normalize_py_name is not None, (
                "custom_reward_normalize_py_path requires custom_reward_normalize_py_name"
            )
        if self.loss_func == "vespo":
            self._validate_vespo()
        if self.use_adaptive_entropy:
            assert self.feature_store_enable, (
                "use_adaptive_entropy=True requires feature_store_enable=True "
                "(controller state is stored via PpoFeatureStore set/get)."
            )
            assert self.ppo_entropy_regularization_type is None, (
                "use_adaptive_entropy is incompatible with "
                f"ppo_entropy_regularization_type={self.ppo_entropy_regularization_type!r}"
            )
            assert self.entropy_coef_min <= self.entropy_coef_max, (
                f"entropy_coef_min ({self.entropy_coef_min}) must be <= "
                f"entropy_coef_max ({self.entropy_coef_max})"
            )

    def _validate_vespo(self):
        assert not self.use_legacy_loss, (
            "loss_func='vespo' is only implemented on the new loss path; "
            "set ppo.use_legacy_loss=False."
        )
        assert not self.skip_prev_logps, (
            "loss_func='vespo' requires prev_log_probs; with skip_prev_logps the "
            "sequence weight W collapses to 1 and the kernel degenerates."
        )
        for name, value in (
            ("vespo_c1_pos", self.vespo_c1_pos), ("vespo_c1_neg", self.vespo_c1_neg)
        ):
            assert value >= 1.0, (
                f"{name} must be >= 1.0, otherwise the VESPO variance bound is vacuous "
                f"(Prop. 3.1), got {value}."
            )
        for name, value in (
            ("vespo_c2_pos", self.vespo_c2_pos), ("vespo_c2_neg", self.vespo_c2_neg)
        ):
            assert value > 0.0, f"{name} must be > 0.0, got {value}."
        assert not self.enable_off_policy_correction, (
            "loss_func='vespo' folds the rollout IS ratio into W before the kernel; "
            "enable_off_policy_correction would apply it a second time."
        )


@dataclass
class T2iPpoConfig(PpoConfig):
    """PPO config for text-to-image RL.

    Attributes
    ----------
    adv_clip_max : float
    adv_clip_min : float
    """
    adv_clip_max: float = field(
        default=5.0,
        metadata={"help": "Clip max for the advantage."},
    )
    adv_clip_min: float = field(
        default=-5.0,
        metadata={"help": "Clip min for the advantage."},
    )


@dataclass
class DistillConfig(PpoConfig):
    """Configuration for distillation loss.

    Attributes
    ----------
    distill_kl_penalty_coef : float
        KL penalty between student and teacher log-probs.
    distill_kl_discount_factor : float
        Discount factor for KL advantage.
    offpd_loss_alpha : float
    offpd_temperature : float
        Temperature for the distillation softmax.
    enable_data_with_alpha : bool
        Use per-sample alpha weights from data.
    g_opd_lambda : float
        ``1.0`` = OPD, ``< 1.0`` = G-OPD, ``> 1.0`` = ExOPD (reward extrapolation).
        See `arXiv:2602.12125 <https://arxiv.org/abs/2602.12125>`_.
    g_opd_use_base_model : bool
        Load base model (student initial ckpt) as ref model for G-OPD reward
        correction. *True* → ref serves as π_base.
    log_prob_top_k : int
        Top-K size for top-K logits distillation. ``0`` (default) uses label-based
        2D log-probs; ``> 0`` enables the top-K path where advantage and PPO
        ratio become 3D ``(B, S, K)``. Ref: OPD §2.2 Eq. (5).
    """
    distill_kl_penalty_coef: float = field(
        default=1.0, metadata={"help": "kl penalty coef for student and teacher logps"}
    )
    distill_kl_discount_factor: float = field(
        default=0.0, metadata={"help": "kl discount factor for student and teacher kl_advantage"}
    )

    offpd_loss_alpha: float = field(default=0.5, metadata={"help": "Alpha for offpd loss."})
    offpd_temperature: float = field(
        default=1.0, metadata={"help": "Temperature for distill loss."}
    )
    enable_data_with_alpha: bool = field(
        default=False, metadata={"help": "Whether to use alpha in data."}
    )

    g_opd_lambda: float = field(
        default=1.0,
        metadata={
            "help":
                "G-OPD reward scaling factor. 1.0=OPD, <1.0=G-OPD, >1.0=ExOPD. "
                "ref: https://arxiv.org/abs/2602.12125"
        }
    )
    g_opd_use_base_model: bool = field(
        default=False,
        metadata={
            "help":
                "Load base model (student initial ckpt) as ref model for G-OPD reward correction."
        }
    )
    g_opd_teacher_routing_field: str = field(
        default="teacher_type",
        metadata={
            "help":
                "Data field name for per-sample teacher routing in multi-teacher mode. "
                "Ignored when only one teacher is configured."
        }
    )
    g_opd_mix_reward_advantage: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to mix GRPO reward-based advantages into the G-OPD reverse KL advantages. "
                "G-OPD paper uses ONLY reverse KL (default=False). Set to True to add reward signal."
        }
    )
    opd_teacher_kl_loss_beta: float = field(
        default=0.0,
        metadata={
            "help":
                "Beta coefficient for the teacher-student KL loss in the OPD loss. "
                "Controls the weight of KL(teacher || student) relative to the actor loss."
        }
    )
    opd_ignore_env_reward: bool = field(
        default=False,
        metadata={
            "help":
                "Pure-OPD switch: when True, drop env-side ``rewards`` before computing "
                "OPD advantages so the advantage is purely -KL(student||teacher) with no "
                "GRPO reward term. Reward is still surfaced for metrics/wandb. Only honored "
                "by ``advantage_type='on_policy_distill'`` (and is a no-op for g_opd, which "
                "has its own ``g_opd_mix_reward_advantage`` toggle)."
        }
    )
    log_prob_top_k: int = field(
        default=0,
        metadata={
            "help":
                "Top-K size for top-K logits distillation. "
                "0 (default) uses sampled-token 2D log-probs; "
                ">0 enables 3D (B, S, K) advantage and PPO ratio. "
                "Paper default is 16. Ref: OPD §2.2 Eq.(5) "
                "(https://arxiv.org/abs/2604.13016)."
        }
    )
    opd_top_k_strategy: str = field(
        default="only_stu",
        metadata={
            "help":
                "Strategy for selecting top-K token ids in OPD distillation. "
                "Effective only when log_prob_top_k > 0. Options: "
                "'only_stu' — use student's top-K ids; "
                "'only_tch' — use teacher's top-K ids; "  # 当teacher与student在topk上的权重分布差异太大，可能训崩。
                "'intersection' — intersection of student and teacher top-K (MOPD)."
            # (rionawang)TODO union: "'union' — union of student and teacher top-K (shape [S-1, 2K])."
        }
    )

    # (rionawang)TODO union: add "union" back once implemented
    _VALID_OPD_TOP_K_STRATEGIES = ("only_stu", "only_tch", "intersection")
    _VALID_STEER_POLICY_METHODS = ("grpo", "cispo", "gspo", "sapo")

    def __post_init__(self):
        super().__post_init__()
        if self.loss_func == "steer":
            assert not self.skip_prev_logps, (
                "loss_func='steer' requires prev_log_probs; set skip_prev_logps=False."
            )
            assert 0.0 < self.steer_token_weight_min <= 1.0, (
                "steer_token_weight_min must be in (0, 1], "
                f"got {self.steer_token_weight_min}."
            )
            assert self.steer_policy_method in self._VALID_STEER_POLICY_METHODS, (
                f"steer_policy_method must be one of {self._VALID_STEER_POLICY_METHODS}, "
                f"got '{self.steer_policy_method}'."
            )
        assert not (self.ppo_entropy_regularization_type is not None and self.log_prob_top_k > 0), (
            "ppo_entropy_regularization_type and log_prob_top_k > 0 are incompatible. "
            "Entropy regularization operates on 2D (B, S) log-probs, but "
            f"log_prob_top_k={self.log_prob_top_k} produces 3D (B, S, K) tensors."
        )
        assert self.opd_top_k_strategy in self._VALID_OPD_TOP_K_STRATEGIES, (
            f"opd_top_k_strategy must be one of {self._VALID_OPD_TOP_K_STRATEGIES}, "
            f"got '{self.opd_top_k_strategy}'."
        )
        if self.opd_top_k_strategy != "only_stu":
            assert self.log_prob_top_k > 0, (
                f"opd_top_k_strategy='{self.opd_top_k_strategy}' requires log_prob_top_k > 0."
            )
        if self.log_prob_top_k > 0:
            assert self.loss_func == "opd", (
                f"log_prob_top_k={self.log_prob_top_k} produces 3D advantages that only "
                f"loss_func='opd' can handle, got loss_func='{self.loss_func}'."
            )
