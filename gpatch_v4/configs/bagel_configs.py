import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.configs.ema_config import EmaConfig
from gpatch_v4.configs.optimizer_config import OptimizerConfig
from gpatch_v4.configs.report_config import ReportConfig
from gpatch_v4.configs.training_config import T2iRlTrainingConfig


@dataclass
class ValidationImageTransformArgs:
    """Explicit image-transform bucket used by in-training validation.

    Mirrors the per-group ``image_transform_args`` / ``vit_image_transform_args``
    in training data yaml. ``max_image_size <= 0`` means unset; validation runner
    falls back to legacy behaviour when all fields are unset.
    """

    max_image_size: int = field(
        default=0,
        metadata={"help": "Max long-edge size for validation ImageTransform. 0 = unset."},
    )
    min_image_size: int = field(
        default=0,
        metadata={"help": "Min short-edge size for validation ImageTransform. 0 = unset."},
    )
    image_stride: int = field(
        default=0,
        metadata={"help": "Stride alignment for validation ImageTransform. 0 = unset."},
    )

    def is_configured(self) -> bool:
        return (self.max_image_size > 0 and self.min_image_size > 0 and self.image_stride > 0)

    def as_image_transform_kwargs(self) -> dict:
        return {
            "max_image_size": int(self.max_image_size),
            "min_image_size": int(self.min_image_size),
            "image_stride": int(self.image_stride),
        }


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT", metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={
            "help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."
        }
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={
            "help": "Path to the pretrained VAE checkpoint for latent-space image generation."
        }
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={
            "help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."
        }
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2, metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14, metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={
            "help": "Maximum number of ViT patches along one image side after cropping / resize."
        }
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={
            "help":
                "Interpolate positional embeddings when image resolution differs from pre-training."
        }
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={
            "help":
                "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."
        }
    )
    vit_rope: bool = field(
        default=False, metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1, metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )
    recompute_layer: int = field(
        default=28,
        metadata={"help": "num layer to recompute. all layer will be recomputed by default."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4, metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={
            "help":
                "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        }
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={
            "help":
                "While batch length is below this, pop from the overflow buffer before new sampling."
        }
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={
            "help": "Soft target token count; yield the batch once it reaches or exceeds this size."
        }
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used when shuffling / sampling data shards to ensure reproducibility."
        }
    )
    debug_num_used_data: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "When set, override all num_used_data in dataset config to this value for fast debugging."
        }
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(default=True, metadata={"help": "Train image generation branch."})
    visual_und: bool = field(default=True, metadata={"help": "Train image understanding branch."})

    # --- bookkeeping & logging ---
    checkpoint_dir: str = field(
        default="results/checkpoints", metadata={"help": "Root directory for model checkpoints."}
    )
    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396, metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)."}
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={
            "help":
                "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."
        }
    )
    finetune_from_hf: bool = field(
        default=False, metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(default=10, metadata={"help": "Print / log every N training steps."})
    log_time_breakdown: bool = field(
        default=False,
        metadata={
            "help":
                "Log detailed time breakdown (IO, Forward+Backward, Optimizer) in print and wandb."
        }
    )

    save_every: int = field(
        default=2000, metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000, metadata={"help": "Total number of optimizer steps to train for."}
    )
    skip_steps: int = field(
        default=0,
        metadata={
            "help":
                "If > 0, skip forward/backward/optimizer for steps < skip_steps (only do data IO). 用于直接跳过前 N 个 step 的计算。"
        }
    )
    validation_every: int = field(
        default=0,
        metadata={"help": "Run validation every N steps. 0 to disable."},
    )
    validation_at_start: bool = field(
        default=False,
        metadata={"help": "Run validation once before the training loop starts."},
    )
    validation_gen_timesteps: int = field(
        default=16,
        metadata={"help": "Flow-matching timesteps for validation image generation."},
    )
    validation_data_dir: str = field(
        default="",
        metadata={"help": "Path to validation JSONL file or directory."},
    )
    validation_save_dir: str = field(
        default="",
        metadata={
            "help":
                "Directory to save validation results (images, metrics). Auto-generated if empty."
        },
    )
    validation_image_transform_args: ValidationImageTransformArgs = field(
        default_factory=ValidationImageTransformArgs,
        metadata={
            "help":
                (
                    "Explicit VAE image-transform bucket for validation "
                    "(max/min/stride). Mirrors per-group image_transform_args "
                    "in training data yaml. Leave all fields at 0 to keep the "
                    "legacy behaviour (use jsonl infer_image_size + model_args "
                    "fallback)."
                )
        },
    )
    validation_vit_image_transform_args: ValidationImageTransformArgs = field(
        default_factory=ValidationImageTransformArgs,
        metadata={
            "help":
                (
                    "Explicit ViT image-transform bucket for validation "
                    "(max/min/stride). Mirrors per-group vit_image_transform_args "
                    "in training data yaml. When all fields are >0 the "
                    "validation runner builds a MoonViTPreprocessor (sharing "
                    "training's ImageTransform + prepare_moonvit_patches code "
                    "path via tasks.omni.wgov3_beta.data.vit_preprocess) and "
                    "feeds it to prepare_vit_images_from_transform, so ViT "
                    "inputs match training byte-for-byte. Leave all fields at 0 "
                    "to fall back to the legacy Kimi-VL AutoProcessor path."
                )
        },
    )

    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0, metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    global_batch_size: int = field(
        default=8,
        metadata={"help": "Number of global batches before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={
            "help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."
        }
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8, metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    fsdp2_num_to_forward_prefetch: int = field(
        default=2,
        metadata={
            "help":
                "Number of future FSDP-wrapped decoder layers to pre-unshard during forward. Set to 0 to disable."
        }
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_projector: bool = field(
        default=False,
        metadata={"help": "Keep the visual projector / connector fixed during training."},
    )
    freeze_vae: bool = field(
        default=True,
        metadata={
            "help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."
        }
    )
    freeze_und: bool = field(
        default=False, metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )

    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    vae_mask: bool = field(default=False, metadata={"help": "Enable or disable VAE mask."})

    use_flash_mask: bool = field(default=False, metadata={"help": "use use_flash_mask to "})

    profile: bool = False

    manual_gc: bool = True

    manual_gc_interval: int = 20

    torch_dist_timeout_minutes: int = 300

    mp_start_method: Literal['spawn', 'fork', 'forkserver'] = "forkserver"

    log_dir: str = "./log"

    def __post_init__(self):
        if self.use_flash_mask:
            assert self.use_flex, "use_flash_mask must be used togather with flex packing"


@dataclass
class BagelConfig:
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)


@dataclass
class BagelT2iTrainingConfig(T2iRlTrainingConfig):
    # --- modality switches ---
    visual_gen: bool = field(default=True, metadata={"help": "Train image generation branch."})
    visual_und: bool = field(default=True, metadata={"help": "Train image understanding branch."})
    # --- bookkeeping & logging ---
    training_backend: str = field(
        default="fsdp2", metadata={"help": "Training backend to use (e.g., fsdp2)."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints", metadata={"help": "Root directory for model checkpoints."}
    )
    save_interval: int = field(default=100, metadata={"help": "Save checkpoint every N steps."})
    num_train_epoches: int = field(
        default=1, metadata={"help": "Number of training epochs (if used)."}
    )

    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)."}
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={
            "help":
                "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."
        }
    )

    finetune_from_hf: bool = field(
        default=False, metadata={"help": "Whether finetune from HugginFace model."}
    )
    use_torch_autocast: bool = field(
        default=False, metadata={"help": "Use torch.autocast for mixed precision."}
    )

    total_steps: int = field(
        default=500_000, metadata={"help": "Total number of optimizer steps to train for."}
    )

    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    train_mbs: int = field(default=1, metadata={"help": "Train micro batch size (per rank)."})

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8, metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    data_parallel_random_init: bool = field(
        default=False,
        metadata={"help": "Randomly init data parallel replicas for diversification."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_projector: bool = field(
        default=False,
        metadata={"help": "Keep the visual projector / connector fixed during training."},
    )
    freeze_vae: bool = field(
        default=True,
        metadata={
            "help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."
        }
    )
    freeze_und: bool = field(
        default=False, metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )

    # --- loss weights (GRPO 需要) ---
    ce_weight: float = field(
        default=0.25, metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )

    # --- training optimization (GRPO 需要) ---
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )
    vae_mask: bool = field(default=False, metadata={"help": "Enable or disable VAE mask."})
    log_time_breakdown: bool = field(
        default=False,
        metadata={
            "help":
                "Log detailed time breakdown (IO, Forward+Backward, Optimizer) in print and wandb."
        }
    )
    manual_gc: bool = field(default=True, metadata={"help": "Enable manual garbage collection."})
    manual_gc_interval: int = field(
        default=20, metadata={"help": "Interval for manual garbage collection."}
    )
    use_flash_mask: bool = field(
        default=False, metadata={"help": "use use_flash_mask instead of flex attention"}
    )

    cfg_renorm_type: str = field(
        default="global", metadata={"help": "velocity renorm type when use cfg"}
    )
    cfg_renorm_min: float = field(
        default=0.0, metadata={"help": "velocity renorm min when use cfg"}
    )


@dataclass
class BagelT2iRlConfig(T2iRlConfig):
    training: BagelT2iTrainingConfig = field(default_factory=BagelT2iTrainingConfig)
    model: ModelArguments = field(default_factory=ModelArguments)
