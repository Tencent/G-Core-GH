import os
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.checkpoint_config import CheckpointConfig
from gpatch_v4.configs.ema_config import EmaConfig
from gpatch_v4.configs.optimizer_config import OptimizerConfig
from gpatch_v4.configs.policy_config import T2iPolicyConfig
from gpatch_v4.configs.report_config import ReportConfig
from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class DpoDataConfig(MappingProtocol):
    #### data configs
    "Datasplit"
    split: str = "train"
    "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    dataloader_num_workers: int = 0
    "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
    " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
    " or to a folder containing files that 🤗 Datasets can understand."
    dataset_name: str = None
    "The config of the Dataset, leave as None if there's only one config.",
    dataset_config_name: str = None
    "A folder containing the training data. Folder contents must follow the structure described in"
    " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
    " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."

    train_data_dir: str = None
    "The column of the dataset containing an image."
    image_column: str = "image"
    "The column of the dataset containing a caption or a list of captions.",
    caption_column: str = "caption"
    proportion_empty_prompts: float = 0.2
    "Only train on pairs where both generations are from dreamlike"
    dreamlike_pairs_only: bool = False
    "If set the images will be randomly"
    " cropped (instead of center). The images will be resized to the resolution first before cropping."
    random_crop: bool = False
    "whether to supress horizontal flipping"
    no_hflip: bool = False
    "The directory where the downloaded models and datasets will be stored."
    cache_dir: str = None


@dataclass
class DpoT2iTrainingConfig(MappingProtocol):
    ### training config
    training_backend: str = "fsdp2"
    "Run Supervised Fine-Tuning instead of Direct Preference Optimization"
    sft: bool = False
    'choices=["no", "fp16", "bf16"],'
    "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
    " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
    " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
    " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    allow_tf32: bool = False
    mixed_precision: str = "fp16"
    "The scale of input perturbation. Recommended 0.1."
    input_perturbation: float = 0.0
    "Revision of pretrained model identifier from huggingface.co/models.",
    revision: str = None
    "For debugging purposes or quicker training, truncate the number of training examples to this "
    "value if set."
    max_train_samples: int = None
    "A seed for reproducible training."
    seed: int = None
    "The resolution for input images, all the images in the dataset will be resized to this"
    " resolution"
    resolution: int = None
    height: int = None
    width: int = None
    "Batch size (per device) for the training dataloader."
    train_batch_size: int = 1
    num_train_epochs: int = 100
    "Total number of training steps to perform.  If provided, overrides num_train_epochs."
    max_train_steps: int = 2000
    "Number of updates steps to accumulate before performing a backward/update pass."
    gradient_accumulation_steps: int = 1
    "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
    recompute: bool = False
    "The scale of noise offset."
    noise_offset: float = 0.0
    "Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    hard_skip_resume: bool = False
    "Initialize start of run from unet (not compatible w/ checkpoint load)"
    unet_init: str = ""
    "Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."
    "Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    choice_model: str = ""
    "Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038."
    pretrained_vae_model_name_or_path: str = None
    "Path to pretrained model or model identifier from huggingface.co/models."
    pretrained_model_name_or_path: str = None
    "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."
    scale_lr: bool = False
    max_prompt_length: int = 256
    guidance_scale: float = 0.0
    save_interval: int = 20

    shift: float = field(
        default=1.0,
        metadata={"help": "Shift for timestep scheduler."},
    )

    sampling_steps: int = field(
        default=16,
        metadata={"help": "Number of sampling steps per images."},
    )

    @property
    def total_ppo_step(self):
        """bypass fsdp engine"""
        return self.max_train_steps

    def __post_init__(self):
        if self.resolution is None:
            self.resolution = 512
        self.train_method = 'sft' if self.sft else 'dpo'
        if self.height is None:
            self.height = self.resolution
        if self.width is None:
            self.width = self.resolution


@dataclass
class DiffusionDpoConfig(MappingProtocol):
    beta_dpo: float = 5000


@dataclass
class T2iDpoConfig(MappingProtocol):
    data: DpoDataConfig = field(default_factory=DpoDataConfig)
    training: DpoT2iTrainingConfig = field(default_factory=DpoT2iTrainingConfig)
    policy: T2iPolicyConfig = field(default_factory=T2iPolicyConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    dpo: DiffusionDpoConfig = field(default_factory=DiffusionDpoConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    local_rank: int = -1
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank
        assert self.training.max_train_steps is not None
