import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class DataConfig(MappingProtocol):
    """Configuration for data loading and preprocessing.

    Attributes
    ----------
    py_path : str or None
        Python file containing the dataset implementation.
    fn_name : str or None
        Function returning the dataloader.
    get_batch_fn_name : str or None
    data_pathes : list of str or None
    eval_data_pathes : list of str or None
    sampler_seed : int
    system_prompt : str or None
    dataloader_num_workers : int
    multiprocessing_method : str
    dataloader_pin_memory : bool
    dataloader_prefetch_factor : int or None
    mask_history : bool
        Mask conversation history tokens in the loss.
    tokenizer_eos_token : str or None
        EOS override (only used when ``dataset_debug=True``).
    custom_add_eos : bool
    dataset_debug : bool
    dataloader_verify_fn_name : str or None
    """
    py_path: Optional[str] = field(default=None, metadata={"help": "dataset impl python file"})
    fn_name: Optional[str] = field(default=None, metadata={"help": "get dataloader function name"})

    get_batch_fn_name: Optional[str] = field(
        default=None, metadata={"help": "get batch function name"}
    )
    data_pathes: Optional[List[str]] = field(
        default=None, metadata={"help": "Pathes to the data files (or configs)"}
    )
    eval_data_pathes: Optional[List[str]] = field(
        default=None, metadata={"help": "Pathes to the evaluation data files (or configs)"}
    )
    # caption sampling ratios (for GRPO JSONL dataset)
    caption_long_ratio: float = field(default=0.7, metadata={"help": "ratio for long captions"})
    caption_medium_ratio: float = field(default=0.2, metadata={"help": "ratio for medium captions"})
    caption_short_ratio: float = field(default=0.1, metadata={"help": "ratio for short captions"})
    sampler_seed: int = field(default=42, metadata={"help": "Seed for the sampler"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt"})
    dataloader_num_workers: int = field(
        default=1, metadata={"help": "Number of workers for the dataloader"}
    )
    multiprocessing_method: str = field(
        default="forkserver", metadata={"help": "Dataloader worker multiprocessing start method"}
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether to pin memory for the dataloader"}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None, metadata={"help": "Number of batches loaded"}
    )
    mask_history: bool = field(default=False, metadata={"help": "Whether to mask history"})
    tokenizer_eos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer EOS token, using only when dataset_debug is True"}
    )
    custom_add_eos: bool = field(default=False, metadata={"help": "Whether to add EOS token"})
    dataset_debug: bool = field(default=False, metadata={"help": "Whether to debug the dataset"})

    dataloader_verify_fn_name: Optional[str] = field(
        default=None, metadata={"help": "the dataloader verify fn_name in 'py_path'"}
    )

    def __post_init__(self):
        assert self.multiprocessing_method in ["fork", "forkserver", "spawn"], "multiprocessing_method must be 'fork' or 'forkserver' or 'spawn'"
