from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import is_mp_head


class RewardAbc(ABC):
    """Abstract base class for reward computation.

    Parameters
    ----------
    config : RlConfig
    rm_idx : int
        Reward model index in the configuration.
    tokenizer : AutoTokenizer
        Reward model tokenizer.
    actor_tokenizer : AutoTokenizer
        Policy actor tokenizer.
    """
    def __init__(
        self, config: RlConfig, rm_idx: int, tokenizer: AutoTokenizer,
        actor_tokenizer: AutoTokenizer
    ):
        self.config = config
        self.rm_idx = rm_idx
        self.tokenizer = tokenizer
        self.actor_tokenizer = actor_tokenizer
        self.reward_model = None

    @abstractmethod
    def setup_reward_model(self):
        """Initialize the underlying reward model."""
        ...

    @abstractmethod
    def compute_rewards(
        self, batched_data: List[Dict[str, Union[int, List[Any]]]], sampling_repeat_n: int = None
    ):
        """Compute reward scores for a batch of data.

        Parameters
        ----------
        batched_data : list of dict
            Batch of data to score.
        """
        ...
