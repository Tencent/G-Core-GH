from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer


class BaseEngine(ABC):
    """Abstract base class for training engines.

    Parameters
    ----------
    config : object
        Top-level training configuration.
    policy_config : object
    tokenizer : AutoTokenizer
    """
    def __init__(self, config, policy_config, tokenizer: AutoTokenizer):
        super().__init__()
        self.config = config
        self.policy_config = policy_config
        self.training_config = config.training
        self.dist_config = policy_config.dist_config
        self.checkpoint_config = config.checkpoint

        self.ppo_config = None
        if hasattr(config, "ppo"):
            self.ppo_config = config.ppo

        self.tokenizer = tokenizer
        # lasy init model
        self.model = None
        self.ref_model = None
        self.swap_impl = None

        # Set by the actor before rl_train_actor() to indicate the current ppo step needs dump.
        self.should_dump_metrics = False

    @abstractmethod
    def setup_model_and_get_optimizer(self):
        """Build the model and return its optimizer."""
        ...

    @abstractmethod
    def compute_log_probs(
        self, rollout_batches: List[Dict[str, List[Any]]], compute_pre_logps=True
    ):
        """Compute log probabilities for rollout batches.

        Parameters
        ----------
        rollout_batches : list of dict
        compute_pre_logps : bool, optional
            Whether to compute reference log probs, by default *True*.
        """
        ...

    @abstractmethod
    def rl_train_actor(self, dataloader_iter):
        """Execute one RL training step.

        Parameters
        ----------
        dataloader_iter : iterator
        """
        ...

    @abstractmethod
    def finetune_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        """Execute one fine-tuning step.

        Parameters
        ----------
        batch : list of dict
        num_microbatches : int
            Gradient-accumulation micro-batches.
        step : int
        """
        ...

    @abstractmethod
    def pretrain_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        """Execute one fine-tuning step.

        Parameters
        ----------
        batch : list of dict
        num_microbatches : int
            Gradient-accumulation micro-batches.
        step : int
        """
        ...

    @abstractmethod
    def set_model_eval(self):
        """Set the model to evaluation mode."""
        ...

    @abstractmethod
    def set_model_train(self):
        """Set the model to training mode."""
        ...
