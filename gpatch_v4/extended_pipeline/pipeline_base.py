from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.training_backend.fsdp2_backend.swap import offload_model, onload_model


class ExtendedPipelineAbc(ABC):
    """Abstract pipeline hiding multi-model interaction complexity for T2I/T2V.

    Mimics the diffusers pipeline interface while adding training and
    scalability optimizations.
    """
    @abstractmethod
    def setup_pipeline(self):
        """Set up tokenizers, encoders, and the model.

        Returns
        -------
        int
            Last PPO step of the previous run (for resume).
        """
        ...

    @abstractmethod
    def encode_prompt(self, **kwargs):
        """Encode text prompts into embeddings."""
        ...

    @abstractmethod
    def repeat_interleave_tensor_or_list(
        self,
        rb: Dict[str, Union[List[Any], torch.Tensor]],
        repeat: int,
    ):
        """Repeat-interleave each value in the rollout batch.

        Parameters
        ----------
        rb : dict
            Rollout batch whose values are tensors or lists.
        repeat : int
        """
        ...

    @abstractmethod
    def permute_timesteps(
        self,
        rb: Dict[str, List[Any]],
    ):
        """Permute timestep-related fields in a rollout batch.

        Parameters
        ----------
        rb : dict
        """
        ...

    @abstractmethod
    def __call__(self, **kwargs):
        """Run the full pipeline (similar to diffusers), returning log probs.

        Argument names should match the corresponding diffusers pipeline.

        Parameters
        ----------
        **kwargs : dict
            Pipeline inputs (e.g. prompts, height, width, …).

        Returns
        -------
        dict
            Pipeline outputs.
        """
        ...

    @abstractmethod
    def ppo_train_step(self, rollout_batches: List[Dict[str, Any]]):
        """Execute one PPO training step.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        dict
            Training metrics.
        """
        ...

    @abstractmethod
    def sft_train_step(self, rollout_batches: List[Dict[str, Any]]):
        """Execute one SFT training step.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        dict
            Training metrics.
        """
        ...
