from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from contextlib import nullcontext

import torch


class ApplySamplingRolloutAttrBase(ABC):
    """Abstract base for rollout attribute manipulation during sampling."""
    @abstractmethod
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Remove heavy attributes before sending data to the sampler.

        Parameters
        ----------
        rollout_batch : dict

        Returns
        -------
        dict
        """
        ...

    @abstractmethod
    def remove_rollout_attr(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Remove rollout attributes from the batch.

        Parameters
        ----------
        rollout_batch : dict

        Returns
        -------
        dict
        """
        ...

    @abstractmethod
    def add_back_rollout_attr_after_sampling(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> List[Dict[str, List[Any]]]:
        """Restore attributes removed before sampling.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        list of dict
        """
        ...

    @abstractmethod
    def add_back_rollout_attr(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> List[Dict[str, List[Any]]]:
        """Restore all removed rollout attributes.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        list of dict
        """
        ...

    @abstractmethod
    def replay_rollout_batch(self, rollout_batch: Dict[str, Any]):
        """Replay a rollout batch (e.g. for experience replay).

        Parameters
        ----------
        rollout_batch : dict
        """
        ...

    @abstractmethod
    def clear_data_cache(self):
        """Clear any cached data from previous rollouts."""
        ...


class SamplerGenerateFunc(ABC):
    """Abstract callable for sampler generation."""
    @abstractmethod
    async def __call__(self, config, infer_engine, idx, tokenizer, batched_data,
                       sampling_repeat_n) -> Dict[str, List[Any]]:
        """Generate samples using the inference engine.

        Parameters
        ----------
        config : object
        infer_engine : object
        idx : int
            Sampler index.
        tokenizer : AutoTokenizer
        batched_data : dict
        sampling_repeat_n : int
            Repeated samples per prompt.

        Returns
        -------
        dict[str, list]
            Generated outputs.
        """
        ...


class PrepareDataForward(ABC):
    """Abstract base for data preparation before model forward passes.

    Parameters
    ----------
    config : object
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def grpo_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ...

    @abstractmethod
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ...

    @abstractmethod
    def prepare_loss_weights(
        self,
        loss_weights: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        ...

    def sft_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError("sft_train_with_dynamic_cp is not implemented")

    def grpo_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError("grpo_train_with_dynamic_cp is not implemented")

    def ppo_value_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError("ppo_value_train is not implemented")

    def opd_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError("opd_train is not implemented")

    def opd_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError("opd_train_with_dynamic_cp is not implemented")

    def sft_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
        raise NotImplementedError("sft_reroute_data_for_dynamic_cp is not implemented")

    def rl_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float, Dict[str, Any]]:
        raise NotImplementedError("rl_reroute_data_for_dynamic_cp is not implemented")


class PostInitModel:
    """Base for post-initialization model."""
    def __init__(self, config):
        self.config = config

    def __call__(self, model):
        """Post-init the model. Default to do nothing.

        Parameters
        ----------
        model : torch.nn.Module
        """
        pass


class CheckpointContextFn:
    """Base for torch.utils.checkpoint.checkpoint context_fn."""
    def __init__(self, config):
        self.config = config

    def __call__(self, *args, **kwargs):
        """torch.utils.checkpoint.checkpoint context_fn. Default to take in any params and do nothing.

        Parameters
        ----------
        *args : any
        **kwargs : any

        Returns
        -------
        tuple[contextlib.nullcontext, contextlib.nullcontext]
        """
        return nullcontext(), nullcontext()


class ResetRouterCorrectionBiasAccum:
    """Base for reset router load counts."""
    def __init__(self, config):
        self.config = config

    def __call__(self, model):
        """Reset router load counts. Default to do nothing.

        Parameters
        ----------
        model : torch.nn.Module
        """
        pass


class UpdateRouterCorrectionBias:
    """Base for update router correction bias."""
    def __init__(self, config):
        self.config = config

    def __call__(self, model, update_speed, use_abs_update):
        """Update router correction bias. Default to do nothing.

        Parameters
        ----------
        model : torch.nn.Module
        update_speed : float
        use_abs_update : bool

        Returns
        -------
        tuple[None, None]
        """
        return None, None