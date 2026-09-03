from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

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

    def cached_rollout_attrs(self) -> Dict[str, Dict[str, Any]]:
        """``{unique_id: {key: value}}`` that ``add_back_*`` will put back.

        Only meaningful on the mp+cp head, which is where the cache is filled;
        used to show a mid-generation consumer the same fields the batched path
        sees. Empty means nothing was stripped.
        """
        return {}

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
    async def __call__(
        self,
        config,
        infer_engine,
        idx,
        tokenizer,
        batched_data,
        sampling_repeat_n,
        is_eval: bool = False
    ) -> Dict[str, List[Any]]:
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
        is_eval : bool
            When True, use ``eval_*`` sampling overrides if configured.

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

    def sft_to_mlite_packed(
        self,
        batch: List[Dict[str, Any]],
        *,
        num_microbatches: int,
        seq_length: int,
        device: torch.device,
        dp_size: int,
        dp_group,
    ):
        """Pack a finetune step batch into mlite ``PackedBatch`` + ``LossContext`` pairs.

        Used by ``training_backend=mlite``. 
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sft_to_mlite_packed"
        )

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

    def pretrain_packed(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """H2D + static-CP shard + ``PackedSeqParams`` for one packed THD mb.

        Parameters
        ----------
        batch : Dict[str, Any]
            CPU packed sample: ``tokens`` / ``labels`` / ``loss_mask`` /
            ``position_ids`` / ``cu_seqlens_padded`` / ``max_seqlen`` /
            ``padded_seq_len``; optional audio feature keys.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            ``(batch, fwd_kwargs)``. ``batch`` is on device with
            ``cp_group`` set; ``fwd_kwargs`` feeds the model forward.
            No dyn-CP / no live ``PackedSeqParams.cp_group``.

        Raises
        ------
        NotImplementedError
            Base class; subclasses MUST override.
        """
        raise NotImplementedError("pretrain_packed is not implemented")

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
