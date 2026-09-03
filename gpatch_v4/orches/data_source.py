import inspect
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils import import_fn_from_path, log
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


class DataSourceBase(ABC):
    """Abstract base class for rollout data sources.

    Defines the interface that all data sources must implement,
    including support for partial rollout sample buffering.
    """
    @abstractmethod
    def get_batch(self, num_microbatches: int) -> List[Dict[str, Any]]:
        """Return a batch of prompt data.

        Parameters
        ----------
        num_microbatches : int

        Returns
        -------
        list of dict
        """

    @abstractmethod
    def add_partial_samples(self, samples: List[Dict[str, Any]]):
        """Buffer partial (aborted) rollout samples for later reuse.

        Parameters
        ----------
        samples : list of dict
        """

    @abstractmethod
    def save(self, step: int):
        """Persist data source state at a checkpoint.

        Parameters
        ----------
        step : int
        """

    @abstractmethod
    def load(self, step: int):
        """Restore data source state from a checkpoint.

        Parameters
        ----------
        step : int
        """


class RolloutDataSource(DataSourceBase):
    """Read-only data source for rollout prompts (no buffer).

    Replaces per-rank ``DistributedSampler`` + ``DataLoader`` with a
    single-point provider that reads all data (dp_rank=0, dp_size=1)
    and lets the controller distribute to DP ranks.

    Parameters
    ----------
    config : RlConfig
    tokenizer : object
    """
    def __init__(self, config: RlConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._current_epoch = -1
        self._train_iter = None

        self._build_dataset_and_dataloader()

    def _build_dataset_and_dataloader(self):
        """Build dataset and dataloader as a single reader (dp_rank=0, dp_size=1)."""
        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters
        self.dp_size = 1

        cond1 = all(
            [
                len(fn_kwargs) >= 4,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            fn_ret = fn(
                config=self.config,
                tokenizer=self.tokenizer,
                dp_rank=0,
                dp_size=self.dp_size,
            )
        else:
            raise ValueError(f'unexpected data factory signature: {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

        log(
            f"[RolloutDataSource] dataset len={len(self.train_dataset)}, "
            f"dataloader len={len(self.train_dataloader)}"
        )

    def maybe_set_epoch(self, epoch: int):
        """Set epoch on the sampler and reset the iterator.

        Parameters
        ----------
        epoch : int
        """
        if self._current_epoch != epoch:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self._train_iter = iter(self.train_dataloader)
            self._current_epoch = epoch

    def get_batch(self, num_microbatches: int) -> List[Dict[str, Any]]:
        """Read a batch of prompt data from the dataloader.

        Parameters
        ----------
        num_microbatches : int
            Global count, not per-DP-rank.

        Returns
        -------
        list of dict
        """
        assert self._train_iter is not None, ("Call maybe_set_epoch() before get_batch()")
        batches = []
        for _ in range(num_microbatches):
            batched_data = next(self._train_iter)
            batches.append(batched_data)
        return batches

    def add_partial_samples(self, samples: List[Dict[str, Any]]):
        """No-op: base data source does not support partial rollout buffering."""
        pass

    def save(self, step: int):
        """No-op: base data source has no extra state to persist."""
        pass

    def load(self, step: int):
        """No-op: base data source has no extra state to restore."""
        pass

    @property
    def dataloader_length(self) -> int:
        """Length of the underlying dataloader."""
        return len(self.train_dataloader)


class BufferedDataSource(RolloutDataSource):
    """Data source with a FIFO buffer for partial (aborted) samples.

    ``get_batch()`` first drains the buffer, then reads fresh data from
    the underlying dataloader — supports the partial rollout pattern
    where aborted generation results are fed back for retry.

    Parameters
    ----------
    config : RlConfig
    tokenizer : object
    """
    def __init__(self, config: RlConfig, tokenizer):
        super().__init__(config, tokenizer)
        self._buffer: List[Dict[str, Any]] = []

    def get_batch(self, num_microbatches: int) -> List[Dict[str, Any]]:
        """Return a batch, prioritizing buffered partial samples.

        Drains the buffer first, then reads remaining micro-batches
        from the dataloader.

        Parameters
        ----------
        num_microbatches : int

        Returns
        -------
        list of dict
        """
        # Phase 1: drain buffer
        from_buffer = self._buffer[:num_microbatches]
        self._buffer = self._buffer[len(from_buffer):]
        remaining = num_microbatches - len(from_buffer)

        # Phase 2: fill from dataloader if needed
        if remaining > 0:
            from_buffer += super().get_batch(remaining)

        return from_buffer

    def add_partial_samples(self, samples: List[Dict[str, Any]]):
        """Add partial (aborted) samples to the buffer for reuse.

        Parameters
        ----------
        samples : list of dict
        """
        self._buffer.extend(samples)

    @property
    def buffer_length(self) -> int:
        """Number of buffered partial samples."""
        return len(self._buffer)

    def save(self, step: int):
        """Persist buffer state at a checkpoint.

        Parameters
        ----------
        step : int
        """
        # TODO: implement buffer serialization for checkpoint/resume
        log(
            f"[BufferedDataSource] save called at step {step}, "
            f"buffer_length={self.buffer_length} (not yet persisted)"
        )

    def load(self, step: int):
        """Restore buffer state from a checkpoint.

        Parameters
        ----------
        step : int
        """
        # TODO: implement buffer deserialization for checkpoint/resume
        log(f"[BufferedDataSource] load called at step {step} "
            f"(not yet implemented)")


def resolve_rollout_data_source_cls(config: RlConfig) -> Type[DataSourceBase]:
    """Return the rollout data source class configured for the controller."""
    py_path = config.data.custom_rollout_data_source_path
    cls_name = config.data.custom_rollout_data_source_name
    if py_path is not None or cls_name is not None:
        if not py_path or not cls_name:
            raise ValueError(
                "custom_rollout_data_source_path and "
                "custom_rollout_data_source_name must be configured together"
            )

        data_source_cls = import_fn_from_path(py_path, cls_name)
        if not (
            isinstance(data_source_cls, type)
            and issubclass(data_source_cls, DataSourceBase)
        ):
            raise TypeError(
                f"custom rollout data source '{py_path}:{cls_name}' must be a "
                f"subclass of DataSourceBase, got {data_source_cls}"
            )
        return data_source_cls

    training_config = getattr(config, "training", None)
    if (
        training_config is not None
        and training_config.rollout_over_dispatch_ratio > 1.0
    ):
        return TailBatchingDataSource

    return RolloutDataSource


def get_data_source(config: RlConfig, tokenizer: Any) -> DataSourceBase:
    """Create the configured rollout data source.

    Parameters
    ----------
    config : RlConfig
        Training configuration.
    tokenizer : object
        Tokenizer passed to the data source constructor.

    Returns
    -------
    DataSourceBase
        Configured rollout data source instance.
    """
    data_source_cls = resolve_rollout_data_source_cls(config)

    return data_source_cls(config, tokenizer)


class TailBatchingDataSource(RolloutDataSource):
    """Data source for tail-batching rollout with prompt reuse.

    Maintains a buffer of rollout-unused prompts (long-tail prompts
    whose generation was paused/aborted due to over-fire).  When the
    buffer has enough prompts for a full step, those are returned
    without over-firing; otherwise fresh data is read entirely from
    the dataloader with over-fire. Exhausted data iterators advance to
    the next data epoch and continue filling the same request.

    Parameters
    ----------
    config : RlConfig
        Training configuration.
    tokenizer : object
        Tokenizer instance.
    """
    def __init__(self, config: RlConfig, tokenizer):
        self._buffer: List[Dict[str, Any]] = []
        self._total_dl_consumed: int = 0
        super().__init__(config, tokenizer)

    def get_batch(self, target_mb: int) -> List[Dict[str, Any]]:
        """Return microbatches for one PPO step.

        If the buffer holds at least ``target_mb`` prompts, exactly
        ``target_mb`` are popped from the buffer (no over-fire, no
        dataloader read).  Otherwise the buffer is left untouched and
        fresh batches are over-fired from the dataloader.
        """
        if len(self._buffer) >= target_mb:
            batches = self._buffer[:target_mb]
            self._buffer = self._buffer[target_mb:]
            return batches
        over_fire_mb = math.ceil(target_mb * self.config.training.rollout_over_dispatch_ratio)
        assert self._train_iter is not None, ("Call maybe_set_epoch() before get_batch()")
        batches = []
        while len(batches) < over_fire_mb:
            try:
                batched_data = next(self._train_iter)
            except StopIteration:
                self._current_epoch += 1
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(self._current_epoch)
                    if isinstance(self.train_sampler, ResumableDistributedSampler):
                        self.train_sampler.start_index = 0
                self._train_iter = iter(self.train_dataloader)
                assert len(self.train_dataloader) > 0, "train dataloader is empty"
                log(f"[TailBatchingDataSource] advanced to data epoch {self._current_epoch}")
                continue
            batches.append(batched_data)
        self._total_dl_consumed += over_fire_mb
        return batches

    def maybe_set_epoch(self, epoch: int):
        """Initialize the iterator without following trainer epoch changes."""
        if self._train_iter is None:
            assert self._current_epoch == -1
            super().maybe_set_epoch(epoch)

    def add_partial_samples(self, samples: List[Dict[str, Any]]):
        """Buffer rollout-unused prompts for future reuse."""
        self._buffer.extend(samples)

    @property
    def buffer_length(self) -> int:
        """Return the number of buffered prompts."""
        return len(self._buffer)

    @property
    def total_dl_consumed(self) -> int:
        """Microbatches read from the dataloader so far, across resumes."""
        return self._total_dl_consumed

    def save(
        self,
        step: int,
        pending_unconsumed_batches: Optional[List[Dict[str, Any]]] = None,
    ):
        """Persist consumption progress, and unconsumed prompts when reuse is on."""
        save_dir = os.path.join(self.config.checkpoint.save_ckpt_path, f"iter_{step:07d}")
        os.makedirs(save_dir, exist_ok=True)

        if self.config.training.rollout_reuse_unused_prompts:
            buffer = list(self._buffer) + (pending_unconsumed_batches or [])
        else:
            buffer = []
        state = {
            "total_dl_consumed": self._total_dl_consumed,
            "current_epoch": self._current_epoch,
            "rollout_mbs": self.config.training.rollout_mbs,
            "buffer": buffer,
        }
        path = os.path.join(save_dir, "controller_data_source.pt")
        tmp_path = f"{path}.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        log(
            f"[TailBatchingDataSource] saved state to {path}: "
            f"total_dl_consumed={self._total_dl_consumed}, "
            f"unconsumed_batches={len(state['buffer'])}"
        )

    def load(self, step: int):
        """Restore data source state, dataloader position, and unconsumed prompts.

        Reads from ``load_ckpt_path`` to mirror the model checkpoint: a resumed
        run points that at the previous run's output directory, which is where
        the matching data-source state was written.
        """
        load_ckpt_path = self.config.checkpoint.load_ckpt_path
        assert load_ckpt_path is not None, (
            "load() needs checkpoint.load_ckpt_path set; the trainer only asks "
            "for a resume after finding a checkpoint there"
        )
        path = os.path.join(load_ckpt_path, f"iter_{step:07d}", "controller_data_source.pt")
        assert os.path.isfile(path), f"missing tail-batching data-source checkpoint: {path}"
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected_keys = {"total_dl_consumed", "current_epoch", "rollout_mbs", "buffer"}
        assert set(state) == expected_keys, (
            f"unexpected tail-batching checkpoint fields: "
            f"expected {sorted(expected_keys)}, got {sorted(state)}"
        )
        assert isinstance(state["total_dl_consumed"], int)
        assert isinstance(state["current_epoch"], int)
        assert isinstance(state["rollout_mbs"], int)
        assert isinstance(state["buffer"], list)
        rollout_mbs = self.config.training.rollout_mbs
        assert state["rollout_mbs"] == rollout_mbs, (
            f"tail-batching rollout_mbs mismatch: "
            f"saved {state['rollout_mbs']} vs current {rollout_mbs}"
        )
        assert self.train_sampler is not None, (
            "tail-batching resume requires a resumable train sampler"
        )
        assert hasattr(self.train_sampler, "set_start_index"
                      ), ("tail-batching resume requires train_sampler.set_start_index")

        self._total_dl_consumed = state["total_dl_consumed"]
        self._current_epoch = state["current_epoch"]
        self.train_sampler.set_start_index(self._total_dl_consumed, rollout_mbs)
        self._train_iter = iter(self.train_dataloader)
        self._buffer = state["buffer"]

        log(
            f"[TailBatchingDataSource] loaded state: "
            f"total_dl_consumed={self._total_dl_consumed}, "
            f"current_epoch={self._current_epoch}, "
            f"buffer_length={self.buffer_length}"
        )
