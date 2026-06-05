import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils import import_fn_from_path, log


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
                dp_size=1,
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
