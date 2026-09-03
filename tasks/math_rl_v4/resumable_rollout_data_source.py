import inspect
import os
from typing import Any, Dict, List, Optional

import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.orches.data_source import RolloutDataSource
from gpatch_v4.utils import import_fn_from_path, log
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


class ResumableRolloutDataSource(RolloutDataSource):
    """Rollout data source that checkpoints its logical training cursor.

    .. warning::

        This data source does not support asynchronous training. Using it
        with ``training.async_rollout=True`` can lose prompt data when
        resuming from a checkpoint.

    The saved cursor follows the model checkpoint step rather than the
    potentially leading prefetch cursor. The dataset factory receives the
    restored PPO step through ``meta_info["resume_step"]``.

    Parameters
    ----------
    config : RlConfig
        Training configuration.
    tokenizer : object
        Tokenizer forwarded to the configured dataset factory.
    """
    def __init__(self, config: RlConfig, tokenizer):
        self._load_ckpt_path = config.checkpoint.load_ckpt_path
        if (
            self._load_ckpt_path is None
            and config.training.auto_load_from_save_ckpt
            and config.checkpoint.save_ckpt_path is not None
        ):
            marker_path = os.path.join(
                config.checkpoint.save_ckpt_path,
                "latest_checkpointed_iteration.txt",
            )
            if os.path.isfile(marker_path):
                self._load_ckpt_path = config.checkpoint.save_ckpt_path

        self._resume_step: Optional[int] = None
        if self._load_ckpt_path is not None:
            marker_path = os.path.join(
                self._load_ckpt_path,
                "latest_checkpointed_iteration.txt",
            )
            if os.path.isfile(marker_path):
                with open(marker_path, "r") as f:
                    self._resume_step = int(f.read().strip())

        super().__init__(config, tokenizer)

        self._total_dl_consumed = 0
        self._dataloader_length = len(self.train_dataloader)
        rollout_gas = (
            self.config.training.rollout_gbs // self.config.training.rollout_mbs
        )
        assert rollout_gas > 0, f"rollout_gas must be positive, got {rollout_gas}"

        if self._resume_step is not None:
            self.load(self._resume_step)

        assert self._dataloader_length > 0, "train dataloader is empty"

        if self._resume_step is not None:
            ppo_step_per_epoch = self._dataloader_length // rollout_gas
            expected_total_dl_consumed = self._resume_step * rollout_gas
            expected_epoch = self._resume_step // ppo_step_per_epoch
            assert self._total_dl_consumed == expected_total_dl_consumed, (
                f"data-source checkpoint total_dl_consumed mismatch for step "
                f"{self._resume_step}: expected {expected_total_dl_consumed}, "
                f"got {self._total_dl_consumed}"
            )
            assert self._current_epoch == expected_epoch, (
                f"data-source checkpoint current_epoch mismatch for step "
                f"{self._resume_step}: expected {expected_epoch}, "
                f"got {self._current_epoch}"
            )
            consumed_in_epoch = (
                self._resume_step % ppo_step_per_epoch
            ) * rollout_gas
            expected_remaining = self._dataloader_length - consumed_in_epoch
            assert len(self.train_dataloader) == expected_remaining, (
                f"resumed dataloader length mismatch: expected {expected_remaining}, "
                f"got {len(self.train_dataloader)}; dataset factory must apply "
                f"meta_info['resume_step']"
            )
            self._train_iter = iter(self.train_dataloader)

    def _build_dataset_and_dataloader(self):
        """Build the rollout dataloader and pass the restored PPO step."""
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
        if not cond1:
            raise ValueError(f'unexpected data factory signature: {fn_kwargs}')

        extra_args = {}
        if self._resume_step is not None:
            assert "meta_info" in fn_kwargs, (
                "data factory must accept meta_info for rollout resume"
            )
            extra_args["meta_info"] = {"resume_step": self._resume_step}
        fn_ret = fn(
            config=self.config,
            tokenizer=self.tokenizer,
            dp_rank=0,
            dp_size=self.dp_size,
            **extra_args,
        )
        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

        log(
            f"[ResumableRolloutDataSource] dataset len={len(self.train_dataset)}, "
            f"dataloader len={len(self.train_dataloader)}"
        )

    def maybe_set_epoch(self, epoch: int):
        """Set the sampler epoch and clear a restored offset on later epochs."""
        if self._current_epoch != epoch:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
                if isinstance(self.train_sampler, ResumableDistributedSampler):
                    self.train_sampler.start_index = 0
            self._train_iter = iter(self.train_dataloader)
            self._current_epoch = epoch

    def get_batch(self, num_microbatches: int) -> List[Dict[str, Any]]:
        """Read prompt microbatches and advance the runtime cursor."""
        assert self._train_iter is not None, (
            "Call maybe_set_epoch() before get_batch()"
        )
        batches = []
        for _ in range(num_microbatches):
            batches.append(next(self._train_iter))
            self._total_dl_consumed += 1
        return batches

    def save(self, step: int):
        """Persist the logical data position matching a model checkpoint."""
        save_ckpt_path = self.config.checkpoint.save_ckpt_path
        assert save_ckpt_path is not None, (
            "ResumableRolloutDataSource.save needs checkpoint.save_ckpt_path"
        )
        rollout_gas = (
            self.config.training.rollout_gbs // self.config.training.rollout_mbs
        )
        ppo_step_per_epoch = self._dataloader_length // rollout_gas
        committed_dl_consumed = step * rollout_gas
        assert self._total_dl_consumed >= committed_dl_consumed, (
            f"runtime dataloader cursor {self._total_dl_consumed} is behind "
            f"checkpoint step {step} ({committed_dl_consumed} microbatches)"
        )

        state = {
            "total_dl_consumed": committed_dl_consumed,
            "current_epoch": step // ppo_step_per_epoch,
            "rollout_mbs": self.config.training.rollout_mbs,
            "dataloader_length": self._dataloader_length,
        }
        save_dir = os.path.join(save_ckpt_path, f"iter_{step:07d}")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "controller_data_source.pt")
        tmp_path = f"{path}.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        log(
            f"[ResumableRolloutDataSource] saved state to {path}: "
            f"committed_dl_consumed={committed_dl_consumed}, "
            f"runtime_dl_consumed={self._total_dl_consumed}"
        )

    def load(self, step: int):
        """Restore the logical data position for a model checkpoint."""
        assert self._load_ckpt_path is not None
        path = os.path.join(
            self._load_ckpt_path,
            f"iter_{step:07d}",
            "controller_data_source.pt",
        )
        assert os.path.isfile(path), (
            f"missing rollout data-source checkpoint: {path}"
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected_keys = {
            "total_dl_consumed",
            "current_epoch",
            "rollout_mbs",
            "dataloader_length",
        }
        assert set(state) == expected_keys, (
            f"unexpected rollout data-source checkpoint fields: "
            f"expected {sorted(expected_keys)}, got {sorted(state)}"
        )
        for key in expected_keys:
            assert type(state[key]) is int, (
                f"rollout data-source checkpoint field {key} must be int, "
                f"got {type(state[key])}"
            )
        rollout_mbs = self.config.training.rollout_mbs
        assert state["rollout_mbs"] == rollout_mbs, (
            f"rollout data-source rollout_mbs mismatch: "
            f"saved {state['rollout_mbs']} vs current {rollout_mbs}"
        )
        assert state["total_dl_consumed"] >= 0
        assert state["current_epoch"] >= 0
        assert state["dataloader_length"] > 0

        self._total_dl_consumed = state["total_dl_consumed"]
        self._current_epoch = state["current_epoch"]
        self._dataloader_length = state["dataloader_length"]
        log(
            f"[ResumableRolloutDataSource] loaded state from {path}: "
            f"total_dl_consumed={self._total_dl_consumed}, "
            f"current_epoch={self._current_epoch}"
        )

    @property
    def dataloader_length(self) -> int:
        """Full dataloader length before applying the restored offset."""
        return self._dataloader_length

    @property
    def total_dl_consumed(self) -> int:
        """Number of microbatches read from the dataloader."""
        return self._total_dl_consumed

    @property
    def resume_step(self) -> int:
        """Checkpoint step loaded during initialization, or zero."""
        return self._resume_step or 0
