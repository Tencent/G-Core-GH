from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, Dict, List, Optional, Tuple

from gpatch_v4.configs.config import RlConfig


class RolloutGeneratorAbc(ABC):
    """Abstract base class for rollout generators.

    Orchestrates sampling, reward computation, and data preparation per PPO step.

    Parameters
    ----------
    config : RlConfig
    sampler_client : object
    gen_rm_client : object
    bt_rm_client : object
    run_eval : bool, optional
    """
    def __init__(
        self, config: RlConfig, sampler_client, gen_rm_client, bt_rm_client, run_eval=False
    ):
        self.config = config
        self.sampler_client = sampler_client
        self.gen_rm_client = gen_rm_client
        self.bt_rm_client = bt_rm_client

        self.training_config = config.training
        self.run_eval = run_eval
        self.sampling_repeat = self._init_sampling_repeat()
        self.process_prefix = 'eval_' if run_eval else ''
        self.sample_idx = 0

    def _init_sampling_repeat(self) -> int:
        return self.training_config.eval_sampling_repeat_n if \
            self.run_eval else self.training_config.sampling_repeat_n

    @abstractmethod
    def set_external_reward(self, external_reward) -> None:
        ...

    @abstractmethod
    def setup_data_source(
        self,
        dataloader,
        reset_iter: Callable[..., Iterator],
        resume_step: int = 0,
    ) -> Tuple[Optional[Any], bool, bool]:
        ...

    @abstractmethod
    def should_stop_for_consumed_data_epochs(self) -> bool:
        ...

    @abstractmethod
    def save_resume_state(self, step: int) -> None:
        ...

    @abstractmethod
    def pop_step_metrics(self) -> Dict[str, float]:
        ...

    @abstractmethod
    async def rollout_samples(self, data_iter, num_microbatches, curr_ppo_step, dp_rank=None):
        """Generate rollout samples.

        Parameters
        ----------
        data_iter : iterator
        num_microbatches : int
        curr_ppo_step : int
        dp_rank : int, optional
        """
        ...

    @abstractmethod
    async def generate_gen_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        """Compute generative RM rewards for rollout batches.

        Parameters
        ----------
        rbs : list of dict
        num_microbatches : int
        curr_ppo_step : int
        """
        ...

    @abstractmethod
    async def calc_bt_rm_reward(
        self, rbs: List[Dict[str, List[Any]]], num_microbatches, curr_ppo_step
    ):
        """Compute BT RM rewards for rollout batches.

        Parameters
        ----------
        rbs : list of dict
        num_microbatches : int
        curr_ppo_step : int
        """
        ...

    @abstractmethod
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step):
        """Execute a full rollout step: sample, reward, post-process.

        Parameters
        ----------
        data_iter : iterator
        num_microbatches : int
        curr_ppo_step : int
        """
        ...

    @abstractmethod
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def add_back_rollout_attr_after_sampling(self, rollout_batches: List[Dict[str, List[Any]]]):
        ...

    @abstractmethod
    def _hook_after_sampling(self, rollout_batches: List[Dict[str, List[Any]]],
                             ppo_step: int) -> List[Dict[str, List[Any]]]:
        ...

    @abstractmethod
    def _post_process_rm_rollout_batch(self, rollout_batches: List[Dict[str, List[Any]]]):
        ...

    @abstractmethod
    def clear_data_cache(self):
        ...
