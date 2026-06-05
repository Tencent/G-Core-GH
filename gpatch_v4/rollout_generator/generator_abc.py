from abc import ABC, abstractmethod
from typing import Any, Dict, List

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
        self.sampling_repeat = self.training_config.eval_sampling_repeat_n if \
            run_eval else self.training_config.sampling_repeat_n
        self.process_prefix = 'eval_' if run_eval else ''
        self.sample_idx = 0

    @abstractmethod
    async def rollout_samples(self, data_iter, num_microbatches, curr_ppo_step):
        """Generate rollout samples.

        Parameters
        ----------
        data_iter : iterator
        num_microbatches : int
        curr_ppo_step : int
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
