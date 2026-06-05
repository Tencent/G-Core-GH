import asyncio
from typing import Any, Dict, List


class SendRequestMixin:
    """Mixin providing async helpers for sending requests to sampler, gen-RM, and BT-RM."""
    async def sampler_gen_out(
        self, rbs: List[Dict[str, List[Any]]], sampler_idx, curr_ppo_step, sidx, repeat_n
    ):
        """Send generation requests to the sampler for all micro-batches.

        Parameters
        ----------
        rbs : list of dict
        sampler_idx : int
        curr_ppo_step : int
        sidx : int
        repeat_n : int

        Returns
        -------
        list
        """
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            _sidx = sidx + rbi
            co = self.sampler_client.generate(
                sampler_idx,
                curr_ppo_step,
                _sidx,
                rollout_batch,
                repeat_n=repeat_n,
                load_aware=self.config.training.load_aware_sampler_routing,
            )
            cos.append(co)
        return await asyncio.gather(*cos)

    async def get_reward_fn(
        self, rbs: List[Dict[str, List[Any]]], rm_client, curr_ppo_step: int, sample_idx: int,
        rm_idx: int
    ):
        """Send reward computation requests to a gen-RM or BT-RM client.

        Parameters
        ----------
        rbs : list of dict
        rm_client : object
        curr_ppo_step : int
        sample_idx : int
        rm_idx : int

        Returns
        -------
        list
        """
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            _sidx = sample_idx + rbi
            co = rm_client.generate_rewards(rm_idx, curr_ppo_step, _sidx, rollout_batch)
            cos.append(co)
        return await asyncio.gather(*cos)

    async def issue_bt_rm(self, rbs: List[Dict[str, List[Any]]], ppo_step, rm_idx):
        """Submit BT-RM scoring requests for all micro-batches.

        Parameters
        ----------
        rbs : list of dict
        ppo_step : int
        rm_idx : int
        """
        sample_idx = self.sample_idx
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            co = self.bt_rm_client.issue_bt_rm(rollout_batch, rm_idx, ppo_step, sample_idx + rbi)
            cos.append(co)
        return await asyncio.gather(*cos)

    async def get_bt_rm_result(self, ppo_step, num_microbatches, rm_idx):
        """Retrieve BT-RM results for all micro-batches.

        Parameters
        ----------
        ppo_step : int
        num_microbatches : int
        rm_idx : int

        Returns
        -------
        list
        """
        sample_idx = self.sample_idx
        cos = []
        for rbi in range(num_microbatches):
            co = self.bt_rm_client.get_bt_rm_result(rm_idx, ppo_step, sample_idx + rbi)
            cos.append(co)
        return await asyncio.gather(*cos)
