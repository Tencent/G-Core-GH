import asyncio
from typing import Any, Dict, List

# popped by add_back_rollout_attr_after_sampling, so the batched path never shows
# them to a reward
_SAMPLING_ONLY_KEYS = ("unique_id", "parent_unique_id", "cache_keys")


def build_stream_ready_view(rollout_batch: Dict[str, Any],
                            cached_attrs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Batch as the batched path shows it to a reward, for an on_ready consumer.

    on_ready fires inside generation, i.e. before
    add_back_rollout_attr_after_sampling puts back what
    remove_rollout_attr_before_sampling stripped, so a consumer would otherwise
    see a batch missing e.g. question / gt_label. ``cached_attrs`` is the
    handler's ``{unique_id: {key: value}}`` cache; walking the batch's own
    unique_ids is what makes this match the add-back, sampling_repeat_n included.
    """
    view = {k: v for k, v in rollout_batch.items() if k not in _SAMPLING_ONLY_KEYS}
    if not cached_attrs:
        return view

    parent_uids = rollout_batch.get("parent_unique_id")
    restored: Dict[str, List[Any]] = {}
    for idx, unique_id in enumerate(rollout_batch.get("unique_id") or []):
        if unique_id not in cached_attrs and parent_uids is not None:
            unique_id = parent_uids[idx]
        if unique_id not in cached_attrs:
            # skipping would leave the restored column short and misalign the batch
            raise RuntimeError(
                f"stream-ready view: {unique_id=} is not in the rollout-attr cache "
                f"({len(cached_attrs)} entries)"
            )
        for k, v in cached_attrs[unique_id].items():
            restored.setdefault(k, []).append(v)
    view.update(restored)
    return view


class SendRequestMixin:
    """Mixin providing async helpers for sending requests to sampler, gen-RM, and BT-RM."""
    async def sampler_gen_out(
        self,
        rbs: List[Dict[str, List[Any]]],
        sampler_idx,
        curr_ppo_step,
        sidx,
        repeat_n,
        on_ready=None
    ):
        """Send generation requests to the sampler for all micro-batches.

        Parameters
        ----------
        rbs : list of dict
        sampler_idx : int
        curr_ppo_step : int
        sidx : int
        repeat_n : int
        on_ready : callable, optional

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
                is_eval=self.run_eval,
            )
            cos.append(co)
        if on_ready is None:
            return await asyncio.gather(*cos)
        # fire on_ready per micro-batch as generation finishes, still return in index order
        results = [None] * len(cos)

        async def _indexed(_rbi, _coro):
            return _rbi, await _coro

        for _fut in asyncio.as_completed([_indexed(i, c) for i, c in enumerate(cos)]):
            _rbi, _res = await _fut
            results[_rbi] = _res
            on_ready(_rbi, _res)
            # yield, else on_ready's coroutine may not be submitted before rollout ends
            await asyncio.sleep(0)
        return results

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

    async def get_bt_rm_result(self, ppo_step, num_microbatches, rm_idx, sampling_repeat_n: int):
        """Retrieve BT-RM results for all micro-batches.

        Parameters
        ----------
        ppo_step : int
        num_microbatches : int
        rm_idx : int
        sampling_repeat_n : int
            Responses per prompt actually generated; differs between train and
            eval rollouts.

        Returns
        -------
        list
        """
        sample_idx = self.sample_idx
        cos = []
        for rbi in range(num_microbatches):
            co = self.bt_rm_client.get_bt_rm_result(
                rm_idx, ppo_step, sample_idx + rbi, sampling_repeat_n=self.sampling_repeat
            )
            cos.append(co)
        return await asyncio.gather(*cos)
