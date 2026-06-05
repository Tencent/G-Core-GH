"""Demo external reward — async per-sample scoring with I/O overlap."""
import asyncio
from typing import Any, Dict, List, Optional

import torch

from gpatch_v4.core.parallel_state import is_mp_and_cp_head
from gpatch_v4.reward.base_external_reward import BaseExternalReward


class DemoExternalReward(BaseExternalReward):
    """Length-penalty reward with async per-sample concurrent scoring."""
    async def calc_external_reward(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
        is_eval: bool = False,
        _started_event: Optional[asyncio.Event] = None,
    ) -> List[Dict[str, Any]]:
        update_list = [{}] * len(rollout_batches)

        if is_mp_and_cp_head():
            # Launch one async task per sample across all batches
            tasks, batch_sizes = [], []
            for batch in rollout_batches:
                seq_lens = batch["sequence_lengths"]
                prompt_lens = batch["prompt_lengths"]
                batch_sizes.append(len(seq_lens))
                for sl, pl in zip(seq_lens, prompt_lens):
                    resp_len = _to_int(sl) - _to_int(pl)
                    tasks.append(asyncio.create_task(self._score_one(resp_len)))

            # All tasks scheduled → actor can proceed with synchronous GPU work
            if _started_event:
                _started_event.set()

            flat_rewards = await asyncio.gather(*tasks)

            # Chunk flat results back into per-batch update dicts
            update_list, idx = [], 0
            for size in batch_sizes:
                update_list.append({"external_reward": list(flat_rewards[idx:idx + size])})
                idx += size
        else:
            if _started_event:
                _started_event.set()

        return update_list

    async def _score_one(self, response_len: int) -> torch.Tensor:
        """Score one sample (replace with real async I/O, e.g. httpx)."""
        await asyncio.sleep(0)  # placeholder for async API call
        return torch.tensor([-0.001 * response_len], dtype=torch.float32)


def _to_int(v) -> int:
    return v.item() if isinstance(v, torch.Tensor) else int(v)
