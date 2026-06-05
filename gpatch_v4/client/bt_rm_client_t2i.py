import asyncio
from typing import Any, Dict, List

import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.client.bt_rm_client import BaseBtRmClient
from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.utils.common_utils import log, logging_rank0


class T2iBtRmClient(BaseBtRmClient):
    """BT reward model client specialized for T2I tasks.

    Parameters
    ----------
    config : T2iRlConfig
    """
    def __init__(self, config: T2iRlConfig):
        super().__init__(config)

    async def generate_rewards(self, rm_idx, ppo_step, sample_idx,
                               batched_data) -> Dict[str, List[Any]]:
        """Request reward scores from a T2I BT reward model.

        Parameters
        ----------
        rm_idx : int
        ppo_step : int
        sample_idx : int
        batched_data : dict

        Returns
        -------
        dict[str, list]
            Reward outputs.
        """
        target_ep = self.rpc_client_lst[rm_idx].get_target_endpoint(
            sample_idx=sample_idx, ep_idx=None
        )
        req_dict = {
            'actor_dp_rank': self.dp_rank,
            'actor_dp_size': self.dp_size,
            'ppo_step': ppo_step,
            'sample_idx': sample_idx,
            'batched_data': batched_data,
        }
        fut = self.rpc_client_lst[rm_idx].call(target_ep, 'generate_rewards', req_dict)
        resp = await fut
        log(f'{self.__class__.__name__}.generate_rewards {ppo_step=} {sample_idx=}')
        return resp
