import asyncio
import copy
import inspect
import uuid
from typing import Any, Dict, List, Union

import torch
import torch.distributed

from gpatch_v4.actor.mixin import TokenizerMixin
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.parallel_state import cpu_barrier, init_pg, initlize_parallel_state
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.reward import RewardFactory
from gpatch_v4.utils import clear_memory, import_fn_from_path, log, unbind_tensor_to_list


class GrpoBtRmActor(BaseActor, TokenizerMixin):
    """Ray actor for Bradley-Terry reward model inference.

    Accumulates batch requests and computes rewards collectively.
    """
    async def init(self, config, rm_idx):
        """Initialize parallel state, tokenizer, and reward model index.

        Parameters
        ----------
        config : RlConfig
        rm_idx : int
        """
        if self.has_gpu:
            # Normal GPU-backed path
            super().init(config, pg_backend='nccl')
            self.device = torch.device(torch.cuda.current_device())
        else:
            # CPU-only path (e.g. rule_only with num_cpu_nodes>0).
            # Skip CUDA device setup and use gloo backend.
            self.config = config
            torch.distributed.init_process_group(backend='gloo')
            self.device = torch.device('cpu')

        fake_dist_config = DistConfig()
        initlize_parallel_state(config, fake_dist_config)
        init_pg(fake_dist_config)

        self.build_tokenizer()
        self.tokenizer = self.rm_tokenizers[rm_idx]

        self.rm_idx = rm_idx
        self.reward_model = None

        # bt rm 没有异步，所以先把请求攒起来在一起处理
        self.lock = asyncio.Lock()
        self.computed = False
        self.batching_reqs: List[Dict[str, Union[int, List[Any]]]] = []
        self.infer_rm_critic_results: Dict[int, Dict[int, Dict]] = {}

    async def init_reward_model(self, config, rm_idx):
        """Create the reward model using the reward factory.

        Parameters
        ----------
        config : RlConfig
        rm_idx : int
        """
        assert self.rm_idx == rm_idx
        g_rank = torch.distributed.get_rank()
        self.model_arch = config.bt_rm.reward_model_info[rm_idx].model_arch

        log(f'GrpoBtRmActor.init create infer engine {rm_idx=} {g_rank=}')

        in_args = {
            "rm_idx": self.rm_idx,
            "tokenizer": self.tokenizer,
            "actor_tokenizer": self.actor_tokenizer
        }
        self.reward_model = RewardFactory.get_reward_engine(self.config, **in_args)

    async def sleep(self):
        """Put the reward model to sleep."""
        assert self.config.placement_type != "disaggregated"
        assert self.reward_model is not None
        self.reward_model.sleep()
        return {"ret": True}

    async def wake_up(self):
        """Wake up the reward model."""
        assert self.config.placement_type != "disaggregated"
        assert self.reward_model is not None
        self.reward_model.wake_up()
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict: Dict[str, Any]):
        """Signal the start of a PPO step; wake up the reward model."""
        assert self.reward_model is not None
        assert len(self.batching_reqs) == 0
        async with self.lock:
            self.computed = False
            if self.config.placement_type != "disaggregated":
                self.reward_model.wake_up()
        clear_memory()
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict: Dict[str, Any]):
        """Signal the end of a PPO step; sleep the reward model and clear caches."""
        assert self.reward_model is not None
        async with self.lock:
            if self.config.placement_type != "disaggregated":
                self.reward_model.sleep()
            self.batching_reqs = []
            self.infer_rm_critic_results.clear()

        return {"ret": True}

    async def issue_bt_rm(self, req_dict: Dict[str, Any]):
        """Queue a reward computation request."""
        assert self.computed == False
        async with self.lock:
            self.batching_reqs.append(req_dict)
        return {"ret": True}

    async def get_bt_rm_result(self, req_dict: Dict[str, Any]):
        """Compute (if needed) and return reward results for a request.

        Parameters
        ----------
        req_dict : dict
            Request with ``'actor_dp_rank'``, ``'ppo_step'``, ``'sample_idx'``,
            and ``'sampling_repeat_n'``.

        Returns
        -------
        dict
            Reward results.
        """
        actor_dp_rank = req_dict['actor_dp_rank']
        ppo_step = req_dict['ppo_step']
        sample_idx = req_dict['sample_idx']
        sampling_repeat_n = req_dict['sampling_repeat_n']

        async with self.lock:
            if not self.computed:
                self.computed = True
                with torch.no_grad():
                    resp_dicts = self.reward_model.compute_rewards(
                        self.batching_reqs, sampling_repeat_n
                    )

                for batch, b_resp_dict in zip(self.batching_reqs, resp_dicts, strict=True):
                    b_actor_dp_rank = batch['actor_dp_rank']
                    b_sample_idx = batch['sample_idx']
                    b_ppo_step = batch['ppo_step']

                    tmpd = self.infer_rm_critic_results.setdefault(b_actor_dp_rank, {})
                    tmpdd = tmpd.setdefault(b_ppo_step, {})
                    tmpdd[b_sample_idx] = b_resp_dict

                self.batching_reqs = []

        tmpd = self.infer_rm_critic_results[actor_dp_rank][ppo_step]
        resp_dict = tmpd[sample_idx]

        return resp_dict
