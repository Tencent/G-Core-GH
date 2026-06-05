import copy
import inspect
import uuid
from typing import Any, Dict, List

import torch

from gpatch_v4.actor.mixin import T2iTokenizerMixin
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.parallel_state import cpu_barrier, init_pg, initlize_parallel_state
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.utils import (
    clear_memory,
    import_fn_from_path,
    log,
    sync_cuda_and_get_time,
    unbind_tensor_to_list,
)


class T2iGrpoBtRmActor(BaseActor, T2iTokenizerMixin):
    async def init(self, config, rm_idx):
        super().init(config, pg_backend='nccl')

        fake_dist_config = DistConfig()
        initlize_parallel_state(config, fake_dist_config)
        init_pg(fake_dist_config)

        self.rm_idx = rm_idx
        self.reward_model = None
        self.device = torch.device(torch.cuda.current_device())
        self.init_importlib(rm_idx)

    def init_importlib(self, rm_idx):
        reward_model_info = self.config.bt_rm.reward_model_info[rm_idx]
        self.reward_model_cls = import_fn_from_path(
            reward_model_info.reward_py_path, reward_model_info.rm_cls_name
        )
        self.reward_model_info = reward_model_info
        init_method = getattr(self.reward_model_cls, '__init__')
        fn_kwargs = inspect.signature(init_method).parameters
        cond1 = all(
            [
                len(fn_kwargs) == 4,
                'self' in fn_kwargs,
                'config' in fn_kwargs,
                'rm_idx' in fn_kwargs,
                'reward_model_info' in fn_kwargs,
            ]
        )
        assert cond1, f"unexpected {cond1}"

        reward_method = getattr(self.reward_model_cls, 'compute_rewards')
        fn_kwargs = inspect.signature(reward_method).parameters
        cond1 = all([
            len(fn_kwargs) == 2,
            'self' in fn_kwargs,
            'batched_data' in fn_kwargs,
        ])
        assert cond1, f"unexpected {cond1}"

    async def init_reward_model(self, config, rm_idx):
        assert self.rm_idx == rm_idx
        g_rank = torch.distributed.get_rank()
        self.model_arch = config.bt_rm.reward_model_info[rm_idx].model_arch

        log(f'T2iGrpoBtRmActor.init create infer engine {rm_idx=} {g_rank=}')

        reward_model = self.reward_model_cls(config, rm_idx, self.reward_model_info)
        assert hasattr(reward_model, "model") and reward_model.model is not None
        self.reward_model = reward_model

    async def sleep(self):
        assert self.config.placement_type != "disaggregated"
        assert self.reward_model is not None
        self.reward_model.sleep()
        return {"ret": True}

    async def wake_up(self):
        assert self.config.placement_type != "disaggregated"
        assert self.reward_model is not None
        self.reward_model.wake_up()
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict: Dict[str, Any]):
        assert self.reward_model is not None
        if self.config.placement_type != "disaggregated":
            self.reward_model.wake_up()
        clear_memory()
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict: Dict[str, Any]):
        assert self.reward_model is not None
        if self.config.placement_type != "disaggregated":
            self.reward_model.sleep()
        return {"ret": True}

    async def generate_rewards(self, req_dict: Dict[str, Any]):
        batched_data: Dict[str, List[Any]] = req_dict["batched_data"]
        rollout_mbs = self.config.training.rollout_mbs
        repeat_n = self.config.training.sampling_repeat_n

        for k, v in batched_data.items():
            assert isinstance(v, list) and len(
                v
            ) == rollout_mbs * repeat_n, f'unexpected {k=} {v=} {rollout_mbs=} {repeat_n=} {len(v)=}'

        with torch.no_grad():
            reward_tensors = self.reward_model.compute_rewards(batched_data)

        req_dict = {
            f"reward_bt_rm_{self.rm_idx}": unbind_tensor_to_list(reward_tensors),
        }
        return req_dict
