import copy
import inspect
import uuid
from typing import Any, Dict, List

import torch
from typing_extensions import override

from gpatch_v4.actor.grpo_sampler_actor import GrpoSamplerActor
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.parallel_state import cpu_barrier, init_pg, initlize_parallel_state
from gpatch_v4.generation_backend import InferEngine
from gpatch_v4.patch.sglang_patch import sglang_hack
from gpatch_v4.utils import import_fn_from_path, log


class OffPolicyDistillSamplerActor(GrpoSamplerActor):
    async def init(self, config):
        await super().init(config)

    @override
    def post_init(self, idx):
        model_info = self.config.sampler.model_info[idx]
        if model_info.gen_rollout_py_path is not None and model_info.gen_rollout_fn_name is not None:
            generate_func = import_fn_from_path(
                model_info.gen_rollout_py_path, model_info.gen_rollout_fn_name
            )
            fn_kwargs = inspect.signature(generate_func).parameters
            cond1 = all(
                [
                    len(fn_kwargs) == 7,
                    'config' in fn_kwargs,
                    'infer_engine' in fn_kwargs,
                    'idx' in fn_kwargs,
                    'tokenizer' in fn_kwargs,
                    'student_tokenizer' in fn_kwargs,
                    'batched_data' in fn_kwargs,
                    'sampling_repeat_n' in fn_kwargs,
                ]
            )
            assert cond1, f"unexpected {cond1}"
            self.generate_func = generate_func
        else:
            raise NotImplementedError(f"generate_func not provided for model {idx}")

    @override
    async def generate(self, req_dict: Dict[str, Any]):
        batched_data: Dict[str, List[Any]] = req_dict["batched_data"]
        repeat_n = req_dict["sampling_repeat"]
        train_mbs = self.config.training.train_mbs
        for k, v in batched_data.items():
            assert isinstance(
                v, list
            ) and len(v) == train_mbs, f'unexpected {k=} {v=} {train_mbs=} {len(v)=}'

        rollout_batch = await self.generate_func(
            config=self.config,
            infer_engine=self.infer_engine,
            idx=self.idx,
            tokenizer=self.tokenizer,
            student_tokenizer=self.actor_tokenizer,
            batched_data=batched_data,
            sampling_repeat_n=repeat_n
        )

        return rollout_batch
