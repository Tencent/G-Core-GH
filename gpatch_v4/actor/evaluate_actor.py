import asyncio
import inspect
import os
import traceback
from typing import Any, Dict, List

import torch.distributed as dist
import torch.nn.functional
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.mixin import TokenizerMixin
from gpatch_v4.client import SamplerClient
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.utils import (
    expand_rollout_batches,
    format_config,
    import_fn_from_path,
    is_same_tokenizer,
    log,
    logging_rank0,
)


class EvaluateActor(BaseActor, TokenizerMixin):
    async def init(self, config):
        super().init(config)

        # init parallel group
        initlize_parallel_state(config, config.policy.dist_config)
        init_pg(config.policy.dist_config)

        self.build_tokenizer()
        self.tokenizer = self.actor_tokenizer
        is_same_tokenizer(self.tokenizer, self.sampler_tokenizers[0])
        self.build_dataset_and_dataloader()

        dist_config = config.policy.dist_config
        assert mpu.get_data_parallel_world_size(
        ) == dist_config.num_gpus_per_node * dist_config.nnodes, f"{mpu.get_data_parallel_world_size()=} != {dist_config.num_gpus_per_node * dist_config.nnodes}"
        self.validate_config()
        logging_rank0(f"{self.__class__.__name__} config {format_config(self.config)}")

    def validate_config(self):
        assert self.config.training.use_fast_tokenizer is True
        assert self.config.training.train_mbs == 1

    def build_dataset_and_dataloader(self):
        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) == 4,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            fn_ret = fn(
                config=self.config,
                tokenizer=self.tokenizer,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
            )
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

    async def setup_client(self):
        self.sampler_client = SamplerClient(self.config)
        await self.sampler_client.maybe_init_distributed_weight_group_for_disagg()

    async def sampler_gen_out(
        self, rbs: List[Dict[str, List[Any]]], sampler_idx, curr_train_step, sidx, repeat_n
    ):
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            _sidx = sidx + rbi
            co = self.sampler_client.generate(
                sampler_idx, curr_train_step, _sidx, rollout_batch, repeat_n=repeat_n
            )
            cos.append(co)
        return await asyncio.gather(*cos)

    async def _evaluate(self):
        evaluate_func = import_fn_from_path(
            self.config.evaluate_result.evaluate_py_path,
            self.config.evaluate_result.evaluate_fn_name
        )
        fn_kwargs = inspect.signature(evaluate_func).parameters

        cond1 = all(
            [
                len(fn_kwargs) == 3,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'rbs' in fn_kwargs,
            ]
        )
        if not cond1:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        batched_request_data = []
        dataloader_len = len(self.train_dataloader)
        train_iter = iter(self.train_dataloader)

        for _ in range(dataloader_len):
            data = next(train_iter)
            batched_request_data.append(data)

        log(f"evaludate info {dataloader_len=} dataset_len {len(self.train_dataset)}")

        sampler_idx = 0
        sample_idx = 0
        curr_step = 0
        repeat_n = self.config.training.eval_sampling_repeat_n
        num_micro_batches_per_request = dataloader_len
        total_request = (
            dataloader_len + num_micro_batches_per_request - 1
        ) // num_micro_batches_per_request
        all_rollout_batches = []

        await self.sampler_client.mark_ppo_step_begin(sampler_idx, curr_step)
        cpu_barrier()

        for request_i in range(total_request):
            log(f"Generating request {request_i} / {total_request}")
            start_index = request_i * num_micro_batches_per_request
            end_index = min((start_index + num_micro_batches_per_request), dataloader_len)
            request_data = batched_request_data[start_index:end_index]
            rbs = await self.sampler_gen_out(
                request_data, sampler_idx, curr_step, sample_idx, repeat_n
            )
            sample_idx += (end_index - start_index) * repeat_n * self.config.training.train_mbs
            all_rollout_batches.extend(rbs)
        cpu_barrier()
        await self.sampler_client.mark_ppo_step_end(sampler_idx, curr_step)
        cpu_barrier()

        expanded_rbs = expand_rollout_batches(all_rollout_batches)
        expected_len = dataloader_len * self.config.training.train_mbs * repeat_n
        assert len(expanded_rbs) == expected_len, f"{len(expanded_rbs)} != {expected_len}"

        evaluate_func(config=self.config, tokenizer=self.tokenizer, rbs=expanded_rbs)
        return True

    async def evaluate(self):
        try:
            await self._evaluate()
        except Exception as e:
            log(f"evaluate error: {e}")
            traceback.print_exc()
            return False
