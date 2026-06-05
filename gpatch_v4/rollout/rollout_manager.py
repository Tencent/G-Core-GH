import asyncio
import math
from datetime import timedelta

import numpy as np
import torch

from megatron.core import parallel_state as mpu

from gpatch.core.parallel_state import is_mp_head
from gpatch.core.utils import gen_unique_id
from gpatch_v4.rollout.request_group import RolloutRequestGroup

#TODO(hessainliu): remove init_pg and cpu_barrier when integrated with v4
_GROUP_GLOO = None


def init_pg(distributed_timeout_minutes: int = 30):
    """Initialize a gloo process group for CPU barriers.

    Parameters
    ----------
    distributed_timeout_minutes : int, optional
    """
    timeout = timedelta(minutes=distributed_timeout_minutes)
    global _GROUP_GLOO
    world_size = torch.distributed.get_world_size()
    ranks = np.arange(world_size)
    _GROUP_GLOO = torch.distributed.new_group(ranks=ranks, timeout=timeout, backend='gloo')


def cpu_barrier():
    """Synchronize all ranks using the gloo process group."""
    from gpatch.core.parallel_state import cpu_barrier as gpatch_cpu_barrier
    global _GROUP_GLOO
    gpatch_cpu_barrier(_GROUP_GLOO)


def _dp_allgather(obj):
    group = mpu.get_data_parallel_group()
    obj_list = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.all_gather_object(obj_list, obj, group=group)
    return obj_list


class RolloutManager:
    """Manage rollout generation, including partial rollout and DP balancing.

    Parameters
    ----------
    sampler_client : object
        Sampler inference engine client.
    coordinator_client : object
        Rollout coordinator client.
    """
    def __init__(self, sampler_client, coordinator_client):
        self.sampler_client = sampler_client
        self.coordinator_client = coordinator_client

        # for partial rollout
        self.data_buffer = []
        self.partial_buffer = []
        self.done_buffer = []
        self.prefill_data_buffer = False

    def get_sample(self, data_iter):
        """Fetch one sample from ``data_iter`` and wrap it as a request group.

        Parameters
        ----------
        data_iter : iterator
            Yields batched data dicts.

        Returns
        -------
        RolloutRequestGroup
        """
        data = next(data_iter)
        json_data_list = data['json_data_list']
        tokens = data["input_ids"]
        assert tokens.shape[
            0] == 1 and len(json_data_list) == 1, "--ppo-rollout-micro-batch-size must be 1"
        image_grid_thw = data["image_grid_thw"]
        position_ids = data["position_ids"]
        prompt_len = data["prompt_len"]
        image_input_mask = data["image_input_mask"]
        pixel_values = None
        if "pixel_values" in data:
            pixel_values = data["pixel_values"].type(torch.bfloat16)
        tokens_for_gen = data.get("input_ids_for_gen", [])

        cache_keys = [
            "position_ids",
            "vision_grid_thw",
            "image_input_mask",
            "vision_data",
        ]

        batch_data = dict(
            # type is list
            unique_id=[gen_unique_id()],
            json_data_list=json_data_list,
            tokens=[tokens.squeeze(0)],
            prompt_len=[prompt_len],
            tokens_for_gen=tokens_for_gen,
            imgs_np_array_list=data["imgs_np_array_list"],
            # do not picked for rollout, type is tensor
            position_ids=position_ids,
            vision_grid_thw=image_grid_thw,
            image_input_mask=image_input_mask,
            vision_data=pixel_values,
        )
        request_group = RolloutRequestGroup(batch_data=batch_data, cache_keys=cache_keys)
        # TODO(hessianliu): support repeats
        request_group.build_repeats(1)
        return request_group

    def generate(self, data_iter, batch_size, get_samlpe_func=None):
        """Generate rollouts synchronously in one shot.

        Parameters
        ----------
        data_iter : iterator
        batch_size : int
            Samples per DP rank.
        get_samlpe_func : callable, optional

        Returns
        -------
        list of RolloutRequestGroup or None
            Completed request groups (only on MP head).
        """

        get_samlpe_func = self.get_sample if get_samlpe_func is None else get_samlpe_func

        def get_batch(data_iter, batch_size):
            batch = []
            for _ in range(batch_size):
                batch.append(get_samlpe_func(data_iter))
            return batch

        cpu_barrier()
        batch = None
        if is_mp_head():
            batch = get_batch(data_iter, batch_size)
            batch_size_list = _dp_allgather(batch_size)
            total_batch_size = sum(batch_size_list)
            if torch.distributed.get_rank() == 0:
                self.sampler_client.resume()
                # notify coordinator
                self.coordinator_client.start_step(total_batch_size)

        cpu_barrier()
        if is_mp_head():

            async def batch_generate(batch):
                tasks = []
                for (i, e) in enumerate(batch):
                    engine_index = i % self.sampler_client.num_engine
                    task = self.sampler_client.generate(e, engine_index)
                    tasks.append(task)
                return await asyncio.gather(*tasks)

            results = asyncio.run(batch_generate(batch))
            assert len(results) == len(batch), f"{len(results)} vs {len(batch)}"
            for (req, rep) in zip(batch, results):
                req.update(rep)
        cpu_barrier()
        return batch
        #TODO(hessianliu): broadcast accross mp

    def _dp_balance_plan(self, done_buffer, batch_size):
        """Compute a DP balancing plan to equalize done counts.

        Parameters
        ----------
        done_buffer : list
        batch_size : int

        Returns
        -------
        tuple[int, list[int]]
            ``(total_moves, give_ups)`` where ``give_ups[rank] > 0`` gives
            samples and ``< 0`` takes.
        """

        done_buffer_size = len(done_buffer)
        done_buffer_size_list = _dp_allgather(done_buffer_size)

        dp_wz = len(done_buffer_size_list)
        # negetive for take, positive for gave
        give_ups = [0] * dp_wz
        # left over sample
        done_buffer_size_list_left = [e - batch_size for e in done_buffer_size_list]
        # total num of move
        moves = sum([-e for e in done_buffer_size_list_left if e < 0])

        if moves == 0:
            return moves, give_ups

        rank_sizes = list(enumerate(done_buffer_size_list_left))
        # print(f"rank_sizes {rank_sizes}", flush=True)
        # stable sort, consist cross dp
        sorted_rank_sizes = sorted(rank_sizes, key=lambda x: x[1])
        # build stages
        stages = []
        pre = None
        for (i, e) in enumerate(sorted_rank_sizes):
            e = e[1]
            if e != pre:
                stages.append([i, e])
                pre = e

        # eliminate mountain
        moved = 0
        while moved < moves:
            left = stages[-1][0]
            top = stages[-1][1]
            second = stages[-2][1]
            level_diff = top - second
            if (moves - moved) >= level_diff * (dp_wz - left):
                stages.pop()
                moved = moved + level_diff * (dp_wz - left)
            else:
                # move stops here
                level = (moves - moved) // (dp_wz - left)
                level_residul = (moves - moved) % (dp_wz - left)

                if level_residul:
                    stages[-1][1] = top - level - 1
                    stages.append([left + level_residul, top - level])
                else:
                    stages[-1][1] = top - level
                moved = moves

        def get_stage_start(index):
            if index >= len(stages):
                return dp_wz
            return stages[index][0]

        next_stage_index = 0
        for (i, e) in enumerate(sorted_rank_sizes):
            if get_stage_start(next_stage_index) <= i:
                next_stage_index = next_stage_index + 1
            rank, left_load = e
            new_load = stages[next_stage_index - 1][1]
            if left_load != new_load:
                assert left_load >= 0

            if left_load < 0:
                give_ups[rank] = left_load
                assert left_load == new_load
            else:
                assert left_load >= new_load
                give_ups[rank] = left_load - new_load

        return moves, give_ups

    def _dp_balance_impl(self, data_buffer, moves, plan):
        dp_rank = mpu.get_data_parallel_rank()
        rank_action = plan[dp_rank]
        gives = []
        # gives
        if rank_action > 0:
            gives = data_buffer[-rank_action:]
            data_buffer = data_buffer[:-rank_action]
        gives_list = _dp_allgather(gives)
        full_gives = [e for l in gives_list for e in l]
        # takes:
        if rank_action < 0:
            left_takes = sum([-e if e < 0 else 0 for e in plan][:dp_rank])
            takes = full_gives[left_takes:left_takes - rank_action]
            data_buffer = data_buffer + takes

        return data_buffer

    def _dp_balance(self, done_buffer, batch_size):
        moves, plan = self._dp_balance_plan(done_buffer, batch_size)
        data_buffer = self._dp_balance_impl(done_buffer, moves, plan)
        return data_buffer

    def generate_with_partial_rollout(self, data_iter, batch_size, get_samlpe_func=None):
        """Generate rollouts with partial-rollout support and DP balancing.

        Parameters
        ----------
        data_iter : iterator
        batch_size : int
            Samples per DP rank.
        get_samlpe_func : callable, optional

        Returns
        -------
        list of RolloutRequestGroup or None
            Completed request groups (only on MP head).
        """

        get_samlpe_func = self.get_sample if get_samlpe_func is None else get_samlpe_func

        def fill_data_buffer(data_buffer, data_iter, batch_size):
            for _ in range(batch_size):
                data_buffer.append(get_samlpe_func(data_iter))

        cpu_barrier()
        if is_mp_head():
            if not self.prefill_data_buffer:
                #TODO(hessianiu): make oversample ratio a config
                fill_data_buffer(self.data_buffer, data_iter, math.ceil(batch_size * 0.2))
                self.prefill_data_buffer = True
            fill_data_buffer(self.data_buffer, data_iter, batch_size)
            done_size = len(self.done_buffer)
            done_size_list = _dp_allgather(done_size)
            total_done_size = sum(done_size_list)
            short_size = batch_size * len(done_size_list) - total_done_size
            if torch.distributed.get_rank() == 0:
                self.sampler_client.resume()
                self.coordinator_client.start_step(short_size)

        cpu_barrier()
        out_batch = None
        if is_mp_head():
            self.data_buffer, data_buffer = [], self.data_buffer
            self.partial_buffer, partial_buffer = [], self.partial_buffer
            batch = partial_buffer + data_buffer

            async def generate(sample, engine_index):
                res = await self.sampler_client.generate(sample, engine_index)
                if not res.is_aborted():
                    import uuid
                    request_id = str(uuid.uuid1())
                    await self.coordinator_client.report_rollout(request_id)
                else:
                    print("generate aborted ", flush=True)
                return res

            async def batch_generate(batch):
                tasks = []
                for (i, e) in enumerate(batch):
                    engine_index = i % self.sampler_client.num_engine
                    task = generate(e, engine_index)
                    tasks.append(task)
                return await asyncio.gather(*tasks)

            results = asyncio.run(batch_generate(batch))
            assert len(results) == len(batch), f"{len(results)} vs {len(batch)}"
            for (req, rep) in zip(batch, results):
                req.update(rep)

            done_buffer = self.done_buffer
            for e in batch:
                if not e.is_done():
                    self.partial_buffer.append(e)
                else:
                    done_buffer.append(e)

            data_for_train = len(done_buffer)
            data_for_train_list = _dp_allgather(data_for_train)
            total_data_for_train = sum(data_for_train_list)
            assert total_data_for_train >= batch_size * len(
                data_for_train_list
            ), f"total_data_for_train {total_data_for_train} vs {batch_size * len(data_for_train_list)}"
            # dp 间均衡
            done_buffer = self._dp_balance(done_buffer, batch_size)
            assert len(done_buffer) >= batch_size
            out_batch, self.done_buffer = done_buffer[:batch_size], done_buffer[batch_size:]
            #TODO(hessianliu): broadcast accross mp
        cpu_barrier()
        return out_batch

    def generate_with_async_consitent_version(self, data_iter, batch_size, get_samlpe_func=None):
        """Placeholder for async consistent-version generation (not implemented)."""
        pass
