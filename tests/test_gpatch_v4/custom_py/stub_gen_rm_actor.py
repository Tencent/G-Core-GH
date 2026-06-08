import copy

import torch

from gpatch_v4.orches.base_actor import RayBaseActor


class StubGrpoGenRmActor(RayBaseActor):
    """Lightweight stub actor for integration testing without sglang.

    Implements the same interface as GrpoGenRmActor but returns
    deterministic fake rewards instead of running real inference.
    """
    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port()
        self._is_master_node = None
        self.rm_idx = None
        self.config = None
        self.configured = False
        self.engine_started = False
        self.start_count = 0
        self.stop_count = 0

    async def init(self, config):
        self.config = config

    async def init_infer_engine(
        self,
        config,
        dist_init_addr,
        rm_idx,
        engine_idx,
        tp_rank,
        is_master_node,
        pg_bundle_indices=None,
    ):
        del dist_init_addr, engine_idx, tp_rank, pg_bundle_indices
        self.config = config
        self._is_master_node = is_master_node
        self.rm_idx = rm_idx
        self.configured = True
        if not config.gen_rm.destroy_engine_after_generation:
            await self.ensure_infer_engine({})

    async def ensure_infer_engine(self, req_dict=None):
        del req_dict
        assert self.configured
        if not self.engine_started:
            self.engine_started = True
            self.start_count += 1
        return {"ret": True}

    async def destroy_infer_engine(self, req_dict=None):
        del req_dict
        if self.engine_started:
            self.engine_started = False
            self.stop_count += 1
        return {"ret": True}

    async def get_lifecycle_state(self):
        return {
            "configured": self.configured,
            "engine_started": self.engine_started,
            "start_count": self.start_count,
            "stop_count": self.stop_count,
        }

    async def sleep(self, req_dict):
        del req_dict
        self.engine_started = False
        return {"ret": True}

    async def wake_up(self, req_dict):
        del req_dict
        await self.ensure_infer_engine({})
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict):
        del req_dict
        if self.config.gen_rm.destroy_engine_after_generation:
            await self.ensure_infer_engine({})
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict):
        del req_dict
        if self.config.gen_rm.destroy_engine_after_generation:
            await self.destroy_infer_engine({})
        return {"ret": True}

    async def generate_rewards(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        assert self.engine_started
        batched_data = req_dict["batched_data"]
        n = len(batched_data["prompt"])
        ret = copy.deepcopy(batched_data)
        ret[f"reward_gen_rm_{self.rm_idx}"] = [
            torch.tensor(1.0 / (self.rm_idx + 1)) for _ in range(n)
        ]
        return ret

    async def flush_cache(self, req_dict):
        return {"ret": True}
