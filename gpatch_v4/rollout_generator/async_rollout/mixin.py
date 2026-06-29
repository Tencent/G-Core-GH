"""Colocate lifecycle mixin for agent-loop actors.

Provides GPU lifecycle coordination (wake/sleep sampler & gen-RM),
gloo barrier synchronisation, and memory logging helpers used by
colocate-mode agent actors.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import List

import torch.distributed as dist

from gpatch_v4.orches.utils import find_free_port, get_current_node_ip
from gpatch_v4.utils import log


class ColocateAgentMixin:
    """Mixin that adds colocate lifecycle coordination to agent actors.

    Expects the consuming class to provide:
    - ``self.sampler_client`` — a ``SamplerClient`` instance
    - ``self.gen_rm_client`` — a ``GenRmClient`` instance (when gen-RM is used)
    - ``self.use_colocate`` — bool flag
    - ``self.agent_rank`` — int, rank in the gloo group
    - ``self.agents_pg`` — gloo process group (or None)
    - ``self._train_actors`` — list of policy train actor handles
    """
    def _init_colocate_state(self):
        """Initialise colocate-related instance attributes.

        Call this from the consuming class's ``__init__``.
        """
        self.agent_rank: int = 0
        self.agents_pg = None
        self.use_colocate = False
        self._train_actors: List = []

    # ------------------------------------------------------------------ #
    #  Network helpers (needed by controller via .remote())
    # ------------------------------------------------------------------ #

    def get_node_ip(self) -> str:
        """Return the IP address of the current node.

        Resolution order: ``__HOST_IP__`` env var → ``bond1`` → ``eth0`` → 127.0.0.1.
        """
        return get_current_node_ip()

    def find_free_port(self, start_port: int = 29600) -> int:
        """Find a free TCP port on this node, scanning from *start_port*."""
        return find_free_port(start_port)

    # ------------------------------------------------------------------ #
    #  Colocate gloo group bootstrap
    # ------------------------------------------------------------------ #

    async def init_agent_group(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        """Initialise a gloo process group for colocate barrier sync.

        Called by the controller after all agents are set up.  All ranks
        must call this concurrently (the controller issues ``asyncio.gather``
        over all remote calls).
        """
        self.agent_rank = rank

        def _init_pg():
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
            )
            return dist.group.WORLD

        self.agents_pg = _init_pg()
        self.use_colocate = True
        log(
            f"[{type(self).__name__}-{self.worker_id}] colocate gloo group ready: "
            f"rank={rank}/{world_size}"
        )

    # ------------------------------------------------------------------ #
    #  Colocate lifecycle helpers
    # ------------------------------------------------------------------ #

    @property
    def is_agent_leader(self) -> bool:
        """True when this agent should issue lifecycle RPCs."""
        return self.agents_pg is None or self.agent_rank == 0

    async def agent_cpu_barrier(self):
        """No-op when no group; blocking gloo barrier otherwise."""
        if self.agents_pg is None:
            return
        await asyncio.to_thread(dist.barrier, self.agents_pg)

    def set_train_actors(self, actors: List):
        """Store references to policy train actors for memory logging."""
        self._train_actors = actors

    async def _log_policy_memory(self, tag: str):
        """Call ``log_memory`` on all policy train actors (colocate only)."""
        if not self._train_actors or not self.use_colocate:
            return
        if not self.is_agent_leader:
            return
        futs = [actor.log_memory.remote(tag=tag) for actor in self._train_actors]
        await asyncio.gather(*futs)

    @asynccontextmanager
    async def sampler_phase(self, ppo_step: int, sampler_idx: int = 0):
        """Wake sampler (leader only) → barrier → yield → barrier → flush + sleep (leader only) → barrier."""
        if self.is_agent_leader:
            await self.sampler_client.mark_ppo_step_begin(sampler_idx, ppo_step)
        await self.agent_cpu_barrier()
        await self._log_policy_memory(f"memory tracking after sampler {sampler_idx} wake_up")
        await self.agent_cpu_barrier()
        try:
            yield
        finally:
            await self.agent_cpu_barrier()
            if self.is_agent_leader:
                await self.sampler_client.infer_engine_flush_cache(sampler_idx)
                await self.sampler_client.mark_ppo_step_end(sampler_idx, ppo_step)
            await self.agent_cpu_barrier()
            await self._log_policy_memory(f"memory tracking after sampler {sampler_idx} sleep")
            await self.agent_cpu_barrier()

    @asynccontextmanager
    async def gen_rm_phase(self, rm_idx: int, ppo_step: int):
        """Wake gen_rm[rm_idx] (leader only) → barrier → yield → barrier → sleep (leader only) → barrier."""
        if self.is_agent_leader:
            await self.gen_rm_client.mark_ppo_step_begin(rm_idx, ppo_step)
        await self.agent_cpu_barrier()
        await self._log_policy_memory(f"memory tracking after gen_rm {rm_idx} wake_up")
        await self.agent_cpu_barrier()
        try:
            yield
        finally:
            await self.agent_cpu_barrier()
            if self.is_agent_leader:
                await self.gen_rm_client.mark_ppo_step_end(rm_idx, ppo_step)
            await self.agent_cpu_barrier()
            await self._log_policy_memory(f"memory tracking after gen_rm {rm_idx} sleep")
            await self.agent_cpu_barrier()

    @asynccontextmanager
    async def gen_rm_all_phase(self, ppo_step: int):
        """Wake all gen_rm actors together, then sleep them together."""
        num_rms = self.gen_rm_client.num_rms
        if self.is_agent_leader:
            await asyncio.gather(
                *[
                    self.gen_rm_client.mark_ppo_step_begin(rm_idx, ppo_step=ppo_step)
                    for rm_idx in range(num_rms)
                ]
            )
        await self.agent_cpu_barrier()
        await self._log_policy_memory("memory tracking after all gen_rm wake_up")
        await self.agent_cpu_barrier()
        try:
            yield
        finally:
            await self.agent_cpu_barrier()
            if self.is_agent_leader:
                await asyncio.gather(
                    *[
                        self.gen_rm_client.mark_ppo_step_end(rm_idx, ppo_step=ppo_step)
                        for rm_idx in range(num_rms)
                    ]
                )
            await self.agent_cpu_barrier()
            await self._log_policy_memory("memory tracking after all gen_rm sleep")
            await self.agent_cpu_barrier()
