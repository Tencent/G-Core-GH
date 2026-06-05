import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import ray
from transformers import AutoTokenizer

from gpatch_v4.client import BtRmClient
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.extended_model import ApplySamplingRolloutAttrFactory
from gpatch_v4.orches.data_source import DataSourceBase, RolloutDataSource
from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import (
    AgentLoopActor,
    BaseAgentLoopActor,
)
from gpatch_v4.utils import (
    Envelope,
    check_rollout_batches,
    log,
    logging_memory_usage_details,
    safe_import_class,
)


@dataclass
class GenerateResult:
    """Result of a single rollout collection step.

    Attributes
    ----------
    dp_refs : list
        One ``ObjectRef`` per DP rank with that rank's rollout batches.
    num_aborted_samples : int
        Aborted partial samples (already re-buffered by the controller).
    """

    dp_refs: List = field(default_factory=list)
    num_aborted_samples: int = 0


# TODO 挪到 rollout 目录下
def resolve_agent_loop_actor_cls(config: RlConfig) -> Type[BaseAgentLoopActor]:
    """Return the agent-loop actor class to use.

    If ``config.training.agent_loop_actor_cls`` is set, import and validate
    it as a ``BaseAgentLoopActor`` subclass; otherwise return the built-in
    ``AgentLoopActor``.

    Parameters
    ----------
    config : RlConfig

    Returns
    -------
    type
    """
    cls_path = getattr(config.training, "agent_loop_actor_cls", None)
    if not cls_path:
        return AgentLoopActor

    cls = safe_import_class(cls_path)
    if cls is None:
        raise ImportError(f"Cannot import agent_loop_actor_cls '{cls_path}'")
    if not (isinstance(cls, type) and issubclass(cls, BaseAgentLoopActor)):
        raise TypeError(
            f"agent_loop_actor_cls '{cls_path}' must be a subclass of "
            f"BaseAgentLoopActor, got {cls}"
        )
    return cls


class RolloutController:
    """Single-controller for centralized rollout data management.

    当前只用在 async train 上面才用到。

    Runs as a Ray actor with 0 GPUs. Owns the ``DataSource`` and
    ``BtRmClient``. Per-microbatch work (sampler generation + gen_rm
    scoring) is distributed across a pool of ``AgentLoopActor`` Ray actors
    that are round-robin scheduled on cluster nodes.

    Completed microbatches are deposited into a queue and consumed by
    :meth:`collect_rollout_step` in first-finished order. BT RM is called
    in batch after collection because its interface requires batched
    issue → batched collect.

    Parameters
    ----------
    config : RlConfig
    """
    def __init__(self, config: RlConfig):
        self.config = config
        self.training_config = config.training

        # Will be populated in setup()
        self.bt_rm_client: Optional[BtRmClient] = None
        self.data_source: Optional[DataSourceBase] = None
        self.apply_sampling_rollout_attr = None
        self.agent_loop_actors: List = []
        self.use_colocate = self.config.placement_type == "colocate"

        # Fire-side counters.  These advance when requests are *dispatched*,
        # not when they are consumed, so sliding prefetch (which issues
        # multiple fires before the next collect) never reuses sampler
        # request IDs or resets round-robin back to actor 0.
        #
        # Relationship to ``_sample_idx``: ``_sample_idx`` is the
        # *consumption-side* counter; it advances inside
        # ``collect_rollout_step`` and is used only by bt_rm batching.
        # The two counters are intentionally independent and must not be
        # assumed equal by any caller.
        self._next_fire_sample_idx = 0
        self._next_fire_actor_idx = 0

        self._sample_idx = 0

        # Streaming pipeline state
        self._ready_queue: asyncio.Queue = asyncio.Queue()
        self._inflight_tasks: List[asyncio.Task] = []

    async def setup(self, train_actors: Optional[List] = None):
        """Initialize data source, tokenizer, AgentLoopActor pool, and BT RM client.

        Must be called once after the Ray actor is created.

        Parameters
        ----------
        train_actors : list, optional
            Forwarded to every ``AgentLoopActor`` for memory logging when
            provided under colocate mode.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy.hf_tokenizer_path,
            use_fast=self.training_config.use_fast_tokenizer,
            trust_remote_code=True,
        )

        self.data_source = RolloutDataSource(self.config, self.tokenizer)

        self.apply_sampling_rollout_attr = (
            ApplySamplingRolloutAttrFactory.get_apply_sampling(self.config)
        )

        if self.training_config.use_bt_rm_reward:
            self.bt_rm_client = BtRmClient(self.config, dp_rank=0, dp_size=1)

        dp_size = (
            self.config.policy.dist_config.nnodes *
            self.config.policy.dist_config.num_gpus_per_node // (
                self.config.policy.dist_config.tensor_model_parallel_size *
                self.config.policy.dist_config.pipeline_model_parallel_size *
                getattr(self.config.policy.dist_config, 'context_parallel_size', 1)
            )
        )
        self._dp_size = dp_size
        self._num_microbatches = (
            self.training_config.rollout_gbs // self.training_config.rollout_mbs
        )
        self.rb_multiplier = self.training_config.rb_multiplier

        await self.setup_agent_loop_actors()

        if (self.use_colocate and len(self.agent_loop_actors) > 1):
            await self.setup_colocate_group()

        if train_actors is not None and self.use_colocate:
            await self.set_train_actors(train_actors)

        log(
            f"[RolloutController] setup done: dp_size={dp_size}, "
            f"num_microbatches={self._num_microbatches}, "
            f"rb_multiplier={self.rb_multiplier}, "
            f"num_agent_loop_workers={self.training_config.num_agent_loop_workers}"
        )

    async def setup_agent_loop_actors(self):
        """Create and initialize the AgentLoopActor pool."""
        num_workers = self.training_config.num_agent_loop_workers
        #NOTE(nrwu and xiaotaoliu): 这里 Alive 状态有时候可能不太准，可能会导致少一两个节点，不过因为是纯 cpu 操作，也是能接受的
        node_ids = [
            n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0
        ]
        AgentLoopActorCls = resolve_agent_loop_actor_cls(self.config)
        log(f"[RolloutController] using agent loop actor: {AgentLoopActorCls.__name__}")
        ActorCls = ray.remote(num_cpus=1, num_gpus=0)(AgentLoopActorCls)
        self.agent_loop_actors = []
        setup_futures = []

        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            actor = ActorCls.options(
                name=f"agent_loop_actor_{i}",
                scheduling_strategy=(
                    ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    )
                ),
            ).remote(self.config, i)
            self.agent_loop_actors.append(actor)
            setup_futures.append(actor.setup.remote())

        await asyncio.gather(*setup_futures)

    async def set_train_actors(self, actors: List):
        """Pass policy train actor handles to all agent-loop actors for memory logging."""
        await asyncio.gather(
            *[agent.set_train_actors.remote(actors) for agent in self.agent_loop_actors]
        )

    def get_schedule_info(self) -> dict:
        """Get training schedule computed from the data source.

        Returns
        -------
        dict
            Contains ``ppo_step_per_epoch``, ``total_ppo_step``,
            ``num_train_epoches``.
        """
        dl_len = self.data_source.dataloader_length
        rollout_gas = self._num_microbatches
        ppo_step_per_epoch = dl_len // rollout_gas
        total_ppo_step = ppo_step_per_epoch * self.training_config.num_train_epoches
        return {
            "ppo_step_per_epoch": ppo_step_per_epoch,
            "total_ppo_step": total_ppo_step,
            "num_train_epoches": self.training_config.num_train_epoches,
        }

    # ------------------------------------------------------------------ #
    #  Shared rollout batch helpers
    # ------------------------------------------------------------------ #

    def read_and_clean_batches(self, num_mb: int) -> List[Dict[str, Any]]:
        """Read rollout microbatches and strip attrs not needed by sampler."""
        batches = self.data_source.get_batch(num_mb)
        for rbi, bd in enumerate(batches):
            self._assign_unique_id(bd, rbi)
        return [
            self.apply_sampling_rollout_attr.remove_rollout_attr_before_sampling(bd)
            for bd in batches
        ]

    def partition_batches(
        self,
        cleaned_batches: List[Dict[str, Any]],
        sample_idx_base: int,
        base_actor_idx: int = 0,
    ) -> tuple[List[list], List[list], List[list]]:
        """Partition microbatches by round-robin agent assignment.

        Returns three parallel lists indexed by agent index:
        ``per_agent_batches``, ``per_agent_indices``, and
        ``per_agent_microbatch_indices``.
        """
        num_agent_actors = len(self.agent_loop_actors)
        per_agent_batches: List[list] = [[] for _ in range(num_agent_actors)]
        per_agent_indices: List[list] = [[] for _ in range(num_agent_actors)]
        per_agent_microbatch_indices: List[list] = [[] for _ in range(num_agent_actors)]

        for rbi, cleaned_data in enumerate(cleaned_batches):
            actor_idx = (base_actor_idx + rbi) % num_agent_actors
            per_agent_batches[actor_idx].append(cleaned_data)
            per_agent_indices[actor_idx].append(sample_idx_base + rbi)
            per_agent_microbatch_indices[actor_idx].append(rbi)

        return per_agent_batches, per_agent_indices, per_agent_microbatch_indices

    async def finalize_rollout_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        num_expected: int,
    ) -> GenerateResult:
        """Run shared post-rollout steps and split data by DP rank."""
        if self.training_config.use_bt_rm_reward:
            rbs = await self._bt_rm_reward(rbs, num_expected, ppo_step)

        assert check_rollout_batches(rbs), "rbs format error before split"

        rbs = self.apply_sampling_rollout_attr.add_back_rollout_attr_after_sampling(rbs)
        self.apply_sampling_rollout_attr.clear_data_cache()
        return GenerateResult(dp_refs=self._split_train_data_by_dp(rbs))

    # ------------------------------------------------------------------ #
    #  Streaming generation pipeline
    # ------------------------------------------------------------------ #

    async def fire_generation_requests(self, epoch: int, ppo_step: int, num_ppo_steps: int):
        """Read data and dispatch rollout requests to AgentLoopActors.

        Disaggregated mode keeps the sliding prefetch semantics: each
        microbatch becomes an independent task and results are enqueued in
        first-finished order.

        Colocate single-controller mode uses the same fire/collect API but
        forces ``rollout_max_staleness == 0`` and ``num_ppo_steps == 1``.
        One task dispatches all agent partitions together so all
        ``AgentLoopActor`` instances enter the same gloo barriers.

        Parameters
        ----------
        epoch : int
            Caller must ensure ``ppo_step .. ppo_step + num_ppo_steps - 1``
            all belong to this epoch; spanning epoch boundaries within a
            single call is unsafe because ``maybe_set_epoch`` is consulted
            once at the top.
        ppo_step : int
        num_ppo_steps : int
        """
        self.data_source.maybe_set_epoch(epoch)
        num_mb = self._num_microbatches
        ppo_step_per_epoch = (self.data_source.dataloader_length // num_mb)
        # Single-epoch invariant: a call may not straddle an epoch boundary
        # because ``maybe_set_epoch`` is consulted only once above and
        # ``get_batch`` then reads ``num_mb * num_ppo_steps`` batches
        # sequentially from the dataloader.  The trainer's ``fire_up_to``
        # helper clips at the epoch boundary; this assert catches any
        # caller that forgets to do so.
        last_step = ppo_step + num_ppo_steps - 1
        assert (ppo_step // ppo_step_per_epoch == last_step // ppo_step_per_epoch), (
            f"fire_generation_requests span {ppo_step}..{last_step} "
            f"crosses an epoch boundary (ppo_step_per_epoch="
            f"{ppo_step_per_epoch})"
        )
        if self.use_colocate:
            assert num_ppo_steps == 1, ("single_controller colocate only supports num_ppo_steps=1")
            assert self.config.placement_type == "colocate", (
                "single_controller requires placement_type='colocate'"
            )
            assert self.training_config.rollout_max_staleness == 0, (
                "single_controller requires rollout_max_staleness == 0"
            )
            assert not self._inflight_tasks, (
                "single_controller does not allow overlapping inflight steps"
            )
            assert self._ready_queue.empty(
            ), ("single_controller requires collect before firing the next step")

        agent_actors = self.agent_loop_actors
        num_agent_actors = len(agent_actors)
        total_fired = 0

        curr_sample_idx = self._next_fire_sample_idx
        base_actor_idx = self._next_fire_actor_idx

        for step_offset in range(num_ppo_steps):
            cur_ppo_step = ppo_step + step_offset
            step_sample_idx_base = curr_sample_idx + step_offset * num_mb

            cleaned_batches = self.read_and_clean_batches(num_mb)
            per_agent_batches, per_agent_indices, per_agent_microbatch_indices = (
                self.partition_batches(
                    cleaned_batches,
                    step_sample_idx_base,
                    base_actor_idx=(base_actor_idx + total_fired) % num_agent_actors,
                )
            )

            if self.use_colocate:
                task = asyncio.create_task(
                    self.dispatch_batches_to_agents(
                        agent_actors,
                        per_agent_batches,
                        per_agent_indices,
                        cur_ppo_step,
                    )
                )
                self._inflight_tasks.append(task)
                total_fired += num_mb
            else:
                dispatch_items = []
                for actor_idx, (batches_part, indices_part, mb_indices_part) in enumerate(
                    zip(per_agent_batches, per_agent_indices, per_agent_microbatch_indices)
                ):
                    for cleaned_data, sidx, rbi in zip(
                        batches_part,
                        indices_part,
                        mb_indices_part,
                    ):
                        dispatch_items.append((rbi, actor_idx, cleaned_data, sidx))

                for rbi, actor_idx, cleaned_data, sidx in sorted(dispatch_items):
                    task = asyncio.create_task(
                        self.dispatch_single_item_to_agent(
                            agent_actors[actor_idx],
                            cleaned_data,
                            cur_ppo_step,
                            rbi,
                            sidx,
                        )
                    )
                    self._inflight_tasks.append(task)
                    total_fired += 1

        self._next_fire_sample_idx = curr_sample_idx + num_ppo_steps * num_mb
        self._next_fire_actor_idx = (base_actor_idx + total_fired) % num_agent_actors

        log(
            f"[RolloutController] fired {total_fired} microbatches "
            f"across {num_ppo_steps} steps "
            f"(ppo_step {ppo_step}..{ppo_step + num_ppo_steps - 1})"
        )

    async def dispatch_single_item_to_agent(
        self,
        actor,
        cleaned_data: Dict[str, Any],
        ppo_step: int,
        microbatch_idx: int,
        sample_idx: int,
    ):
        """Dispatch one microbatch to one AgentLoopActor and enqueue results.

        On failure the exception is enqueued so ``collect_rollout_step``
        fails fast instead of dead-locking.
        """
        try:
            rb_list = await actor.agent_loop.remote(
                cleaned_data, ppo_step, microbatch_idx, sample_idx
            )
            for rb in rb_list:
                await self._ready_queue.put(rb)
        except Exception as e:
            traceback.print_exc()
            await self._ready_queue.put(e)
            raise

    async def dispatch_batches_to_agents(
        self,
        agents: List,
        per_agent_batches: List[List[Dict[str, Any]]],
        per_agent_indices: List[List[int]],
        ppo_step: int,
    ):
        """Dispatch a batch of microbatches to all agents and enqueue results.

        All agents are called concurrently via ``asyncio.gather`` so they
        enter gloo barriers in lockstep.
        """
        assert self.use_colocate, "dispatch_batches_to_agents only supports colocate mode"
        try:
            agent_results = await asyncio.gather(
                *[
                    agent.agent_loop.remote(
                        batches_part,
                        ppo_step,
                        [],
                        indices_part,
                        use_colocate=self.use_colocate,
                    ) for agent, batches_part, indices_part in zip(
                        agents,
                        per_agent_batches,
                        per_agent_indices,
                    )
                ]
            )
            for agent_rbs in agent_results:
                for rb in agent_rbs:
                    await self._ready_queue.put(rb)
        except Exception as e:
            traceback.print_exc()
            await self._ready_queue.put(e)
            raise

    async def collect_rollout_step(self, ppo_step: int) -> GenerateResult:
        """Collect the next complete PPO step worth of microbatches.

        Drains ``num_microbatches`` items from the ready queue (in
        first-finished order), restores cached rollout attrs, and splits
        by DP rank.

        Parameters
        ----------
        ppo_step : int
            In async mode this is determined at collection/consumption
            time, not when the microbatch was originally fired.

        Returns
        -------
        GenerateResult

        Raises
        ------
        Exception
            Re-raised from a failed ``AgentLoopActor`` pipeline task.
        """
        num_mb = self._num_microbatches
        num_collect = num_mb * self.rb_multiplier

        rbs = []
        for _ in range(num_collect):
            rb = await self._ready_queue.get()
            if isinstance(rb, Exception):
                raise rb
            rbs.append(rb)

        result = await self.finalize_rollout_batches(rbs, ppo_step, num_collect)
        self._sample_idx += num_mb
        return result

    async def wait_all_inflight(self):
        """Wait for all inflight pipeline tasks to complete.

        Call before ``update_weights`` to guarantee the sampler is idle
        (otherwise a mid-generation weight swap would mix old and new
        weights on the same sample). Ready rollout data in ``_ready_queue``
        is intentionally NOT drained — those batches are already finalized
        under the previous weights and are safe to carry across the update
        boundary.

        On return (including exceptional return), ``_inflight_tasks`` is
        empty; this is asserted as a cross-boundary invariant.
        """
        if self._inflight_tasks:
            try:
                await asyncio.gather(*self._inflight_tasks)
            finally:
                self._inflight_tasks.clear()
        assert not self._inflight_tasks, (
            "wait_all_inflight post-condition: _inflight_tasks must be empty"
        )

    # Convenience single-step wrapper (kept for backward compat)
    async def generate(self, epoch: int, ppo_step: int) -> GenerateResult:
        """Single-step convenience: fire one step and collect."""
        await self.fire_generation_requests(epoch, ppo_step, num_ppo_steps=1)
        result = await self.collect_rollout_step(ppo_step)
        await self.wait_all_inflight()
        return result

    # ------------------------------------------------------------------ #
    #  Batched RM helpers
    # ------------------------------------------------------------------ #

    async def _bt_rm_reward(
        self,
        rbs: List[Dict[str, List[Any]]],
        num_microbatches: int,
        ppo_step: int,
    ) -> List[Dict[str, List[Any]]]:
        """Call BT reward model(s) for a batch of microbatches.

        Unlike gen_rm which runs per-microbatch in the streaming pipeline,
        bt_rm uses a batched issue → batched collect pattern.

        Parameters
        ----------
        rbs : list of dict
            Microbatches (may come from different ppo_steps). Each rb
            carries ``_ppo_step`` and ``_sidx`` metadata tags set by
            ``AgentLoopActor``.
        num_microbatches : int
        ppo_step : int
        """
        assert len(rbs) == num_microbatches
        num_rms = self.bt_rm_client.num_rms
        assert num_rms == 1, f"num_rms should be 1, but got {num_rms}"

        for rm_idx in range(num_rms):
            await self.bt_rm_client.mark_ppo_step_begin(rm_idx, ppo_step)
            # Issue all
            issue_cos = []

            for rbi, rb in enumerate(rbs):
                sidx = self._sample_idx + rbi
                co = self.bt_rm_client.issue_bt_rm(rb, rm_idx, ppo_step, sidx)
                issue_cos.append(co)
            await asyncio.gather(*issue_cos)

            # Collect all
            result_cos = []

            for rbi, rb in enumerate(rbs):
                sidx = self._sample_idx + rbi
                co = self.bt_rm_client.get_bt_rm_result(rm_idx, ppo_step, sidx)
                result_cos.append(co)
            bt_rm_resp_dicts = await asyncio.gather(*result_cos)

            await self.bt_rm_client.mark_ppo_step_end(rm_idx, ppo_step)
            for rb, resp_dict in zip(rbs, bt_rm_resp_dicts):
                rb.update(resp_dict)
            assert check_rollout_batches(rbs), "rbs format error after bt_rm"
        return rbs

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def add_partial_samples(self, samples: List[Dict[str, Any]]):
        """Feed aborted partial samples back into the data source buffer."""
        self.data_source.add_partial_samples(samples)

    def _assign_unique_id(self, batched_data: Dict[str, Any], rbi: int):
        """Assign a unique ID to each sample in the batch."""
        if "unique_id" not in batched_data:
            first_key = list(batched_data.keys())[0]
            batch_size = len(batched_data[first_key])
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_id_list = [f"ctrl_rbi_{rbi}_batch_id_{bid}_{ts}" for bid in range(batch_size)]
            batched_data["unique_id"] = unique_id_list

    def _split_train_data_by_dp(self, rbs: List[Dict[str, List[Any]]]) -> List[ray.ObjectRef]:
        """Split rollout batches across DP ranks and ``ray.put`` each shard.

        Round-robin: micro-batch ``i`` goes to DP rank ``i % dp_size``.

        Parameters
        ----------
        rbs : list of dict

        Returns
        -------
        list[ray.ObjectRef]
        """
        dp_size = self._dp_size
        partitions: List[List[Dict]] = [[] for _ in range(dp_size)]

        for i, rb in enumerate(rbs):
            partitions[i % dp_size].append(rb)

        refs = []
        for dp_rank in range(dp_size):
            ref = ray.put(partitions[dp_rank])
            refs.append(Envelope(ref))

        log(f"[RolloutController] split {len(rbs)} micro-batches across "
            f"{dp_size} DP ranks")
        return refs

    # ------------------------------------------------------------------ #
    #  Colocate group bootstrap
    # ------------------------------------------------------------------ #

    async def setup_colocate_group(self):
        """Bootstrap a gloo process group among all ``agent_loop_actors``.

        Agent 0 is the master; its IP and a free port are discovered via
        remote calls, then all agents concurrently join the group.
        """
        agents = self.agent_loop_actors
        n = len(agents)
        if n <= 1:
            return

        master_ip = await agents[0].get_node_ip.remote()
        master_port = await agents[0].find_free_port.remote()

        log(
            f"[RolloutController] bootstrapping colocate gloo group: "
            f"n={n}, master={master_ip}:{master_port}"
        )

        await asyncio.gather(
            *[
                agent.init_agent_group.remote(i, n, master_ip, master_port)
                for i, agent in enumerate(agents)
            ]
        )

        log(f"[RolloutController] colocate gloo group ready ({n} agents)")
