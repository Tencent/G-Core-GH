import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import ray
from transformers import AutoTokenizer

from gpatch_v4.client import BtRmClient, GenRmClient, SamplerClient
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.dynamic_batch_reward_normalize import (
    prepare_sample_masks,
    reward_normalize,
    uses_group_reward_normalization,
)
from gpatch_v4.extended_model import ApplySamplingRolloutAttrFactory
from gpatch_v4.orches.data_source import DataSourceBase, get_data_source
from gpatch_v4.orches.utils import build_actor_env_vars
from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import (
    AgentLoopActor,
    BaseAgentLoopActor,
)
from gpatch_v4.utils import (
    Envelope,
    GenerationAborted,
    check_rollout_batches,
    import_fn_from_path,
    log,
    logging_memory_usage_details,
    safe_import_class,
)
from gpatch_v4.utils.dynamic_batch_utils import (
    TRAIN_STEP_ID_KEY,
    assign_train_steps,
    compute_rollout_metrics,
    convert_samples_to_train_data,
    split_train_steps_by_dp,
    validate_dynamic_batch_samples,
)
from gpatch_v4.utils.filter_samplings import filter_rollout_samples
from gpatch_v4.utils.placement import is_colocate, is_partial_colocated
from gpatch_v4.utils.training_utils import expand_rollout_batches

ORIGIN_PPO_STEP_KEY = "_origin_ppo_step"
ORIGIN_MICROBATCH_IDX_KEY = "_origin_microbatch_idx"


@dataclass
class PendingMicrobatch:
    """A microbatch fired during the current tail-batching window.

    Keyed by ``(ppo_step, microbatch_idx)``, created when the controller
    reads the microbatch and dropped by the next :meth:`begin_step`.

    ``rollout_consumed`` and ``train_consumed`` are not interchangeable: a
    microbatch can be accepted into the fire window (rollout) while its
    result still sits in the ready queue waiting for the trainer (train).
    Reclaim uses the former.  Checkpoint unconsumed-batch accounting uses
    the latter, and only when ``rollout_reuse_unused_prompts`` is on.

    Attributes
    ----------
    raw : dict
        Shallow copy of the batched data taken *before*
        ``remove_rollout_attr_before_sampling`` mutates it, so an unused
        prompt can be handed back to the data source unchanged.
    rollout_consumed : bool
        ``True`` once ``_try_accept_completed_microbatch`` accepts this microbatch.
        ``False`` at the next ``begin_step`` means reclaim the prompt.
    train_consumed : bool
        ``True`` once the trainer dequeues this microbatch from the ready
        queue.  ``False`` at checkpoint time means its prompt still counts as
        unconsumed even if rollout already accepted it.
    """

    raw: Dict[str, Any]
    rollout_consumed: bool = False
    train_consumed: bool = False


@dataclass
class GenerateResult:
    """Result of a single rollout collection step.

    Attributes
    ----------
    dp_refs : list
        One ``ObjectRef`` per DP rank with that rank's rollout batches.
    num_aborted_samples : int
        Aborted partial samples (already re-buffered by the controller).
    metrics : dict
        Rollout metrics from the controller, including scalars and
        ``*_histogram`` entries.
    """

    dp_refs: List = field(default_factory=list)
    num_aborted_samples: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueuedRolloutBatch:
    """Controller-internal queue item with ordering metadata."""

    rollout_batch: Dict[str, List[Any]]
    ppo_step: int
    microbatch_idx: int
    sample_idx: int


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
    :meth:`collect_rollout_step` in first-finished order by default.  When
    ``training.rollout_ordered_collection`` is enabled, collection waits
    for the requested original PPO step and emits microbatches in original
    order.  BT RM is called in batch after collection because its interface
    requires batched issue → batched collect.

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
        self.use_colocate = is_colocate(self.config)
        self.is_partial_colocated = is_partial_colocated(self.config)

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
        self._next_prompt_idx = 0

        self._sample_idx = 0

        # Streaming pipeline state
        self._ready_queue: asyncio.Queue = asyncio.Queue()
        self._ordered_ready: Dict[int, Dict[int, List[QueuedRolloutBatch]]] = {}
        self._inflight_tasks: List[asyncio.Task] = []

        # Tail-batching state.  ``_tail_batching`` gates every code path below;
        # when it is False the controller behaves exactly as it did before
        # tail-batching existed.
        self._tail_batching = self.training_config.rollout_over_dispatch_ratio > 1.0
        self._sampler_client: Optional[SamplerClient] = None
        self._gen_rm_client: Optional[GenRmClient] = None
        self._pending: Dict[Tuple[int, int], PendingMicrobatch] = {}
        self._step_target_rb = 0
        self._step_collected_rb = 0
        self._pause_task: Optional[asyncio.Task] = None
        self._abort_t0: Optional[float] = None
        self._last_inflight_done_at: Optional[float] = None
        self._fire_metrics: Dict[int, Dict[str, float]] = {}
        self._discard_hook: Optional[Callable[[List], None]] = None

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

        self.data_source = get_data_source(self.config, self.tokenizer)
        if self._tail_batching:
            # Colocate hands a whole partition to each agent and only reports
            # once it is fully generated, so the collection target cannot be
            # honoured at microbatch granularity there.
            assert not self.use_colocate, (
                "rollout_over_dispatch_ratio > 1.0 requires per-microbatch "
                "dispatch (partial_colocated or disaggregated)"
            )
            self._next_fire_sample_idx = self.data_source.total_dl_consumed
            # Pausing means aborting in-flight sampler requests; the
            # controller talks to the sampler directly so agent classes stay
            # untouched.
            self._sampler_client = SamplerClient(
                self.config, dp_rank=0, dp_size=1, skip_init_ipc_meta=True
            )
            if self.training_config.use_gen_rm_reward:
                self._gen_rm_client = GenRmClient(self.config, dp_rank=0, dp_size=1)
            if self.training_config.tail_batching_discard_py_path is not None:
                self._discard_hook = import_fn_from_path(
                    self.training_config.tail_batching_discard_py_path,
                    self.training_config.tail_batching_discard_fn_name,
                )

        self.apply_sampling_rollout_attr = (
            ApplySamplingRolloutAttrFactory.get_apply_sampling(self.config)
        )
        self.custom_reward_normalize = None
        if self.config.ppo.custom_reward_normalize_py_path is not None:
            self.custom_reward_normalize = import_fn_from_path(
                self.config.ppo.custom_reward_normalize_py_path,
                self.config.ppo.custom_reward_normalize_py_name,
            )
        self.custom_filter_samples = None
        if self.training_config.ppo_filter_samplings_path is not None:
            self.custom_filter_samples = import_fn_from_path(
                self.training_config.ppo_filter_samplings_path,
                self.training_config.ppo_filter_samplings_name,
            )
        self.custom_convert_samples_to_train_data = None
        if self.training_config.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data = import_fn_from_path(
                self.training_config.custom_convert_samples_to_train_data_path,
                self.training_config.custom_convert_samples_to_train_data_name,
            )
        self.custom_compute_rollout_metrics = None
        if self.training_config.custom_compute_rollout_metrics_path is not None:
            self.custom_compute_rollout_metrics = import_fn_from_path(
                self.training_config.custom_compute_rollout_metrics_path,
                self.training_config.custom_compute_rollout_metrics_name,
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
        self._ppo_step_per_epoch = (self.data_source.dataloader_length // self._num_microbatches)
        self.rb_multiplier = self.training_config.rb_multiplier

        await self.setup_agent_loop_actors()

        if self.use_colocate:
            if len(self.agent_loop_actors) > 1:
                await self.setup_colocate_group()
            if train_actors is not None:
                await self.set_train_actors(train_actors)

        log(
            f"[RolloutController] setup done: dp_size={dp_size}, "
            f"num_microbatches={self._num_microbatches}, "
            f"rb_multiplier={self.rb_multiplier}, "
            f"num_agent_loop_workers={self.training_config.num_agent_loop_workers}"
        )

    @asynccontextmanager
    async def partial_colocated_rollout(self, ppo_step: int):
        """Keep sampler and gen-RMs awake while one PPO step is running."""
        assert self.agent_loop_actors, "agent_loop_actors must be initialized"
        await self.agent_loop_actors[0].begin_partial_colocated_rollout.remote(ppo_step)
        try:
            yield
        finally:
            await self.agent_loop_actors[0].end_partial_colocated_rollout.remote(ppo_step)

    async def setup_agent_loop_actors(self):
        """Create and initialize the AgentLoopActor pool."""
        num_workers = self.training_config.num_agent_loop_workers
        #NOTE(nrwu and xiaotaoliu): 这里 Alive 状态有时候可能不太准，可能会导致少一两个节点，不过因为是纯 cpu 操作，也是能接受的
        node_ids = [
            n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0
        ]
        AgentLoopActorCls = resolve_agent_loop_actor_cls(self.config)
        log(f"[RolloutController] using agent loop actor: {AgentLoopActorCls.__name__}")
        ActorCls = ray.remote(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": build_actor_env_vars()},
        )(AgentLoopActorCls)
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
        ppo_step_per_epoch = self._ppo_step_per_epoch
        total_ppo_step = ppo_step_per_epoch * self.training_config.num_train_epoches
        return {
            "ppo_step_per_epoch": ppo_step_per_epoch,
            "total_ppo_step": total_ppo_step,
            "num_train_epoches": self.training_config.num_train_epoches,
        }

    # ------------------------------------------------------------------ #
    #  aborting generation and recycling unused prompts
    # ------------------------------------------------------------------ #

    async def begin_step(self, target_mb: int):
        """Open a tail-batching fire window.

        Called by the trainer immediately before ``fire_generation_requests``.
        Reclaims prompts whose generation was dropped in the previous window,
        resets per-window bookkeeping, and pins how many rollout batches this
        window collects before generation is paused.

        Parameters
        ----------
        target_mb : int
            Microbatches this window must yield, summed over the PPO steps
            the following fire call covers.  Must be ``> 0``; zero would make
            every arrival look excess and hang ``collect_rollout_step``.
            ``rb_multiplier`` is applied here to derive the rollout-batch
            target.
        """
        if not self._tail_batching:
            self._step_target_rb = target_mb * self.rb_multiplier
            self._step_collected_rb = 0
            return
        assert target_mb > 0, (f"begin_step: target_mb must be positive, got {target_mb}")
        # Clearing ``_pause_task`` below must not race late GenerationAborted
        # handlers; the trainer drains via wait_all_inflight first.
        assert not self._inflight_tasks, (
            "begin_step: inflight tasks remain; call wait_all_inflight first"
        )
        # Let a pending pause finish first, otherwise a late-arriving
        # microbatch could flip ``rollout_consumed`` between the reclaim scan
        # and the clear below, silently dropping its prompt.
        if self._pause_task is not None:
            await self._pause_task
            self._pause_task = None

        discarded = [p for p in self._pending.values() if not p.rollout_consumed]
        if discarded:
            # The hook sees raw prompts before ``_release_discarded_attrs``
            # pops their unique_ids, and runs whether or not reuse is on.
            if self._discard_hook is not None:
                self._discard_hook([p.raw for p in discarded])
            self._release_discarded_attrs(discarded)
            if self.training_config.rollout_reuse_unused_prompts:
                self.data_source.add_partial_samples([p.raw for p in discarded])
                log(f"[RolloutController] reclaimed {len(discarded)} unused microbatches")

        self._pending.clear()
        self._step_target_rb = target_mb * self.rb_multiplier
        self._step_collected_rb = 0
        await asyncio.gather(*[a.begin_step.remote() for a in self.agent_loop_actors])
        log(
            f"[RolloutController] begin_step: target_mb={target_mb}, "
            f"target_rb={self._step_target_rb}"
        )

    def _release_discarded_attrs(self, discarded: List[PendingMicrobatch]) -> None:
        """Retire the sampling-attr entries of microbatches that were never collected.

        ``clear_data_cache`` only evicts ids that made it through
        ``add_back_rollout_attr_after_sampling``; over-fired microbatches never
        do, so their entries would accumulate for the whole run and collide
        with themselves once a reclaimed prompt is fired again.  Dropping
        ``unique_id`` from the reclaimed copy also makes the next window mint a
        fresh id, which is what a re-fired prompt should get.
        """
        cache = self.apply_sampling_rollout_attr.cached_rollout_attrs()
        for pending in discarded:
            for unique_id in pending.raw["unique_id"]:
                assert unique_id in cache, f"sampling attr cache lost {unique_id=}"
                del cache[unique_id]
            self._strip_controller_dispatch_attrs(pending.raw)

    def _try_accept_completed_microbatch(
        self,
        rb_list: List[Dict[str, List[Any]]],
        ppo_step: int,
        microbatch_idx: int,
    ) -> bool:
        """Claim collection slots for one finished microbatch.

        The whole ``rb_list`` is claimed or refused together, so a
        microbatch's ``rb_multiplier`` rollout batches never straddle the
        window target and GRPO grouping stays intact.

        Returns
        -------
        bool
            ``False`` when the target is already met and the caller must drop
            this microbatch, leaving its prompt eligible for reclaim.
        """
        if not self._tail_batching:
            return True
        assert len(rb_list) == self.rb_multiplier, (
            f"agent returned {len(rb_list)} rollout batches for one microbatch, "
            f"expected rb_multiplier={self.rb_multiplier}"
        )
        if self._step_collected_rb >= self._step_target_rb:
            log(
                f"[RolloutController] discarding excess microbatch "
                f"{ppo_step=} {microbatch_idx=} "
                f"({self._step_collected_rb}/{self._step_target_rb} rb collected)"
            )
            return False
        pending = self._pending.get((ppo_step, microbatch_idx))
        assert pending is not None, (f"no pending record for {ppo_step=} {microbatch_idx=}")
        pending.rollout_consumed = True
        self._step_collected_rb += len(rb_list)
        if self._step_collected_rb >= self._step_target_rb:
            self._abort_t0 = time.monotonic()
            self._last_inflight_done_at = None
            self._pause_task = asyncio.create_task(self._broadcast_pause())
        return True

    async def _broadcast_pause(self):
        """Stop in-flight generation and scoring once the window target is met.

        Every microbatch counted towards the target has already returned, so
        whatever is still running belongs to microbatches this window will
        refuse anyway.  gen-RM runs on its own engine and is frequently the
        larger half of a microbatch's cost, so leaving it alone would waste
        most of the pause.
        """
        log("[RolloutController] target reached -> pausing generation")
        await asyncio.gather(*[a.pause_generation.remote() for a in self.agent_loop_actors])
        aborts = [self._sampler_client.abort_all()]
        if self._gen_rm_client is not None:
            aborts.extend(
                self._gen_rm_client.abort_all(rm_idx)
                for rm_idx in range(self._gen_rm_client.num_rms)
            )
        await asyncio.gather(*aborts)

    def _track_inflight(self, task: asyncio.Task) -> None:
        task.add_done_callback(self._on_inflight_done)
        self._inflight_tasks.append(task)

    def _on_inflight_done(self, _task: asyncio.Task) -> None:
        if self._abort_t0 is None:
            return
        self._last_inflight_done_at = time.monotonic()

    # ------------------------------------------------------------------ #
    #  work for resume training
    # ------------------------------------------------------------------ #

    def load_data_source(self, step: int):
        """Restore tail-batching data-source state for a resumed run.

        Driven by the trainer with the step it actually resumed from: the data
        source must not guess from a checkpoint tracker it does not own, or a
        fresh run could silently skip everything a previous run consumed.
        """
        self.data_source.load(step)

    def save_data_source(self, step: int):
        """Persist data-source state alongside a checkpoint.

        Tail-batching snapshots unused / train-unconsumed prompts only when
        ``rollout_reuse_unused_prompts`` is on. Other data sources own their
        checkpoint semantics.
        """
        assert self.data_source is not None
        if not self._tail_batching:
            self.data_source.save(step)
            return
        pending_batches = []
        if self.training_config.rollout_reuse_unused_prompts:
            for pending in self._pending.values():
                if pending.train_consumed:
                    continue
                batch = dict(pending.raw)
                batch["cache_keys"] = list(batch["cache_keys"])
                self._strip_controller_dispatch_attrs(batch)
                pending_batches.append(batch)
        self.data_source.save(step, pending_unconsumed_batches=pending_batches)

    @staticmethod
    def _strip_controller_dispatch_attrs(batch: Dict[str, Any]) -> None:
        """Remove per-dispatch fields before a prompt is reused."""
        assert "unique_id" in batch
        assert "prompt_idx" in batch
        assert "cache_keys" in batch
        del batch["unique_id"]
        del batch["prompt_idx"]
        cache_keys = list(batch["cache_keys"])
        assert "prompt_idx" in cache_keys
        cache_keys.remove("prompt_idx")
        if cache_keys:
            batch["cache_keys"] = cache_keys
        else:
            del batch["cache_keys"]

    # ------------------------------------------------------------------ #
    #  Shared rollout batch helpers
    # ------------------------------------------------------------------ #

    def read_and_clean_batches(self, num_mb: int, ppo_step: int) -> List[Dict[str, Any]]:
        """Read rollout microbatches and strip attrs not needed by sampler.

        Returns more than ``num_mb`` items when tail-batching over-fires.
        """
        batches = self.data_source.get_batch(num_mb)
        for rbi, bd in enumerate(batches):
            self._assign_unique_id(bd, rbi)
            if self.training_config.rollout_ordered_collection:
                self._tag_rollout_origin(bd, ppo_step, rbi)
        cleaned_batches = []
        for rbi, bd in enumerate(batches):
            # The raw copy must be taken before the cleaning step mutates bd.
            self._record_pending(bd, ppo_step, rbi)
            cleaned_batches.append(
                self.apply_sampling_rollout_attr.remove_rollout_attr_before_sampling(bd)
            )
        return cleaned_batches

    def _record_pending(self, batched_data: Dict[str, Any], ppo_step: int, rbi: int) -> None:
        """Track one in-flight microbatch so its prompt can be reclaimed."""
        if not self._tail_batching:
            return
        self._pending[(ppo_step, rbi)] = PendingMicrobatch(raw=dict(batched_data))

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
        if self.training_config.rollout_ordered_collection:
            self._validate_and_drop_rollout_origin_attrs(rbs, ppo_step, num_expected)
        self.apply_sampling_rollout_attr.clear_data_cache()
        if self.training_config.dynamic_batch_train:
            result = self._prepare_variable_sample_count_train_data(rbs)
        else:
            result = GenerateResult(dp_refs=self._split_train_data_by_dp(rbs))
        fire_metrics = self._fire_metrics.pop(ppo_step, None)
        if fire_metrics is not None:
            result.metrics.update(fire_metrics)
        return result

    def _prepare_variable_sample_count_train_data(
        self,
        rbs: List[Dict[str, List[Any]]],
    ) -> GenerateResult:
        """Prepare variable-sample-count train data for distributed training.

        Here, ``dynamic`` means that sample-level filtering and conversion may
        leave each train step with a different number and structure of samples.
        The controller must therefore plan train-step boundaries from the full
        rollout view and partition the actual post-processed samples, instead
        of deriving fixed batches from nominal GBS. This is independent of
        dynamic microbatch sizing and Dynamic Context Parallel; Dynamic CP is
        an optional downstream mechanism for routing variable-length sequences.

        The preparation pipeline performs the following operations:

        1. Expand each batched dictionary into per-sample dictionaries and
           assign complete prompts to train steps.
        2. Build sample masks, filter samples, normalize rewards, and compute
           rollout metrics.
        3. Convert the samples to the policy training format and validate the
           resulting fields and train-step grouping.
        4. Balance every train step by token count across DP ranks and store
           each rank's nested train-step data in the Ray object store.

        Parameters
        ----------
        rbs : List[Dict[str, List[Any]]]
            Collected rollout batches. Each element is a batched dictionary
            that maps a field name to an aligned list of field values. Position
            ``i`` across all fields represents one rollout sample. Expanding
            the outer list produces ``List[Dict[str, Any]]``.

        Returns
        -------
        GenerateResult
            Prepared training data and metrics. ``dp_refs`` has length
            ``self._dp_size``. Element ``dp_refs[rank]`` is an ``Envelope``
            whose ``inner_data`` is a Ray ``ObjectRef`` resolving to
            ``List[List[Dict[str, Any]]]``: the outer list is ordered by train
            step, and each inner list contains the converted samples assigned
            to that DP rank for the step. ``metrics`` maps metric names to
            scalars and ``*_histogram`` values computed from the filtered samples.

        Raises
        ------
        AssertionError
            If the rollout or training configuration cannot form complete
            train steps, filtering or conversion removes an entire train step,
            converted samples violate dynamic-batch invariants, or a train
            step cannot be partitioned across DP ranks.
        """
        training_config = self.training_config
        samples = expand_rollout_batches(rbs)
        # 1. stamp _train_step_id by prompt_idx
        samples = assign_train_steps(samples, training_config)
        expected_train_step_ids = {sample[TRAIN_STEP_ID_KEY] for sample in samples}
        # 2. prepare train masks
        prepare_sample_masks(samples)
        # 3. filter samples
        samples = filter_rollout_samples(
            self.config,
            samples,
            custom_filter_fn=self.custom_filter_samples,
        )
        # 4. normalize rewards
        pre_advantage_metrics = reward_normalize(
            self.config,
            samples,
            custom_reward_normalize_fn=self.custom_reward_normalize,
        )
        requires_normalized_rewards = uses_group_reward_normalization(self.config)
        assert {sample[TRAIN_STEP_ID_KEY]
                for sample in samples
               } == expected_train_step_ids, ("filter removed an entire train step")
        # 5. compute rollout metrics
        metrics = compute_rollout_metrics(
            self.config,
            samples,
            training_config.metrics_report or [],
            custom_metrics_fn=self.custom_compute_rollout_metrics,
        )
        metrics.update(pre_advantage_metrics)
        # 6. convert samples to train data
        samples = convert_samples_to_train_data(
            self.config,
            samples,
            custom_convert_fn=self.custom_convert_samples_to_train_data,
        )
        validate_dynamic_batch_samples(
            samples,
            require_normalized_rewards=requires_normalized_rewards,
        )
        assert {
            sample[TRAIN_STEP_ID_KEY]
            for sample in samples
        } == expected_train_step_ids, ("custom train-data conversion removed an entire train step")
        # 7. re-group by _train_step_id and balance each step by DP rank
        dp_partitions = split_train_steps_by_dp(
            samples,
            self._dp_size,
            training_config.train_mbs,
            self.config.policy.dist_config.dynamic_context_parallel,
        )
        refs = [Envelope(ray.put(partition)) for partition in dp_partitions]
        num_train_steps = len(dp_partitions[0]) if dp_partitions else 0
        log(
            f"[RolloutController] split {len(samples)} samples in "
            f"{num_train_steps} train steps across {self._dp_size} DP ranks"
        )
        return GenerateResult(dp_refs=refs, metrics=metrics)

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
        ppo_step_per_epoch = self._ppo_step_per_epoch
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
        elif self.is_partial_colocated:
            assert num_ppo_steps == 1, (
                "partial_colocated single_controller only supports num_ppo_steps=1"
            )
            assert self.config.placement_type == "partial_colocated", (
                "step rollout lifecycle requires placement_type='partial_colocated'"
            )
            assert self.training_config.rollout_max_staleness == 0, (
                "partial_colocated requires rollout_max_staleness == 0"
            )
            assert not self._inflight_tasks, (
                "partial_colocated does not allow overlapping inflight steps"
            )
            assert self._ready_queue.empty(
            ), ("partial_colocated requires collect before firing the next step")
        else:
            assert self.config.placement_type == "disaggregated", (
                f"unsupported placement_type={self.config.placement_type!r}"
            )

        agent_actors = self.agent_loop_actors
        num_agent_actors = len(agent_actors)
        total_fired = 0

        curr_sample_idx = self._next_fire_sample_idx
        base_actor_idx = self._next_fire_actor_idx

        for step_offset in range(num_ppo_steps):
            cur_ppo_step = ppo_step + step_offset
            # Over-firing makes per-step counts vary, so sample indices must
            # advance by what was actually fired to stay globally unique.
            step_sample_idx_base = curr_sample_idx + total_fired

            buffer_length = None
            if self._tail_batching:
                buffer_length = self.data_source.buffer_length
            cleaned_batches = self.read_and_clean_batches(num_mb, cur_ppo_step)
            fired_this_step = len(cleaned_batches)
            fire_metrics = {
                "rollout-metrics/fired_this_step":
                    float(fired_this_step) if self.use_colocate else 0.0,
            }
            if self._tail_batching:
                fire_metrics["rollout-metrics/buffer_length"] = float(buffer_length)
            self._fire_metrics[cur_ppo_step] = fire_metrics
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
                        per_agent_microbatch_indices,
                        cur_ppo_step,
                    )
                )
                self._track_inflight(task)
                total_fired += fired_this_step
            elif self.is_partial_colocated:
                task = asyncio.create_task(
                    self.dispatch_partial_colocated_step(
                        agent_actors,
                        per_agent_batches,
                        per_agent_indices,
                        per_agent_microbatch_indices,
                        cur_ppo_step,
                    )
                )
                self._track_inflight(task)
                total_fired += fired_this_step
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
                    self._track_inflight(task)
                    total_fired += 1

        self._next_fire_sample_idx = curr_sample_idx + total_fired
        self._next_fire_actor_idx = (base_actor_idx + total_fired) % num_agent_actors

        log(
            f"[RolloutController] fired {total_fired} microbatches "
            f"across {num_ppo_steps} steps "
            f"(ppo_step {ppo_step}..{ppo_step + num_ppo_steps - 1})"
        )

    async def dispatch_partial_colocated_step(
        self,
        agent_actors: List,
        per_agent_batches: List[List[Dict[str, Any]]],
        per_agent_indices: List[List[int]],
        per_agent_microbatch_indices: List[List[int]],
        ppo_step: int,
    ):
        """Dispatch one partial-colocated PPO step under one lifecycle context."""
        assert self.is_partial_colocated
        dispatch_items = []
        for actor_idx, (batches_part, indices_part, mb_indices_part) in enumerate(
            zip(per_agent_batches, per_agent_indices, per_agent_microbatch_indices)
        ):
            for cleaned_data, sample_idx, microbatch_idx in zip(
                batches_part,
                indices_part,
                mb_indices_part,
            ):
                dispatch_items.append((microbatch_idx, actor_idx, cleaned_data, sample_idx))

        async with self.partial_colocated_rollout(ppo_step):
            results = await asyncio.gather(
                *[
                    self.dispatch_single_item_to_agent(
                        agent_actors[actor_idx],
                        cleaned_data,
                        ppo_step,
                        microbatch_idx,
                        sample_idx,
                    ) for microbatch_idx, actor_idx, cleaned_data, sample_idx in
                    sorted(dispatch_items)
                ],
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise errors[0]

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
        fails fast instead of dead-locking.  Under tail-batching the result
        list is first offered to the window accountant, which may refuse it
        as excess.
        """
        if self._step_collected_rb >= self._step_target_rb:
            return
        try:
            ref = actor.agent_loop.remote(cleaned_data, ppo_step, microbatch_idx, sample_idx)
            metrics = self._fire_metrics.setdefault(ppo_step, {})
            key = "rollout-metrics/fired_this_step"
            metrics[key] = metrics.get(key, 0.0) + 1.0
            rb_list = await ref
            if not self._try_accept_completed_microbatch(rb_list, ppo_step, microbatch_idx):
                return
            for rb in rb_list:
                await self._ready_queue.put(
                    QueuedRolloutBatch(
                        rollout_batch=rb,
                        ppo_step=ppo_step,
                        microbatch_idx=microbatch_idx,
                        sample_idx=sample_idx,
                    )
                )
        except GenerationAborted as e:
            # The only abort source is the tail-batching pause, which fires
            # after the window target is met, so this microbatch is excess by
            # construction and its prompt stays reclaimable.  An abort with no
            # pause in flight is a real failure.
            if self._pause_task is None:
                traceback.print_exc()
                await self._ready_queue.put(e)
                raise
            log(
                f"[RolloutController] dropping aborted microbatch "
                f"{ppo_step=} {microbatch_idx=}"
            )
        except Exception as e:
            traceback.print_exc()
            await self._ready_queue.put(e)
            raise

    async def dispatch_batches_to_agents(
        self,
        agents: List,
        per_agent_batches: List[List[Dict[str, Any]]],
        per_agent_indices: List[List[int]],
        per_agent_microbatch_indices: List[List[int]],
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
                        strict=True,
                    )
                ]
            )
            for agent_rbs, sample_indices, microbatch_indices in zip(
                agent_results,
                per_agent_indices,
                per_agent_microbatch_indices,
                strict=True,
            ):
                assert len(agent_rbs) == len(microbatch_indices) * self.rb_multiplier, (
                    "agent returned an unexpected number of rollout batches: "
                    f"{len(agent_rbs)=}, {len(microbatch_indices)=}, "
                    f"{self.rb_multiplier=}"
                )
                result_idx = 0
                for sample_idx, microbatch_idx in zip(
                    sample_indices, microbatch_indices, strict=True
                ):
                    for _ in range(self.rb_multiplier):
                        await self._ready_queue.put(
                            QueuedRolloutBatch(
                                rollout_batch=agent_rbs[result_idx],
                                ppo_step=ppo_step,
                                microbatch_idx=microbatch_idx,
                                sample_idx=sample_idx,
                            )
                        )
                        result_idx += 1
        except Exception as e:
            traceback.print_exc()
            await self._ready_queue.put(e)
            raise

    async def collect_rollout_step(self, ppo_step: int) -> GenerateResult:
        """Collect the next complete PPO step worth of microbatches.

        Drains ``num_microbatches * rb_multiplier`` items from the ready
        queue, restores cached rollout attrs, and splits by DP rank.
        Default collection preserves first-finished queue order.  With
        ``training.rollout_ordered_collection=True``, later PPO steps are
        buffered until the requested original ``ppo_step`` is complete.

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

        if self.training_config.rollout_ordered_collection:
            rbs = await self.collect_ordered_rollout_batches(ppo_step, num_mb)
        else:
            rbs = await self.collect_first_finished_rollout_batches(num_collect)

        result = await self.finalize_rollout_batches(rbs, ppo_step, num_collect)
        self._sample_idx += num_mb
        return result

    async def collect_first_finished_rollout_batches(
        self, num_collect: int
    ) -> List[Dict[str, List[Any]]]:
        """Collect rollout batches in queue arrival order."""
        rbs = []
        for _ in range(num_collect):
            item = await self._ready_queue.get()
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, QueuedRolloutBatch
                             ), (f"unexpected ready queue item type: {type(item)}")
            if self._tail_batching:
                pending = self._pending.get((item.ppo_step, item.microbatch_idx))
                if pending is not None:
                    pending.train_consumed = True
            rbs.append(item.rollout_batch)
        return rbs

    async def collect_ordered_rollout_batches(self, ppo_step: int,
                                              num_mb: int) -> List[Dict[str, List[Any]]]:
        """Collect rollout batches for ``ppo_step`` in original order."""
        self._assert_no_stale_ordered_steps(ppo_step)

        while not self._ordered_step_complete(ppo_step, num_mb):
            item = await self._ready_queue.get()
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, QueuedRolloutBatch
                             ), (f"unexpected ready queue item type: {type(item)}")
            assert item.ppo_step >= ppo_step, (
                f"ordered collection saw stale ppo_step {item.ppo_step} "
                f"while collecting {ppo_step}"
            )
            self._buffer_ordered_rollout_batch(item)

        step_buf = self._ordered_ready.pop(ppo_step)
        rbs = []
        for microbatch_idx in range(num_mb):
            items = step_buf.pop(microbatch_idx)
            assert len(items) == self.rb_multiplier, (
                f"expected {self.rb_multiplier} rollout batches for "
                f"{ppo_step=} {microbatch_idx=}, got {len(items)}"
            )
            rbs.extend(item.rollout_batch for item in items)
        assert not step_buf, f"unexpected leftover microbatches for {ppo_step=}: {step_buf}"
        return rbs

    def _buffer_ordered_rollout_batch(self, item: QueuedRolloutBatch) -> None:
        """Buffer a queued rollout item by original PPO step and microbatch."""
        step_buf = self._ordered_ready.setdefault(item.ppo_step, {})
        microbatch_buf = step_buf.setdefault(item.microbatch_idx, [])
        microbatch_buf.append(item)
        assert len(microbatch_buf) <= self.rb_multiplier, (
            f"too many rollout batches for ppo_step={item.ppo_step} "
            f"microbatch_idx={item.microbatch_idx}: "
            f"{len(microbatch_buf)} > {self.rb_multiplier}"
        )

    def _ordered_step_complete(self, ppo_step: int, num_mb: int) -> bool:
        """Return whether all microbatches for ``ppo_step`` are buffered."""
        step_buf = self._ordered_ready.get(ppo_step)
        if step_buf is None:
            return False
        for microbatch_idx in range(num_mb):
            items = step_buf.get(microbatch_idx)
            if items is None or len(items) != self.rb_multiplier:
                return False
        return True

    def _assert_no_stale_ordered_steps(self, ppo_step: int) -> None:
        """Fail loudly if ordered collection skipped an older PPO step."""
        stale_steps = [step for step in self._ordered_ready if step < ppo_step]
        assert not stale_steps, (
            f"ordered collection has stale buffered steps before {ppo_step}: "
            f"{sorted(stale_steps)}"
        )

    async def wait_all_inflight(self) -> Dict[str, float]:
        """Wait for all inflight pipeline tasks to complete.

        Call before ``update_weights`` to guarantee the sampler is idle
        (otherwise a mid-generation weight swap would mix old and new
        weights on the same sample). Ready rollout data in ``_ready_queue``
        is intentionally NOT drained — those batches are already finalized
        under the previous weights and are safe to carry across the update
        boundary.

        On return (including exceptional return), ``_inflight_tasks`` is
        empty; this is asserted as a cross-boundary invariant.

        Returns
        -------
        dict[str, float]
            ``time_perf/abort_to_idle`` when a tail-batching pause ran:
            abort dispatch to the last inflight task actually finishing.
        """
        results = []
        if self._inflight_tasks:
            try:
                # NOTE: now we must wait all tasks to complete before raising exceptions
                results = await asyncio.gather(
                    *self._inflight_tasks,
                    return_exceptions=True,
                )
            finally:
                self._inflight_tasks.clear()
        assert not self._inflight_tasks, (
            "wait_all_inflight post-condition: _inflight_tasks must be empty"
        )
        metrics: Dict[str, float] = {}
        if self._abort_t0 is not None:
            end = self._last_inflight_done_at
            if end is None:
                end = self._abort_t0
            metrics["time_perf/abort_to_idle"] = end - self._abort_t0
            self._abort_t0 = None
            self._last_inflight_done_at = None
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return metrics

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
            sampling_repeat_n = self.training_config.sampling_repeat_n

            for rbi, rb in enumerate(rbs):
                sidx = self._sample_idx + rbi
                co = self.bt_rm_client.get_bt_rm_result(
                    rm_idx, ppo_step, sidx, sampling_repeat_n=sampling_repeat_n
                )
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

    def _assign_unique_id(
        self,
        batched_data: Dict[str, Any],
        rbi: int,
    ) -> None:
        """Assign unique IDs and prompt indices to each sample in the batch."""
        if "unique_id" not in batched_data:
            first_key = list(batched_data.keys())[0]
            batch_size = len(batched_data[first_key])
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_id_list = [f"ctrl_rbi_{rbi}_batch_id_{bid}_{ts}" for bid in range(batch_size)]
            batched_data["unique_id"] = unique_id_list
        batch_size = len(batched_data["unique_id"])
        batched_data["prompt_idx"] = list(
            range(self._next_prompt_idx, self._next_prompt_idx + batch_size)
        )
        self._next_prompt_idx += batch_size
        cache_keys = list(batched_data.get("cache_keys", []))
        if "prompt_idx" not in cache_keys:
            cache_keys.append("prompt_idx")
        batched_data["cache_keys"] = cache_keys

    def _tag_rollout_origin(
        self,
        batched_data: Dict[str, Any],
        ppo_step: int,
        microbatch_idx: int,
    ) -> None:
        """Attach controller provenance attrs via the rollout attr cache."""
        assert "unique_id" in batched_data
        batch_size = len(batched_data["unique_id"])
        assert ORIGIN_PPO_STEP_KEY not in batched_data
        assert ORIGIN_MICROBATCH_IDX_KEY not in batched_data

        if "cache_keys" in batched_data:
            cache_keys = list(batched_data["cache_keys"])
        else:
            cache_keys = []
        for key in (ORIGIN_PPO_STEP_KEY, ORIGIN_MICROBATCH_IDX_KEY):
            assert key not in cache_keys
            cache_keys.append(key)
        batched_data["cache_keys"] = cache_keys
        batched_data[ORIGIN_PPO_STEP_KEY] = [ppo_step] * batch_size
        batched_data[ORIGIN_MICROBATCH_IDX_KEY] = [microbatch_idx] * batch_size

    def _validate_and_drop_rollout_origin_attrs(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        num_expected: int,
    ) -> None:
        """Validate sample-level provenance, then remove internal attrs."""
        assert len(rbs
                  ) == num_expected, (f"expected {num_expected} rollout batches, got {len(rbs)}")
        collected_microbatch_indices = []
        for rb in rbs:
            assert ORIGIN_PPO_STEP_KEY in rb
            assert ORIGIN_MICROBATCH_IDX_KEY in rb
            if self.training_config.rollout_ordered_collection:
                for origin_step in rb[ORIGIN_PPO_STEP_KEY]:
                    assert origin_step == ppo_step, (
                        f"ordered collection mixed rollout origin ppo_step "
                        f"{origin_step} into collect ppo_step {ppo_step}"
                    )
                origin_microbatch_indices = rb[ORIGIN_MICROBATCH_IDX_KEY]
                assert origin_microbatch_indices, "origin microbatch index list is empty"
                origin_microbatch_idx = origin_microbatch_indices[0]
                for idx in origin_microbatch_indices:
                    assert idx == origin_microbatch_idx, (
                        f"one rollout batch contains mixed origin microbatch indices: "
                        f"{origin_microbatch_indices}"
                    )
                collected_microbatch_indices.append(origin_microbatch_idx)
            del rb[ORIGIN_PPO_STEP_KEY]
            del rb[ORIGIN_MICROBATCH_IDX_KEY]
        if self.training_config.rollout_ordered_collection:
            expected_microbatch_indices = [
                microbatch_idx for microbatch_idx in range(self._num_microbatches)
                for _ in range(self.rb_multiplier)
            ]
            assert collected_microbatch_indices == expected_microbatch_indices, (
                f"ordered collection returned microbatches in wrong order: "
                f"{collected_microbatch_indices=} {expected_microbatch_indices=}"
            )

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
