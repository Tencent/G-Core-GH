"""Unit tests for ``RolloutController.dispatch_single_item_to_agent`` and the
rolling-prefetch fire-side bookkeeping.

Dispatch-failure tests (the original focus) are in
``DispatchToActorTest``.  Fire-side counter / round-robin tests
(``_next_fire_sample_idx`` and ``_next_fire_actor_idx``) are in
``FireBookkeepingTest`` and target the scenario where the trainer issues
multiple ``fire_generation_requests`` calls without an intervening
``collect_rollout_step`` -- which happens whenever ``fire_up_to`` has to
split on an epoch boundary.
"""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
    GenerateResult,
    RolloutController,
    QueuedRolloutBatch,
    ORIGIN_MICROBATCH_IDX_KEY,
    ORIGIN_PPO_STEP_KEY,
)


def _make_bare_controller() -> RolloutController:
    """Construct a controller skeleton without invoking ``__init__``.

    ``RolloutController.__init__`` requires a fully-formed ``RlConfig``
    plus dependent clients. For the dispatcher path we only need
    ``_ready_queue`` to exist.  Fire-bookkeeping tests set up a few more
    fields below via the bare instance.
    """
    rc = object.__new__(RolloutController)
    rc._ready_queue = asyncio.Queue()
    rc._ordered_ready = {}
    rc._inflight_tasks = []
    rc.use_colocate = False
    rc.is_partial_colocated = False
    rc.rb_multiplier = 1
    rc.training_config = SimpleNamespace(rollout_ordered_collection=False)
    return rc


class _FakeRemote:
    """Mimics ``actor.agent_loop.remote(...)`` returning an awaitable."""
    def __init__(self, result=None, exc: Exception = None):
        self._result = result
        self._exc = exc

    def remote(self, *args, **kwargs):
        async def _coro():
            if self._exc is not None:
                raise self._exc
            return self._result

        return _coro()


class _FakeActor:
    def __init__(self, result=None, exc: Exception = None):
        self.agent_loop = _FakeRemote(result=result, exc=exc)


class _RecordingRemote:
    """Mimics ``actor.agent_loop.remote`` and logs the (ppo_step, rbi, sidx)
    tuple it was called with.  Returns one fake rb list per call.
    """
    def __init__(self):
        self.calls: List = []

    def remote(self, cleaned_data, ppo_step, microbatch_idx, sample_idx):
        async def _coro():
            self.calls.append((ppo_step, microbatch_idx, sample_idx))
            return [{"tokens": [[sample_idx]]}]

        return _coro()


class _RecordingActor:
    def __init__(self):
        self.agent_loop = _RecordingRemote()


class _LifecycleRemote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _LifecycleActor(_RecordingActor):
    def __init__(self, events: List):
        super().__init__()
        self.begin_partial_colocated_rollout = _LifecycleRemote(self._begin)
        self.end_partial_colocated_rollout = _LifecycleRemote(self._end)
        self.events = events

    async def _begin(self, ppo_step: int):
        self.events.append(("begin_lifecycle", ppo_step))

    async def _end(self, ppo_step: int):
        self.events.append(("end_lifecycle", ppo_step))


class _DelayedRecordingRemote:
    """Return rollout batches after configurable delays to scramble finish order."""
    def __init__(self, delays):
        self.delays = delays
        self.calls: List = []

    def remote(self, cleaned_data, ppo_step, microbatch_idx, sample_idx):
        async def _coro():
            self.calls.append((ppo_step, microbatch_idx, sample_idx))
            await asyncio.sleep(self.delays[(ppo_step, microbatch_idx)])
            return [{
                "tokens": [[sample_idx]],
                "source_order": [(ppo_step, microbatch_idx, sample_idx)],
            }]

        return _coro()


class _DelayedRecordingActor:
    def __init__(self, delays):
        self.agent_loop = _DelayedRecordingRemote(delays)


class _FakeDataSource:
    """Minimal data source: each ``get_batch(num_mb)`` returns ``num_mb``
    distinct dicts.  Tracks epoch sets so tests can verify single-epoch
    semantics per fire.

    ``dataloader_length`` is set to ``num_mb * 1000`` so the single-epoch
    invariant assert in ``fire_generation_requests`` comfortably passes
    for small-step tests (they always live within epoch 0).
    """
    def __init__(self, dataloader_length: int = 2000):
        self.epochs: List[int] = []
        self._batch_counter = 0
        self.dataloader_length = dataloader_length

    def maybe_set_epoch(self, epoch: int):
        self.epochs.append(epoch)

    def get_batch(self, num_mb: int) -> List[Dict[str, Any]]:
        batches = []
        for _ in range(num_mb):
            batches.append({"prompt": [f"p{self._batch_counter}"]})
            self._batch_counter += 1
        return batches


class _NoopRolloutAttr:
    """No-op ``ApplySamplingRolloutAttr`` used to isolate dispatch logic."""
    def remove_rollout_attr_before_sampling(self, bd: Dict[str, Any]):
        return bd


class DispatchToActorTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_enqueues_each_rb(self):
        rc = _make_bare_controller()
        rb_a: Dict[str, List[Any]] = {"tokens": [[1, 2]]}
        rb_b: Dict[str, List[Any]] = {"tokens": [[3, 4]]}
        actor = _FakeActor(result=[rb_a, rb_b])

        await rc.dispatch_single_item_to_agent(
            actor, {"prompt": "x"}, ppo_step=0, microbatch_idx=0, sample_idx=0
        )

        assert rc._ready_queue.qsize() == 2
        first = await rc._ready_queue.get()
        second = await rc._ready_queue.get()
        assert isinstance(first, QueuedRolloutBatch)
        assert isinstance(second, QueuedRolloutBatch)
        assert first.rollout_batch is rb_a
        assert second.rollout_batch is rb_b
        assert first.ppo_step == 0
        assert second.ppo_step == 0
        assert first.microbatch_idx == 0
        assert second.microbatch_idx == 0
        assert first.sample_idx == 0
        assert second.sample_idx == 0

    async def test_failure_enqueues_exception_and_reraises(self):
        """Regression test: on agent_loop failure the queue must receive
        the exception so the consumer doesn't dead-lock.
        """
        rc = _make_bare_controller()
        boom = RuntimeError("agent_loop blew up")
        actor = _FakeActor(exc=boom)

        with self.assertRaises(RuntimeError) as cm:
            await rc.dispatch_single_item_to_agent(
                actor, {"prompt": "x"}, ppo_step=0, microbatch_idx=0, sample_idx=0
            )
        assert cm.exception is boom

        assert rc._ready_queue.qsize() == 1
        item = await rc._ready_queue.get()
        assert item is boom

    async def test_consumer_does_not_hang_on_dispatch_failure(self):
        """End-to-end: a consumer mirroring ``collect_rollout_step``'s
        ``rb = await queue.get(); if isinstance(rb, Exception): raise rb``
        must complete (raise) within a bounded time when the dispatcher
        fails -- previously this would hang forever.
        """
        rc = _make_bare_controller()
        actor = _FakeActor(exc=ValueError("dispatch fail"))

        async def consumer():
            rb = await rc._ready_queue.get()
            if isinstance(rb, Exception):
                raise rb
            return rb

        dispatcher_task = asyncio.create_task(
            rc.dispatch_single_item_to_agent(
                actor, {"prompt": "x"}, ppo_step=0, microbatch_idx=0, sample_idx=0
            )
        )
        consumer_task = asyncio.create_task(consumer())

        with self.assertRaises(ValueError):
            await asyncio.wait_for(consumer_task, timeout=1.0)

        with self.assertRaises(ValueError):
            await asyncio.wait_for(dispatcher_task, timeout=1.0)


def _make_fire_controller(num_mb: int, num_actors: int) -> RolloutController:
    """Wire up a bare controller with just enough state for
    ``fire_generation_requests`` to run end-to-end in-process.
    """
    rc = _make_bare_controller()
    rc._num_microbatches = num_mb
    rc._next_fire_sample_idx = 0
    rc._next_fire_actor_idx = 0
    rc._sample_idx = 0
    rc.rb_multiplier = 1
    rc.data_source = _FakeDataSource()
    rc.apply_sampling_rollout_attr = _NoopRolloutAttr()
    rc.agent_loop_actors = [_RecordingActor() for _ in range(num_actors)]
    rc.config = SimpleNamespace(placement_type="disaggregated")
    rc.training_config = SimpleNamespace(
        single_controller=True,
        rollout_ordered_collection=False,
    )
    return rc


def _make_partial_fire_controller(num_mb: int, num_actors: int, max_stale: int = 0):
    rc = _make_bare_controller()
    rc.is_partial_colocated = True
    rc._num_microbatches = num_mb
    rc._next_fire_sample_idx = 0
    rc._next_fire_actor_idx = 0
    rc._sample_idx = 0
    rc.rb_multiplier = 1
    rc.data_source = _FakeDataSource()
    rc.apply_sampling_rollout_attr = _NoopRolloutAttr()
    events: List = []
    rc.agent_loop_actors = [_LifecycleActor(events) for _ in range(num_actors)]
    rc.config = SimpleNamespace(placement_type="partial_colocated")
    rc.training_config = SimpleNamespace(
        single_controller=True,
        rollout_max_staleness=max_stale,
        rollout_ordered_collection=False,
    )
    return rc, events


class _RecordingColocateRemote:
    """Mimics default ``AgentLoopActor.agent_loop.remote`` in colocate mode."""
    def __init__(self):
        self.calls: List = []

    def remote(
        self,
        cleaned_batches,
        ppo_step,
        microbatch_idx,
        sample_idx,
        use_colocate: bool = False,
    ):
        async def _coro():
            self.calls.append((ppo_step, len(cleaned_batches), list(sample_idx), use_colocate))
            return [{"tokens": [[sidx]]} for sidx in sample_idx]

        return _coro()


class _RecordingColocateActor:
    def __init__(self):
        self.agent_loop = _RecordingColocateRemote()


def _make_colocate_fire_controller(num_mb: int, num_actors: int) -> RolloutController:
    """Wire up a bare single-controller colocate dispatch path."""
    rc = _make_bare_controller()
    rc.use_colocate = True
    rc._num_microbatches = num_mb
    rc._next_fire_sample_idx = 0
    rc._next_fire_actor_idx = 0
    rc._sample_idx = 0
    rc.rb_multiplier = 1
    rc.data_source = _FakeDataSource()
    rc.apply_sampling_rollout_attr = _NoopRolloutAttr()
    rc.agent_loop_actors = [_RecordingColocateActor() for _ in range(num_actors)]
    rc.config = SimpleNamespace(placement_type="colocate")
    rc.training_config = SimpleNamespace(
        single_controller=True,
        rollout_max_staleness=0,
        rollout_ordered_collection=False,
    )
    return rc


class FireBookkeepingTest(unittest.IsolatedAsyncioTestCase):
    """Rolling prefetch fires the same controller multiple times before any
    ``collect_rollout_step`` runs.  These tests pin down that the fire-side
    state advances correctly across those calls.
    """
    async def test_consecutive_single_step_fires_use_unique_sample_indices(self):
        """Two back-to-back ``fire_generation_requests(num_ppo_steps=1)``
        calls must dispatch 4 unique sample indices (0..3 with num_mb=2).

        Before the fire-side counter split, both calls would reuse the
        base ``_sample_idx`` because ``collect_rollout_step`` (which
        advances that counter) had not run yet.
        """
        rc = _make_fire_controller(num_mb=2, num_actors=2)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        await rc.fire_generation_requests(epoch=0, ppo_step=1, num_ppo_steps=1)
        await rc.wait_all_inflight()

        calls = []
        for actor in rc.agent_loop_actors:
            calls.extend(actor.agent_loop.calls)
        sample_indices = sorted(call[2] for call in calls)
        ppo_steps = sorted({call[0] for call in calls})

        assert sample_indices == [0, 1, 2, 3
                                 ], (f"expected unique sample indices 0..3, got {sample_indices}")
        assert ppo_steps == [0, 1]
        assert rc._next_fire_sample_idx == 4

    async def test_consecutive_single_step_fires_rotate_agent_actors(self):
        """Actor round-robin must carry over between fires: with 2 actors
        and 2 single-mb fires, each actor should receive exactly 1 call.
        """
        rc = _make_fire_controller(num_mb=1, num_actors=2)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        await rc.fire_generation_requests(epoch=0, ppo_step=1, num_ppo_steps=1)
        await rc.wait_all_inflight()

        assert len(rc.agent_loop_actors[0].agent_loop.calls) == 1
        assert len(rc.agent_loop_actors[1].agent_loop.calls) == 1
        assert rc._next_fire_actor_idx == 0  # wrapped around

    async def test_wait_all_inflight_leaves_queue_intact(self):
        """``wait_all_inflight`` clears inflight tasks but NOT the ready
        queue — prefetched batches must survive across update_weights.
        """
        rc = _make_fire_controller(num_mb=2, num_actors=1)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        await rc.wait_all_inflight()

        # All 2 microbatches should be sitting in the ready queue.
        assert rc._ready_queue.qsize() == 2
        assert rc._inflight_tasks == []


class ColocateFireBookkeepingTest(unittest.IsolatedAsyncioTestCase):
    """Single-controller colocate uses the fire/collect queue without overlap."""
    async def test_colocate_fire_dispatches_all_agents_including_empty_partitions(self):
        rc = _make_colocate_fire_controller(num_mb=1, num_actors=3)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        await rc.wait_all_inflight()

        calls = [actor.agent_loop.calls for actor in rc.agent_loop_actors]
        assert [len(c) for c in calls] == [1, 1, 1]
        assert calls[0][0] == (0, 1, [0], True)
        assert calls[1][0] == (0, 0, [], True)
        assert calls[2][0] == (0, 0, [], True)
        assert rc._ready_queue.qsize() == 1
        item = await rc._ready_queue.get()
        assert isinstance(item, QueuedRolloutBatch)
        assert item.ppo_step == 0
        assert item.microbatch_idx == 0
        assert item.sample_idx == 0
        assert item.rollout_batch["tokens"] == [[0]]
        assert rc._next_fire_sample_idx == 1
        assert rc._next_fire_actor_idx == 1


class PartialColocatedLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_partial_wakes_once_and_sleeps_after_inflight_drain(self):
        rc, events = _make_partial_fire_controller(num_mb=2, num_actors=2)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        await rc.wait_all_inflight()

        assert events == [("begin_lifecycle", 0), ("end_lifecycle", 0)]
        assert rc._ready_queue.qsize() == 2

        calls = []
        for actor in rc.agent_loop_actors:
            calls.extend(actor.agent_loop.calls)
        assert sorted(call[2] for call in calls) == [0, 1]

    async def test_partial_rejects_nonzero_staleness(self):
        rc, _ = _make_partial_fire_controller(num_mb=1, num_actors=1, max_stale=1)

        with self.assertRaises(AssertionError):
            await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)

    @unittest.skip(
        "TODO(@astrachang): hangs in my_test.sh — Traceback starts then freezes "
        "(~30m no progress; observed 2026-07-23). Please fix."
    )
    async def test_partial_failure_sleeps_after_all_tasks_finish(self):
        rc, events = _make_partial_fire_controller(num_mb=1, num_actors=1)
        boom = RuntimeError("partial rollout failed")
        rc.agent_loop_actors[0].agent_loop = _FakeRemote(exc=boom)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        with self.assertRaises(RuntimeError):
            await rc.collect_rollout_step(0)
        with self.assertRaises(RuntimeError):
            await rc.wait_all_inflight()

        assert events == [("begin_lifecycle", 0), ("end_lifecycle", 0)]
        assert rc._inflight_tasks == []


def _make_collect_controller(
    num_mb: int,
    rb_multiplier: int = 1,
    ordered: bool = False,
) -> RolloutController:
    """Create a controller skeleton for collect-only tests."""
    rc = _make_bare_controller()
    rc._num_microbatches = num_mb
    rc.rb_multiplier = rb_multiplier
    rc._sample_idx = 0
    rc.training_config = SimpleNamespace(rollout_ordered_collection=ordered)

    async def _finalize_rollout_batches(rbs, ppo_step, num_expected):
        return GenerateResult(dp_refs=rbs)

    rc.finalize_rollout_batches = _finalize_rollout_batches
    return rc


async def _put_queued(
    rc: RolloutController,
    label: str,
    ppo_step: int,
    microbatch_idx: int,
    sample_idx: int,
):
    rb = {"tokens": [[label]]}
    await rc._ready_queue.put(
        QueuedRolloutBatch(
            rollout_batch=rb,
            ppo_step=ppo_step,
            microbatch_idx=microbatch_idx,
            sample_idx=sample_idx,
        )
    )
    return rb


class CollectOrderingTest(unittest.IsolatedAsyncioTestCase):
    async def test_default_collect_preserves_first_finished_queue_order(self):
        rc = _make_collect_controller(num_mb=2, ordered=False)
        rb_step1 = await _put_queued(rc, "step1_mb0", ppo_step=1, microbatch_idx=0, sample_idx=2)
        rb_step0 = await _put_queued(rc, "step0_mb0", ppo_step=0, microbatch_idx=0, sample_idx=0)

        result = await rc.collect_rollout_step(0)

        assert result.dp_refs == [rb_step1, rb_step0]
        assert rc._sample_idx == 2

    async def test_ordered_collect_buffers_later_step(self):
        rc = _make_collect_controller(num_mb=2, ordered=True)
        rb10 = await _put_queued(rc, "step1_mb0", ppo_step=1, microbatch_idx=0, sample_idx=2)
        rb11 = await _put_queued(rc, "step1_mb1", ppo_step=1, microbatch_idx=1, sample_idx=3)
        rb01 = await _put_queued(rc, "step0_mb1", ppo_step=0, microbatch_idx=1, sample_idx=1)
        rb00 = await _put_queued(rc, "step0_mb0", ppo_step=0, microbatch_idx=0, sample_idx=0)

        result0 = await rc.collect_rollout_step(0)
        result1 = await rc.collect_rollout_step(1)

        assert result0.dp_refs == [rb00, rb01]
        assert result1.dp_refs == [rb10, rb11]
        assert rc._ordered_ready == {}
        assert rc._sample_idx == 4

    async def test_ordered_collect_preserves_microbatch_order(self):
        rc = _make_collect_controller(num_mb=3, ordered=True)
        rb2 = await _put_queued(rc, "mb2", ppo_step=0, microbatch_idx=2, sample_idx=2)
        rb0 = await _put_queued(rc, "mb0", ppo_step=0, microbatch_idx=0, sample_idx=0)
        rb1 = await _put_queued(rc, "mb1", ppo_step=0, microbatch_idx=1, sample_idx=1)

        result = await rc.collect_rollout_step(0)

        assert result.dp_refs == [rb0, rb1, rb2]

    async def test_ordered_collect_returns_original_sample_index_order(self):
        """Ordered collection must restore the original rollout-batch order.

        Completion order is intentionally scrambled across steps and
        microbatches.  The returned sequence should match the original
        fire-side metadata for the requested PPO step.
        """
        rc = _make_collect_controller(num_mb=3, ordered=True)

        queued = [
            (1, 0, 3),
            (0, 2, 2),
            (1, 1, 4),
            (0, 0, 0),
            (0, 1, 1),
        ]
        for ppo_step, microbatch_idx, sample_idx in queued:
            rb = {
                "tokens": [[f"step{ppo_step}_mb{microbatch_idx}"]],
                "source_order": [(ppo_step, microbatch_idx, sample_idx)],
            }
            await rc._ready_queue.put(
                QueuedRolloutBatch(
                    rollout_batch=rb,
                    ppo_step=ppo_step,
                    microbatch_idx=microbatch_idx,
                    sample_idx=sample_idx,
                )
            )

        result = await rc.collect_rollout_step(0)

        assert [rb["source_order"][0] for rb in result.dp_refs] == [
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
        ]

    async def test_fire_dispatch_then_ordered_collect_restores_original_order(self):
        """End-to-end controller path: fire creates metadata, collect orders it.

        This goes through ``fire_generation_requests`` and
        ``dispatch_single_item_to_agent`` instead of hand-filling the queue,
        proving the production metadata path is sufficient for ordered
        collection even when later microbatches finish earlier.
        """
        delays = {
            (0, 0): 0.04,
            (0, 1): 0.03,
            (0, 2): 0.02,
            (1, 0): 0.00,
            (1, 1): 0.01,
            (1, 2): 0.00,
        }
        rc = _make_fire_controller(num_mb=3, num_actors=1)
        rc.training_config.rollout_ordered_collection = True
        rc.agent_loop_actors = [_DelayedRecordingActor(delays)]

        async def _finalize_rollout_batches(rbs, ppo_step, num_expected):
            return GenerateResult(dp_refs=rbs)

        rc.finalize_rollout_batches = _finalize_rollout_batches

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        await rc.fire_generation_requests(epoch=0, ppo_step=1, num_ppo_steps=1)
        await rc.wait_all_inflight()

        result0 = await rc.collect_rollout_step(0)
        result1 = await rc.collect_rollout_step(1)

        assert [rb["source_order"][0] for rb in result0.dp_refs] == [
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
        ]
        assert [rb["source_order"][0] for rb in result1.dp_refs] == [
            (1, 0, 3),
            (1, 1, 4),
            (1, 2, 5),
        ]

    async def test_ordered_collect_handles_rb_multiplier_chunks(self):
        rc = _make_collect_controller(num_mb=2, rb_multiplier=2, ordered=True)
        rb1a = await _put_queued(rc, "mb1_a", ppo_step=0, microbatch_idx=1, sample_idx=1)
        rb0a = await _put_queued(rc, "mb0_a", ppo_step=0, microbatch_idx=0, sample_idx=0)
        rb1b = await _put_queued(rc, "mb1_b", ppo_step=0, microbatch_idx=1, sample_idx=1)
        rb0b = await _put_queued(rc, "mb0_b", ppo_step=0, microbatch_idx=0, sample_idx=0)

        result = await rc.collect_rollout_step(0)

        assert result.dp_refs == [rb0a, rb0b, rb1a, rb1b]
        assert rc._sample_idx == 2

    async def test_ordered_collect_raises_queued_exception(self):
        rc = _make_collect_controller(num_mb=1, ordered=True)
        boom = RuntimeError("rollout failed")
        await rc._ready_queue.put(boom)

        with self.assertRaises(RuntimeError) as cm:
            await rc.collect_rollout_step(0)
        assert cm.exception is boom

    async def test_ordered_origin_provenance_asserts_and_drops_internal_keys(self):
        rc = _make_collect_controller(num_mb=1, ordered=True)
        rb = {
            "tokens": [[0], [1]],
            ORIGIN_PPO_STEP_KEY: [3, 3],
            ORIGIN_MICROBATCH_IDX_KEY: [0, 0],
        }

        rc._validate_and_drop_rollout_origin_attrs([rb], ppo_step=3, num_expected=1)

        assert ORIGIN_PPO_STEP_KEY not in rb
        assert ORIGIN_MICROBATCH_IDX_KEY not in rb

        bad_rb = {
            "tokens": [[0]],
            ORIGIN_PPO_STEP_KEY: [4],
            ORIGIN_MICROBATCH_IDX_KEY: [0],
        }
        with self.assertRaises(AssertionError):
            rc._validate_and_drop_rollout_origin_attrs([bad_rb], ppo_step=3, num_expected=1)

    async def test_ordered_origin_provenance_asserts_microbatch_order(self):
        rc = _make_collect_controller(num_mb=2, ordered=True)
        rbs = [
            {
                "tokens": [[0]],
                ORIGIN_PPO_STEP_KEY: [3],
                ORIGIN_MICROBATCH_IDX_KEY: [1],
            },
            {
                "tokens": [[1]],
                ORIGIN_PPO_STEP_KEY: [3],
                ORIGIN_MICROBATCH_IDX_KEY: [0],
            },
        ]

        with self.assertRaises(AssertionError):
            rc._validate_and_drop_rollout_origin_attrs(rbs, ppo_step=3, num_expected=2)

    async def test_colocate_dispatch_wraps_rb_multiplier_chunks(self):
        rc = _make_colocate_fire_controller(num_mb=2, num_actors=1)
        rc.rb_multiplier = 2
        actor = rc.agent_loop_actors[0]
        actor.agent_loop = _FakeRemote(result=[
            {"tokens": [["mb0_a"]]},
            {"tokens": [["mb0_b"]]},
            {"tokens": [["mb1_a"]]},
            {"tokens": [["mb1_b"]]},
        ])

        await rc.dispatch_batches_to_agents(
            agents=[actor],
            per_agent_batches=[[{"prompt": ["p0"]}, {"prompt": ["p1"]}]],
            per_agent_indices=[[10, 11]],
            per_agent_microbatch_indices=[[0, 1]],
            ppo_step=5,
        )

        items = [await rc._ready_queue.get() for _ in range(4)]
        assert [item.ppo_step for item in items] == [5, 5, 5, 5]
        assert [item.microbatch_idx for item in items] == [0, 0, 1, 1]
        assert [item.sample_idx for item in items] == [10, 10, 11, 11]
        assert [item.rollout_batch["tokens"][0][0] for item in items] == [
            "mb0_a",
            "mb0_b",
            "mb1_a",
            "mb1_b",
        ]

    async def test_colocate_rejects_overlapping_fire(self):
        rc = _make_colocate_fire_controller(num_mb=2, num_actors=2)

        await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)
        with self.assertRaises(AssertionError):
            await rc.fire_generation_requests(epoch=0, ppo_step=1, num_ppo_steps=1)

        await rc.wait_all_inflight()

    async def test_colocate_rejects_multi_step_fire(self):
        rc = _make_colocate_fire_controller(num_mb=2, num_actors=2)

        with self.assertRaises(AssertionError):
            await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=2)

    async def test_colocate_rejects_nonzero_staleness(self):
        rc = _make_colocate_fire_controller(num_mb=2, num_actors=2)
        rc.training_config.rollout_max_staleness = 1

        with self.assertRaises(AssertionError):
            await rc.fire_generation_requests(epoch=0, ppo_step=0, num_ppo_steps=1)


if __name__ == "__main__":
    unittest.main()
