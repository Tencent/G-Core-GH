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
    RolloutController,
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
    rc._inflight_tasks = []
    rc.use_colocate = False
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
        assert first is rb_a
        assert second is rb_b

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
    rc.data_source = _FakeDataSource()
    rc.apply_sampling_rollout_attr = _NoopRolloutAttr()
    rc.agent_loop_actors = [_RecordingActor() for _ in range(num_actors)]
    rc.config = SimpleNamespace(placement_type="disaggregated")
    rc.training_config = SimpleNamespace(single_controller=True)
    return rc


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
    rc.data_source = _FakeDataSource()
    rc.apply_sampling_rollout_attr = _NoopRolloutAttr()
    rc.agent_loop_actors = [_RecordingColocateActor() for _ in range(num_actors)]
    rc.config = SimpleNamespace(placement_type="colocate")
    rc.training_config = SimpleNamespace(
        single_controller=True,
        rollout_max_staleness=0,
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
        assert rc._next_fire_sample_idx == 1
        assert rc._next_fire_actor_idx == 1

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
