"""tail-batching 窗口状态机与多轮协作停止单测。

不起 ray、不用 GPU：绕过 setup 直接装配控制器 / stub actor，只验证记账与 pause 契约。
"""

import asyncio
import copy
from types import SimpleNamespace

from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
    PendingMicrobatch,
    RolloutController,
)
from gpatch_v4.rollout_generator.async_rollout.two_turn_reflect_agent_loop_actor import (
    TwoTurnReflectAgentLoopActor,
)
from gpatch_v4.utils import GenerationAborted


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeAgent:
    def __init__(self):
        self.begin_step_count = 0
        self.pause_count = 0
        self.begin_step = _Remote(self._begin_step)
        self.pause_generation = _Remote(self._pause)

    def _begin_step(self):
        self.begin_step_count += 1

    def _pause(self):
        self.pause_count += 1


class _FakeSamplerClient:
    def __init__(self):
        self.abort_count = 0

    async def abort_all(self):
        self.abort_count += 1


class _FakeGenRmClient:
    def __init__(self, num_rms=2):
        self.num_rms = num_rms
        self.aborted = []

    async def abort_all(self, rm_idx):
        self.aborted.append(rm_idx)


class _FakeDataSource:
    def __init__(self):
        self.reclaimed = []
        self.saved = None
        self.get_batch_calls = []

    def get_batch(self, num_mb):
        self.get_batch_calls.append(num_mb)
        return [{"tokens": [i]} for i in range(num_mb)]

    def add_partial_samples(self, samples):
        self.reclaimed.append(samples)

    def save(self, step, pending_unconsumed_batches=None):
        self.saved = (step, pending_unconsumed_batches)


class _FakeAttrHook:
    def __init__(self):
        self.cache = {}

    def cached_rollout_attrs(self):
        return self.cache

    def remove_rollout_attr_before_sampling(self, batched_data):
        return batched_data


def _make_controller(ratio=1.5, rb_multiplier=1, reuse=True, num_agents=2):
    ctrl = object.__new__(RolloutController)
    ctrl.apply_sampling_rollout_attr = _FakeAttrHook()
    ctrl.training_config = SimpleNamespace(
        rollout_over_dispatch_ratio=ratio,
        rollout_reuse_unused_prompts=reuse,
        rollout_ordered_collection=False,
    )
    ctrl._tail_batching = ratio > 1.0
    ctrl.rb_multiplier = rb_multiplier
    ctrl.data_source = _FakeDataSource()
    ctrl.agent_loop_actors = [_FakeAgent() for _ in range(num_agents)]
    ctrl._sampler_client = _FakeSamplerClient()
    ctrl._gen_rm_client = _FakeGenRmClient()
    ctrl._pending = {}
    ctrl._inflight_tasks = []
    ctrl._step_target_rb = 0
    ctrl._step_collected_rb = 0
    ctrl._pause_task = None
    ctrl._abort_t0 = None
    ctrl._last_inflight_done_at = None
    ctrl._fire_metrics = {}
    ctrl._discard_hook = None
    ctrl._next_prompt_idx = 0
    return ctrl


def _add_pending(ctrl, ppo_step, microbatch_idx):
    unique_id = f"uid_{ppo_step}_{microbatch_idx}"
    ctrl.apply_sampling_rollout_attr.cache[unique_id] = {}
    ctrl._pending[(ppo_step, microbatch_idx)] = PendingMicrobatch(
        raw={
            "prompt": [ppo_step, microbatch_idx],
            "dataset_attr": [microbatch_idx],
            "unique_id": [unique_id],
            "prompt_idx": [microbatch_idx],
            "cache_keys": ["dataset_attr", "prompt_idx"],
        },
    )


def test_read_and_clean_uses_unified_data_source_interface():
    ctrl = _make_controller(ratio=1.5)
    batches = ctrl.read_and_clean_batches(32, ppo_step=0)
    assert ctrl.data_source.get_batch_calls == [32]
    assert len(batches) == 32

    ctrl = _make_controller(ratio=1.0)
    batches = ctrl.read_and_clean_batches(32, ppo_step=0)
    assert ctrl.data_source.get_batch_calls == [32]
    assert len(batches) == 32


def test_begin_step_sets_target_and_reclaims_only_unused():
    ctrl = _make_controller(rb_multiplier=3)
    _add_pending(ctrl, 0, 0)
    _add_pending(ctrl, 0, 1)
    ctrl._pending[(0, 0)].rollout_consumed = True

    asyncio.run(ctrl.begin_step(4))

    assert ctrl._step_target_rb == 12
    assert ctrl._step_collected_rb == 0
    assert ctrl._pending == {}
    # 只有未被采纳的 microbatch 会被回收；controller dispatch 字段要剥掉
    samples, = ctrl.data_source.reclaimed
    assert samples == [{
        "prompt": [0, 1],
        "dataset_attr": [1],
        "cache_keys": ["dataset_attr"],
    }]
    assert "uid_0_1" not in ctrl.apply_sampling_rollout_attr.cache
    assert all(a.begin_step_count == 1 for a in ctrl.agent_loop_actors)


def test_begin_step_skips_reclaim_when_reuse_disabled():
    ctrl = _make_controller(reuse=False)
    _add_pending(ctrl, 0, 0)
    asyncio.run(ctrl.begin_step(1))
    assert ctrl.data_source.reclaimed == []
    # 不复用也必须退掉缓存，否则每个窗口都漏一批
    assert ctrl.apply_sampling_rollout_attr.cache == {}


def test_begin_step_calls_discard_hook_before_attr_release():
    """hook 只拿到未被消费的 microbatch，且 raw 里 unique_id 还没被剥掉。"""
    ctrl = _make_controller()
    calls = []
    ctrl._discard_hook = lambda samples: calls.append(copy.deepcopy(samples))
    _add_pending(ctrl, 0, 0)
    ctrl._pending[(0, 0)].rollout_consumed = True
    _add_pending(ctrl, 0, 1)

    asyncio.run(ctrl.begin_step(1))

    samples, = calls
    assert samples == [{
        "prompt": [0, 1],
        "dataset_attr": [1],
        "unique_id": ["uid_0_1"],
        "prompt_idx": [1],
        "cache_keys": ["dataset_attr", "prompt_idx"],
    }]


def test_begin_step_calls_discard_hook_when_reuse_disabled():
    """hook 与回收开关无关：reuse=False 时照样触发，且不改变丢弃行为。"""
    ctrl = _make_controller(reuse=False)
    calls = []
    ctrl._discard_hook = lambda samples: calls.append(samples)
    _add_pending(ctrl, 0, 0)

    asyncio.run(ctrl.begin_step(1))

    assert len(calls) == 1
    assert ctrl.data_source.reclaimed == []
    assert ctrl.apply_sampling_rollout_attr.cache == {}


def test_begin_step_no_discard_no_hook_call():
    ctrl = _make_controller()
    calls = []
    ctrl._discard_hook = lambda samples: calls.append(samples)
    _add_pending(ctrl, 0, 0)
    ctrl._pending[(0, 0)].rollout_consumed = True

    asyncio.run(ctrl.begin_step(1))

    assert calls == []


def test_begin_step_propagates_discard_hook_failure():
    ctrl = _make_controller()
    _add_pending(ctrl, 0, 0)

    def fail(_samples):
        raise RuntimeError("discard hook failed")

    ctrl._discard_hook = fail
    try:
        asyncio.run(ctrl.begin_step(1))
    except RuntimeError as e:
        assert str(e) == "discard hook failed"
    else:
        raise AssertionError("expected discard hook failure")


def test_reserve_accepts_until_target_then_refuses_and_pauses_once():
    ctrl = _make_controller(rb_multiplier=1)

    async def scenario():
        await ctrl.begin_step(2)
        for idx in range(3):
            _add_pending(ctrl, 0, idx)
        first = ctrl._try_accept_completed_microbatch([{"tokens": []}], 0, 0)
        second = ctrl._try_accept_completed_microbatch([{"tokens": []}], 0, 1)
        third = ctrl._try_accept_completed_microbatch([{"tokens": []}], 0, 2)
        if ctrl._pause_task is not None:
            await ctrl._pause_task
        return first, second, third

    first, second, third = asyncio.run(scenario())

    assert (first, second, third) == (True, True, False)
    assert ctrl._step_collected_rb == 2
    assert ctrl._pending[(0, 2)].rollout_consumed is False
    assert all(a.pause_count == 1 for a in ctrl.agent_loop_actors)
    assert ctrl._sampler_client.abort_count == 1
    assert ctrl._gen_rm_client.aborted == [0, 1]


def test_reserve_is_atomic_per_microbatch():
    """rb_multiplier>1 时一个 microbatch 的多个 rb 必须整组收或整组丢。"""
    ctrl = _make_controller(rb_multiplier=3)

    async def scenario():
        await ctrl.begin_step(1)
        _add_pending(ctrl, 0, 0)
        _add_pending(ctrl, 0, 1)
        rb_list = [{"tokens": []}] * 3
        first = ctrl._try_accept_completed_microbatch(rb_list, 0, 0)
        second = ctrl._try_accept_completed_microbatch(rb_list, 0, 1)
        if ctrl._pause_task is not None:
            await ctrl._pause_task
        return first, second

    first, second = asyncio.run(scenario())
    assert (first, second) == (True, False)
    assert ctrl._step_collected_rb == 3


def test_dispatch_skips_agent_after_target_is_reached():
    ctrl = _make_controller()
    ctrl._step_target_rb = 1
    ctrl._step_collected_rb = 1
    _add_pending(ctrl, 0, 0)
    calls = []
    actor = SimpleNamespace(
        agent_loop=_Remote(lambda *args: calls.append(args) or [{"tokens": []}])
    )

    asyncio.run(ctrl.dispatch_single_item_to_agent(actor, {}, 0, 0, 0))

    assert calls == []
    assert ctrl._pending[(0, 0)].rollout_consumed is False


def test_dispatch_without_tail_batching_has_no_target_limit():
    ctrl = _make_controller(ratio=1.0)
    ctrl._ready_queue = asyncio.Queue()
    calls = []
    actor = SimpleNamespace(
        agent_loop=_Remote(lambda *args: calls.append(args) or [{"tokens": []}])
    )

    async def scenario():
        await ctrl.begin_step(1)
        assert ctrl._try_accept_completed_microbatch([{"tokens": []}], 0, 0)
        await ctrl.dispatch_single_item_to_agent(actor, {}, 0, 0, 0)

    asyncio.run(scenario())

    assert ctrl._step_target_rb == 1
    assert ctrl._step_collected_rb == 0
    assert len(calls) == 1
    assert ctrl._ready_queue.qsize() == 1


def test_wait_all_inflight_drains_siblings_before_raising():
    ctrl = _make_controller()
    completed = []

    async def fail():
        await asyncio.sleep(0)
        raise RuntimeError("generation failed")

    async def complete():
        await asyncio.sleep(0.01)
        completed.append(True)

    async def scenario():
        tasks = [asyncio.create_task(fail()), asyncio.create_task(complete())]
        ctrl._inflight_tasks = tasks
        try:
            await ctrl.wait_all_inflight()
        except RuntimeError as e:
            assert str(e) == "generation failed"
        else:
            raise AssertionError("expected generation failure")
        return tasks

    tasks = asyncio.run(scenario())

    assert completed == [True]
    assert all(task.done() for task in tasks)
    assert ctrl._inflight_tasks == []


def test_save_data_source_snapshots_train_unconsumed_batches_without_mutation():
    ctrl = _make_controller()
    _add_pending(ctrl, 0, 0)
    _add_pending(ctrl, 0, 1)
    _add_pending(ctrl, 0, 2)
    ctrl._pending[(0, 0)].train_consumed = True
    original_raw = {
        key: list(value) if isinstance(value, list) else value
        for key, value in ctrl._pending[(0, 1)].raw.items()
    }

    ctrl.save_data_source(7)

    step, batches = ctrl.data_source.saved
    assert step == 7
    assert batches == [
        {
            "prompt": [0, 1],
            "dataset_attr": [1],
            "cache_keys": ["dataset_attr"],
        },
        {
            "prompt": [0, 2],
            "dataset_attr": [2],
            "cache_keys": ["dataset_attr"],
        },
    ]
    assert ctrl._pending[(0, 1)].raw == original_raw


class _AbortingAgent:
    def __init__(self):
        self.agent_loop = _Remote(self._raise)

    def _raise(self, *args, **kwargs):
        raise GenerationAborted("sglang aborted a generation request")


def test_aborted_microbatch_is_dropped_while_paused():
    ctrl = _make_controller(rb_multiplier=1)
    ctrl._ready_queue = asyncio.Queue()

    async def scenario():
        await ctrl.begin_step(1)
        _add_pending(ctrl, 0, 0)
        ctrl._pause_task = asyncio.create_task(asyncio.sleep(0))
        await ctrl.dispatch_single_item_to_agent(_AbortingAgent(), {}, 0, 0, 0)
        await ctrl._pause_task

    asyncio.run(scenario())
    assert ctrl._ready_queue.empty()
    assert ctrl._pending[(0, 0)].rollout_consumed is False


def test_abort_without_pause_is_a_real_failure():
    ctrl = _make_controller(rb_multiplier=1)
    ctrl._ready_queue = asyncio.Queue()

    async def scenario():
        await ctrl.begin_step(1)
        _add_pending(ctrl, 0, 0)
        assert ctrl._pause_task is None
        try:
            await ctrl.dispatch_single_item_to_agent(_AbortingAgent(), {}, 0, 0, 0)
        except GenerationAborted:
            return True
        return False

    assert asyncio.run(scenario()) is True
    assert isinstance(ctrl._ready_queue.get_nowait(), GenerationAborted)


class _StubTwoTurnActor(TwoTurnReflectAgentLoopActor):
    """跳过 setup 与真实采样，只保留 _batch_loop 的控制流。"""

    def __init__(self):
        self._pause_event = asyncio.Event()
        self.worker_id = 0
        self.reflect_top_k = 1
        self.turn2_started = False

    async def generate_batches(self, cleaned_batches, ppo_step, sample_indices, use_colocate):
        if self.turn1_done:
            self.turn2_started = True
        return [{"rewards": [0.0], "tokens": [[]], "unique_id": ["uid"]} for _ in cleaned_batches]

    async def score_rollout_batches(self, rbs, ppo_step, sample_indices, use_colocate):
        self.turn1_done = True
        return rbs

    def _select_topk_bottomk(self, rewards, k):
        return [0], [0]

    def _build_reflect_prompt(self, rb_turn1, sample_idx, tag):
        return {"prompt": [sample_idx]}

    turn1_done = False


def test_pause_between_turns_raises_instead_of_returning_short_list():
    # 契约是抛 GenerationAborted，不是返回短列表；否则 rb_multiplier 记账会坏。
    actor = _StubTwoTurnActor()
    actor._pause_event.set()
    try:
        asyncio.run(actor._batch_loop([{"prompt": [1]}], 0, [0], False))
    except GenerationAborted as e:
        assert "before turn 2" in str(e)
    else:
        raise AssertionError("expected GenerationAborted")
    assert actor.turn2_started is False
