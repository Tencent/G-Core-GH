"""Unit tests for streaming external reward on the agent-loop path.

This is the ``single_controller`` / ``async_rollout`` path (``RolloutController``
-> ``AgentLoopActor`` / ``EnvAgentLoopActor``); the sync-path counterpart lives
in ``test_stream_reward_sampler.py``.  The heavy deps are stubbed, so these run
under bare ``python3`` with no GPU.

Run:  cd .../gcore-dev && python3 tests/test_gpatch_v4/test_stream_reward_agent_loop.py
"""
import asyncio
import importlib.util
import os
import sys
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _stub_if_missing(name, **attrs):
    """Prefer the real package where it exists.

    A stub only has to stand in on a machine without the training deps. Where
    they are installed (the cluster image), stubbing is actively wrong: under
    Python < 3.14 annotations are evaluated at def time, so a bare ``torch``
    stub breaks ``data_manipulate_utils``' ``torch.Tensor`` annotation.
    """
    try:
        importlib.import_module(name)
    except Exception:
        _stub(name, **attrs)


def _load(dotted, relpath):
    path = os.path.join(_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs():
    """Stub only the heavy deps; the modules under test are loaded for real."""
    _stub_if_missing("torch", Tensor=object)
    _stub_if_missing("tensordict", TensorDict=object)
    _stub_if_missing("transformers", AutoTokenizer=object)
    _stub("gpatch_v4")
    _stub("gpatch_v4.client", GenRmClient=object, SamplerClient=object)
    _stub("gpatch_v4.configs")
    _stub("gpatch_v4.configs.config", RlConfig=object)
    _stub("gpatch_v4.rollout_generator")
    _stub("gpatch_v4.rollout_generator.async_rollout")
    _stub(
        "gpatch_v4.rollout_generator.async_rollout.mixin",
        ColocateAgentMixin=type(
            "ColocateAgentMixin", (), {"_init_colocate_state": lambda self: None}
        ),
    )
    _stub("gpatch_v4.agentic")
    _stub("gpatch_v4.agentic.env", register_custom_env_for_agentic=lambda cfg: None)
    _stub("gpatch_v4.reward")  # placeholder for the package path
    _stub(
        "gpatch_v4.utils", **{
            "import_fn_from_path": lambda *a, **k: None,
            "safe_import_class": lambda *a, **k: None,
            "log": lambda *a, **k: None,
            "log_debug": lambda *a, **k: None,
        }
    )
    _stub("gpatch_v4.utils.data_manipulate_utils")  # placeholder for the package path


_install_stubs()
# real, not stubbed: the streaming callback must fill exactly what the batched path does
_dmu = _load("gpatch_v4.utils.data_manipulate_utils", "gpatch_v4/utils/data_manipulate_utils.py")
ensure_sample_hierarchical_id = _dmu.ensure_sample_hierarchical_id

# real, not stubbed: the concurrency opt-in's DEFAULT is one of the things under
# test, and a stub would let it drift from the base class
_ber = _load("gpatch_v4.reward.base_external_reward", "gpatch_v4/reward/base_external_reward.py")
declares_concurrent_calls = _ber.declares_concurrent_calls
stream_external_reward_declined_reason = _ber.stream_external_reward_declined_reason

_ala = _load(
    "gpatch_v4.rollout_generator.async_rollout.agent_loop_actor",
    "gpatch_v4/rollout_generator/async_rollout/agent_loop_actor.py",
)
_eala = _load(
    "gpatch_v4.rollout_generator.async_rollout.env_agent_loop_actor",
    "gpatch_v4/rollout_generator/async_rollout/env_agent_loop_actor.py",
)
AgentLoopActor = _ala.AgentLoopActor
EnvAgentLoopActor = _eala.EnvAgentLoopActor

# --------------------------------------------------------------------------- #
#  Fakes.  A shared clock records the order events happened in, which is what
#  the overlap assertions are actually about.
# --------------------------------------------------------------------------- #


class _Trace:
    def __init__(self):
        self.events = []

    def record(self, what):
        self.events.append(what)

    def index_of(self, what):
        return self.events.index(what)

    def __contains__(self, what):
        return what in self.events


class _FakeSampler:
    """fire_generate returns a ref immediately; await_generate waits delays[i]."""
    def __init__(self, delays, trace):
        self.delays = delays
        self.trace = trace

    async def fire_generate(
        self, sampler_idx, ppo_step, sidx, cleaned_data, repeat_n, load_aware=False
    ):
        return {"_i": cleaned_data["_i"], "unique_id": cleaned_data["unique_id"]}

    async def await_generate(self, ref):
        i = ref["_i"]
        await asyncio.sleep(self.delays[i])
        self.trace.record(("gen_done", i))
        return {"_i": [i], "unique_id": list(ref["unique_id"])}


class _FakeReward:
    """Records when each micro-batch's reward call starts and finishes."""

    # honest: every call keeps its state in locals and appends to plain lists,
    # so overlapping calls on one instance cannot corrupt each other
    supports_concurrent_calls = True

    def __init__(self, trace, delay=0.0, fail_on=None):
        self.trace = trace
        self.delay = delay
        self.fail_on = fail_on
        self.calls = []
        self.cancelled = []

    async def calc_external_reward(self, rollout_batches, ppo_step, is_eval=False):
        assert len(rollout_batches) == 1, "agent-loop path scores one batch per call"
        i = rollout_batches[0]["_i"][0]
        self.calls.append(i)
        self.trace.record(("reward_start", i))
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.append(i)
            raise
        if self.fail_on is not None and i == self.fail_on:
            raise RuntimeError(f"reward backend refused micro-batch {i}")
        self.trace.record(("reward_done", i))
        return [{"external_reward": [float(i)]}]


class _Cfg:
    def __init__(self, stream, use_gen_rm=False, repeat_n=1):
        self.stream_external_reward = stream
        self.use_gen_rm_reward = use_gen_rm
        self.use_external_reward = True
        self.sampling_repeat_n = repeat_n
        self.load_aware_sampler_routing = False


def _make_actor(cls, delays, trace, stream=True, use_gen_rm=False, reward=None):
    actor = object.__new__(cls)
    actor.config = None
    actor.training_config = _Cfg(stream, use_gen_rm=use_gen_rm)
    actor.worker_id = 0
    actor.tokenizer = None
    actor.gen_rm_client = None
    actor.sampler_client = _FakeSampler(delays, trace)
    actor.external_reward = reward if reward is not None else _FakeReward(trace)
    actor._stream_external_reward = actor._resolve_stream_external_reward()
    return actor


def _batches(n):
    return [{"_i": i, "unique_id": [f"u{i}"]} for i in range(n)]


async def _old_signature_generate_batches(cleaned_batches, ppo_step, sample_indices, use_colocate):
    """A custom agent_loop_actor_cls override written before on_ready existed."""
    return [{"_i": [b["_i"]], "unique_id": list(b["unique_id"])} for b in cleaned_batches]


# --------------------------------------------------------------------------- #
#  AgentLoopActor — sampler-driven generation
# --------------------------------------------------------------------------- #


class AgentLoopStreamingTest(unittest.IsolatedAsyncioTestCase):
    async def test_reward_fires_while_later_microbatches_still_generate(self):
        """mb0's reward starts before mb2 has finished generating."""
        trace = _Trace()
        delays = [0.01, 0.05, 0.12]
        actor = _make_actor(AgentLoopActor, delays, trace)

        await actor._batch_loop(
            _batches(3), ppo_step=0, sample_indices=[0, 1, 2], use_colocate=False
        )

        self.assertLess(trace.index_of(("reward_start", 0)), trace.index_of(("gen_done", 2)))
        self.assertLess(trace.index_of(("reward_start", 1)), trace.index_of(("gen_done", 2)))

    async def test_batched_path_scores_only_after_all_generation(self):
        """Flag off: control flow is the pre-change one, no reward before the last gen."""
        trace = _Trace()
        delays = [0.01, 0.05, 0.12]
        actor = _make_actor(AgentLoopActor, delays, trace, stream=False)

        await actor._batch_loop(
            _batches(3), ppo_step=0, sample_indices=[0, 1, 2], use_colocate=False
        )

        last_gen = trace.index_of(("gen_done", 2))
        for i in range(3):
            self.assertGreater(trace.index_of(("reward_start", i)), last_gen)

    async def test_updates_stay_pinned_to_their_microbatch(self):
        """Out-of-order completion must not shuffle rewards onto the wrong batch."""
        trace = _Trace()
        # mb0 slowest, mb3 fastest => completion order is reversed
        delays = [0.08, 0.06, 0.04, 0.01]
        actor = _make_actor(AgentLoopActor, delays, trace)

        rbs = await actor._batch_loop(
            _batches(4), ppo_step=0, sample_indices=[0, 1, 2, 3], use_colocate=False
        )

        self.assertEqual([rb["_i"][0] for rb in rbs], [0, 1, 2, 3])
        for i, rb in enumerate(rbs):
            self.assertEqual(rb["external_reward"], [float(i)])
        self.assertEqual(actor.external_reward.calls, [3, 2, 1, 0])

    async def test_streaming_and_batched_produce_the_same_batches(self):
        """Same inputs, same reward: the two paths must agree field for field."""
        streamed = await _make_actor(AgentLoopActor, [0.03, 0.01, 0.02], _Trace())._batch_loop(
            _batches(3), ppo_step=7, sample_indices=[0, 1, 2], use_colocate=False
        )
        batched = await _make_actor(AgentLoopActor, [0.03, 0.01, 0.02], _Trace(),
                                    stream=False)._batch_loop(
                                        _batches(3),
                                        ppo_step=7,
                                        sample_indices=[0, 1, 2],
                                        use_colocate=False
                                    )

        self.assertEqual(streamed, batched)

    async def test_hierarchical_ids_filled_before_the_reward_sees_the_batch(self):
        """The batched path fills these before scoring; streaming must too."""
        trace = _Trace()
        seen = {}

        class _Recording(_FakeReward):
            async def calc_external_reward(self, rollout_batches, ppo_step, is_eval=False):
                rb = rollout_batches[0]
                seen[rb["_i"][0]] = {
                    k: list(v)
                    for k, v in rb.items() if k in ("group_id", "traj_id", "segment_id")
                }
                return await super().calc_external_reward(rollout_batches, ppo_step, is_eval)

        actor = _make_actor(AgentLoopActor, [0.01, 0.02], trace, reward=_Recording(trace))
        await actor._batch_loop(_batches(2), ppo_step=0, sample_indices=[0, 1], use_colocate=False)

        for i in range(2):
            self.assertEqual(seen[i]["group_id"], [f"u{i}"])
            self.assertEqual(seen[i]["traj_id"], [0])
            self.assertEqual(seen[i]["segment_id"], [0])

    async def test_gen_rm_declines_streaming_rather_than_reordering(self):
        trace = _Trace()
        actor = _make_actor(AgentLoopActor, [0.01, 0.02], trace, use_gen_rm=True)
        self.assertFalse(actor._stream_external_reward)

    async def test_reward_that_never_declared_concurrency_declines_streaming(self):
        """Default-deny: a reward that never opted in keeps the batched path."""
        trace = _Trace()
        reward = _FakeReward(trace)
        # what BaseExternalReward hands a subclass that says nothing
        type(reward).supports_concurrent_calls = False
        try:
            actor = _make_actor(AgentLoopActor, [0.01, 0.02], trace, reward=reward)
            self.assertFalse(actor._stream_external_reward)
        finally:
            type(reward).supports_concurrent_calls = True

    async def test_reward_failure_propagates_and_cancels_siblings(self):
        trace = _Trace()
        reward = _FakeReward(trace, delay=0.2, fail_on=0)
        # mb0 lands first and fails; mb1/mb2 are still in flight at collect time
        actor = _make_actor(AgentLoopActor, [0.01, 0.02, 0.03], trace, reward=reward)

        with self.assertRaises(RuntimeError):
            await actor._batch_loop(
                _batches(3), ppo_step=0, sample_indices=[0, 1, 2], use_colocate=False
            )

        # no yield first: cleanup drains, so the siblings are already done
        self.assertEqual(sorted(reward.cancelled), [1, 2])

    async def test_generation_failure_cancels_already_fired_rewards(self):
        trace = _Trace()
        reward = _FakeReward(trace, delay=0.5)
        actor = _make_actor(AgentLoopActor, [0.01, 0.02], trace, reward=reward)

        async def _boom(ref):
            i = ref["_i"]
            await asyncio.sleep(actor.sampler_client.delays[i])
            trace.record(("gen_done", i))
            if i == 1:
                raise RuntimeError("sampler died mid-generation")
            return {"_i": [i], "unique_id": list(ref["unique_id"])}

        actor.sampler_client.await_generate = _boom

        with self.assertRaises(RuntimeError):
            await actor._batch_loop(
                _batches(2), ppo_step=0, sample_indices=[0, 1], use_colocate=False
            )

        # no yield first: cleanup drains, so mb0's reward is already done
        self.assertEqual(reward.cancelled, [0])

    async def test_old_signature_override_still_works_when_streaming_is_off(self):
        trace = _Trace()
        actor = _make_actor(AgentLoopActor, [0.01, 0.02], trace, stream=False)
        actor.generate_batches = _old_signature_generate_batches

        rbs = await actor._batch_loop(
            _batches(2), ppo_step=0, sample_indices=[0, 1], use_colocate=False
        )
        self.assertEqual([rb["external_reward"] for rb in rbs], [[0.0], [1.0]])

    async def test_old_signature_override_declines_streaming_instead_of_raising(self):
        """agent_loop_actor_cls is a config hook, so an override can predate on_ready.
        Passing the callback to one raises TypeError mid-step; refuse at setup."""
        trace = _Trace()
        actor = _make_actor(AgentLoopActor, [0.01, 0.02], trace, stream=True)
        actor.generate_batches = _old_signature_generate_batches
        actor._stream_external_reward = actor._resolve_stream_external_reward()
        self.assertFalse(actor._stream_external_reward)

        rbs = await actor._batch_loop(
            _batches(2), ppo_step=0, sample_indices=[0, 1], use_colocate=False
        )
        self.assertEqual([rb["external_reward"] for rb in rbs], [[0.0], [1.0]])

    async def test_generation_failure_does_not_leave_siblings_unowned(self):
        """as_completed schedules every await; an early exit must not let the rest
        outlive the step with their exceptions never retrieved."""
        trace = _Trace()
        # mb1 raises at ~0.02s; mb2 would finish on its own at ~0.06s, so waiting
        # past that is what distinguishes "cancelled" from "still running"
        actor = _make_actor(AgentLoopActor, [0.01, 0.02, 0.06], trace)
        finished_late = []

        async def _await(ref):
            i = ref["_i"]
            try:
                await asyncio.sleep(actor.sampler_client.delays[i])
            except asyncio.CancelledError:
                raise
            if i == 1:
                raise RuntimeError("sampler died mid-generation")
            finished_late.append(i)
            return {"_i": [i], "unique_id": list(ref["unique_id"])}

        actor.sampler_client.await_generate = _await

        with self.assertRaises(RuntimeError):
            await actor._batch_loop(
                _batches(3), ppo_step=0, sample_indices=[0, 1, 2], use_colocate=False
            )

        # mb2 was still generating; it must be cancelled, not left running
        await asyncio.sleep(0.2)
        self.assertNotIn(2, finished_late)

    async def test_missing_reward_task_is_caught_not_silently_dropped(self):
        """A generate_batches that yields more batches than it fired on_ready for."""
        trace = _Trace()
        actor = _make_actor(AgentLoopActor, [0.01], trace)
        rbs = [{"_i": [0], "unique_id": ["u0"]}, {"_i": [1], "unique_id": ["u1"]}]
        futs = {0: asyncio.ensure_future(asyncio.sleep(0, result=[{}]))}

        # RuntimeError, not assert: -O must not strip an alignment check
        with self.assertRaises(RuntimeError):
            await actor._collect_streamed_external_rewards(rbs, futs, ppo_step=0)
        await actor._cancel_stream_reward_futs(futs)


# --------------------------------------------------------------------------- #
#  EnvAgentLoopActor — env trajectories, one prompt group per micro-batch,
#  the groups themselves running concurrently
# --------------------------------------------------------------------------- #


class EnvAgentLoopStreamingTest(unittest.IsolatedAsyncioTestCase):
    def _make_env_actor(self, n, trace, stream=True, reward=None, env_delay=0.03):
        """``env_delay`` is per micro-batch when a list, uniform when a scalar."""
        actor = object.__new__(EnvAgentLoopActor)
        actor.config = None
        actor.training_config = _Cfg(stream)
        actor.worker_id = 0
        actor.tokenizer = None
        actor.gen_rm_client = None
        actor.sampler_client = None
        actor.external_reward = reward if reward is not None else _FakeReward(trace)
        actor._stream_external_reward = actor._resolve_stream_external_reward()

        # one manager per micro-batch; _merge_env_results is bypassed so the test
        # does not depend on torch-backed segment merging
        actor._build_managers_for_step = lambda: [object()]

        delays = env_delay if isinstance(env_delay, list) else [env_delay] * n

        async def _run_env(mgr, seed, ppo_step, cd):
            trace.record(("env_start", cd["_i"]))
            await asyncio.sleep(delays[cd["_i"]])
            trace.record(("env_done", cd["_i"]))
            return {"_i": [cd["_i"]], "unique_id": list(cd["unique_id"])}

        actor._run_env_in_thread = _run_env

        def _merge(results, cd, repeat_n, group_id):
            # the real _merge_env_results assigns these; the END log reads them
            merged = dict(results[0])
            merged["tokens"] = [[0]]
            merged["group_id"] = list(merged["unique_id"])
            merged["traj_id"] = [0]
            merged["segment_id"] = [0]
            return merged

        actor._merge_env_results = _merge
        actor.sampler_phase = None
        return actor

    async def test_reward_of_one_microbatch_overlaps_the_next_env_run(self):
        """The win: mb0 scores while mb1's trajectories are still running."""
        trace = _Trace()
        actor = self._make_env_actor(2, trace, env_delay=[0.01, 0.06])

        await actor._batch_loop(_batches(2), ppo_step=0, sample_indices=[0, 1], use_colocate=False)

        self.assertLess(trace.index_of(("reward_start", 0)), trace.index_of(("env_done", 1)))

    async def test_batched_path_waits_for_every_env_run(self):
        trace = _Trace()
        actor = self._make_env_actor(2, trace, stream=False, env_delay=[0.01, 0.06])

        await actor._batch_loop(_batches(2), ppo_step=0, sample_indices=[0, 1], use_colocate=False)

        self.assertGreater(trace.index_of(("reward_start", 0)), trace.index_of(("env_done", 1)))

    async def test_env_updates_land_in_microbatch_order(self):
        trace = _Trace()
        actor = self._make_env_actor(3, trace)

        rbs = await actor._batch_loop(
            _batches(3), ppo_step=0, sample_indices=[0, 1, 2], use_colocate=False
        )

        self.assertEqual([rb["_i"][0] for rb in rbs], [0, 1, 2])
        for i, rb in enumerate(rbs):
            self.assertEqual(rb["external_reward"], [float(i)])


# The shared decision helper.  GrpoTrainActor consults the same one but cannot be
# imported without the training image, so nothing here pins that it still does.


class ConcurrencyOptInTest(unittest.TestCase):
    """Exhaustive over the three shapes a reward can have."""
    class _Declared:
        supports_concurrent_calls = True

    class _Refused:
        supports_concurrent_calls = False

    class _Silent:
        """Duck-typed reward that never subclassed BaseExternalReward."""

    def test_declared_may_stream(self):
        r = self._Declared()
        self.assertTrue(declares_concurrent_calls(r))
        self.assertIsNone(stream_external_reward_declined_reason(r))

    def test_explicitly_refused_may_not(self):
        r = self._Refused()
        self.assertFalse(declares_concurrent_calls(r))
        self.assertIn("supports_concurrent_calls", stream_external_reward_declined_reason(r))

    def test_reward_outside_the_base_class_is_not_assumed_safe(self):
        """`_setup_external_reward` only requires the one method, so this shape exists."""
        r = self._Silent()
        self.assertFalse(hasattr(r, "supports_concurrent_calls"))
        with self.assertRaises(AttributeError):
            declares_concurrent_calls(r)

    def test_base_class_default_is_off(self):
        """The default has to live on the base class, not only in the helper."""
        self.assertIs(_ber.BaseExternalReward.supports_concurrent_calls, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
