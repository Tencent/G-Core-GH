"""Unit tests for ``SendRequestMixin.sampler_gen_out`` (streaming external reward).

``mixin.py`` is loaded standalone (deps: asyncio, typing only), so these run with
bare ``python3`` — no torch, no ray, no GPU.  They assert the properties the
``on_ready`` branch must guarantee: index-ordered results despite out-of-order
generation, one callback per micro-batch in completion order, per-index reward
collection on the actor side, that the reward coroutines are actually submitted
before rollout returns, and that the batch handed to the callback carries the
attrs the sampler strip removed.

Run:  cd .../gcore-dev && python3 tests/test_gpatch_v4/test_stream_reward_sampler.py
"""
import asyncio
import importlib.util
import os
import unittest

_MIXIN = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "gpatch_v4", "rollout_generator", "mixin.py")
)
_spec = importlib.util.spec_from_file_location("mixin_standalone", _MIXIN)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SendRequestMixin = _mod.SendRequestMixin
build_stream_ready_view = _mod.build_stream_ready_view


class _FakeSampler:
    """generate() returns a coroutine that finishes after delays[i] seconds."""
    def __init__(self, delays):
        self.delays = delays

    async def generate(
        self, sampler_idx, curr_ppo_step, sidx, rollout_batch, repeat_n, load_aware, is_eval=False
    ):
        i = rollout_batch["_i"]
        await asyncio.sleep(self.delays[i])
        return {"_i": i, "sidx": sidx}


class _Cfg:
    class training:
        load_aware_sampler_routing = False


class _Harness(SendRequestMixin):
    def __init__(self, delays):
        self.sampler_client = _FakeSampler(delays)
        self.config = _Cfg()
        # !1709 (68a5bb9) made fire_generate read self.run_eval; without it every
        # test in this file has errored since 2026-08-11.  The generators set it
        # from their run_eval argument, and these are train-path harnesses.
        self.run_eval = False


def _reversed_delays(n):
    # mb0 slowest ... mb(n-1) fastest  => completion order = reversed(range(n))
    return [(n - i) * 0.02 for i in range(n)]


class StreamSamplerTest(unittest.IsolatedAsyncioTestCase):

    async def test_results_in_index_order_despite_out_of_order_completion(self):
        n = 6
        h = _Harness(_reversed_delays(n))
        rbs = [{"_i": i} for i in range(n)]
        fired = []

        def on_ready(rbi, res):
            fired.append(rbi)
            self.assertEqual(res["_i"], rbi, "callback got mismatched result")

        results = await h.sampler_gen_out(rbs, 0, 0, 100, 1, on_ready=on_ready)

        # (1) reassembled in index order
        self.assertEqual([r["_i"] for r in results], list(range(n)))
        # sidx offset preserved per index
        self.assertEqual([r["sidx"] for r in results], [100 + i for i in range(n)])
        # (2) fired once per micro-batch, in COMPLETION order (reversed)
        self.assertEqual(fired, list(range(n))[::-1])
        self.assertEqual(sorted(fired), list(range(n)))

    async def test_none_callback_matches_gather(self):
        n = 4
        h = _Harness([0.01] * n)
        rbs = [{"_i": i} for i in range(n)]
        results = await h.sampler_gen_out(rbs, 0, 0, 0, 1, on_ready=None)
        self.assertEqual([r["_i"] for r in results], list(range(n)))

    async def test_actor_collect_reward_per_index(self):
        """Mirror the actor: on_ready fires a per-mb reward future; collect them
        in index order at the end. Reward for mb i must land at position i even
        though generation finished in reverse order."""
        n = 5
        h = _Harness(_reversed_delays(n))
        rbs = [{"_i": i} for i in range(n)]
        futs = {}

        async def _mock_reward(batch_list):
            # emulate a per-sample external reward: returns [update] for the 1 batch
            rb = batch_list[0]
            await asyncio.sleep(0.005)
            return [{"external_reward": rb["_i"] * 10}]

        def on_ready(rbi, res, _futs=futs):
            _futs[rbi] = asyncio.ensure_future(_mock_reward([res]))

        results = await h.sampler_gen_out(rbs, 0, 0, 0, 1, on_ready=on_ready)

        ordered = [futs[i] for i in range(len(results))]
        nested = await asyncio.gather(*ordered)
        updates = [x[0] for x in nested]

        # reward for micro-batch i is at index i, value i*10
        self.assertEqual([u["external_reward"] for u in updates],
                         [i * 10 for i in range(n)])
        self.assertEqual(set(futs.keys()), set(range(n)))

    async def test_reward_tasks_submitted_before_return(self):
        """When several generations are already done, as_completed can drain them
        without yielding, leaving the reward coroutines unsubmitted when rollout
        returns -- no overlap at all."""
        n = 4
        h = _Harness([0.0] * n)  # all micro-batches finish back-to-back
        rbs = [{"_i": i} for i in range(n)]
        submitted = []

        async def _mock_reward(rbi):
            submitted.append(rbi)
            await asyncio.sleep(0.005)

        def on_ready(rbi, res):
            asyncio.ensure_future(_mock_reward(rbi))

        await h.sampler_gen_out(rbs, 0, 0, 0, 1, on_ready=on_ready)

        # every reward request was actually submitted while still inside rollout
        self.assertEqual(sorted(submitted), list(range(n)))


class StreamReadyViewTest(unittest.TestCase):
    """The batch handed to on_ready must look like the one the batched path scores.

    remove_rollout_attr_before_sampling deletes every key listed in cache_keys
    (production caches question / gt_label / messages there) and
    add_back_rollout_attr_after_sampling only restores them after generation, so
    without the view a streamed reward sees a batch missing exactly the fields it
    reads.
    """
    def test_stripped_attrs_are_visible_again(self):
        cached = {"u0": {"question": "q0", "gt_label": 0.0},
                  "u1": {"question": "q1", "gt_label": 1.0}}
        generated = {"tokens": [[1, 2], [3, 4]], "unique_id": ["u0", "u1"]}

        view = build_stream_ready_view(generated, cached)

        self.assertEqual(view["question"], ["q0", "q1"])
        self.assertEqual(view["gt_label"], [0.0, 1.0])
        self.assertEqual(view["tokens"], [[1, 2], [3, 4]])

    def test_repeated_samples_get_one_value_each(self):
        """sampling_repeat_n makes the sampler return the same unique_id N times;
        the add-back appends once per entry, so the view must too."""
        cached = {"u0": {"gt_label": 7.0}}
        generated = {"tokens": [[1], [2], [3], [4]], "unique_id": ["u0"] * 4}

        view = build_stream_ready_view(generated, cached)

        self.assertEqual(view["gt_label"], [7.0] * 4)

    def test_parent_unique_id_is_the_fallback_lookup(self):
        cached = {"p0": {"gt_label": 3.0}}
        generated = {
            "tokens": [[1], [2]],
            "unique_id": ["c0", "c1"],
            "parent_unique_id": ["p0", "p0"],
        }

        view = build_stream_ready_view(generated, cached)

        self.assertEqual(view["gt_label"], [3.0, 3.0])

    def test_sampling_only_keys_are_dropped(self):
        generated = {
            "tokens": [[1]],
            "unique_id": ["u0"],
            "parent_unique_id": ["p0"],
            "cache_keys": ["question"],
        }

        view = build_stream_ready_view(generated, {"u0": {"question": "q0"}})

        # the batched path pops these before the reward ever sees the batch
        for key in ("unique_id", "parent_unique_id", "cache_keys"):
            self.assertNotIn(key, view)

    def test_generated_batch_is_not_mutated(self):
        generated = {"tokens": [[1]], "unique_id": ["u0"]}

        build_stream_ready_view(generated, {"u0": {"question": "q0"}})

        self.assertNotIn("question", generated)
        self.assertIn("unique_id", generated)

    def test_nothing_cached_is_a_passthrough(self):
        generated = {"tokens": [[1]], "gt_label": [1.0]}

        view = build_stream_ready_view(generated, {})

        self.assertEqual(view, generated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
