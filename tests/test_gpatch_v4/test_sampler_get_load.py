"""Test GrpoSamplerActor.get_load, SamplerClient.get_all_loads, and
load-aware sampler routing.

Tests:
  1. Idle engines report zero load.
  2. During active generation, at least one engine reports non-zero load.
  3. Full async-rollout training with LoadAwareAgentLoopActor produces
     correct metrics (end-to-end validation of load-aware routing).

GPU layout (8 GPU on 1 node, TP=2 → 4 engine clusters):
    sampler: 1 node = 8 GPU (tp=2, 4 clusters)
"""

import asyncio
import os
import shutil
import unittest

import ray

from gpatch_v4 import orches
from gpatch_v4.client import SamplerClient
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.orches.placement_group import (
    create_placement_groups,
    create_sampler_group,
)
from gpatch_v4.trainer import GrpoSingleCtrlTrainer
from gpatch_v4.trainer.helper import set_nnodes_default
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_LONG_PROMPT = (
    "Write a very long and extremely detailed story about a wizard who "
    "travels through 100 different magical kingdoms. Describe every "
    "kingdom in great detail including the people, culture, food, "
    "architecture, and magical systems. "
) * 5


def _print_loads(title, loads):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for i, ld in enumerate(loads):
        print(
            f"  Cluster {i}: num_reqs={ld['num_reqs']}, "
            f"num_tokens={ld['num_tokens']}, "
            f"num_waiting_reqs={ld['num_waiting_reqs']}"
        )
    print("=" * 60)


def _assert_load_schema(loads, num_clusters):
    assert len(loads) == num_clusters, (f"expected {num_clusters} load entries, got {len(loads)}")
    for i, ld in enumerate(loads):
        for key in ("num_reqs", "num_tokens", "num_waiting_reqs"):
            assert key in ld, f"cluster {i}: missing '{key}'"
            assert isinstance(ld[key], int), f"cluster {i}: {key} not int"


@requires_sglang
class SamplerGetLoadTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def test_get_load_idle_and_under_generation(self):
        """Phase 1: idle → zero load.  Phase 2: during generation → non-zero."""
        config = load_config("test_math_rl_async_rollout_rule_only", RlConfig)
        config.sampler.backend = "sglang"
        assert config.placement_type == "disaggregated"

        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        sampler_group = create_sampler_group(config, pgs)
        await sampler_group.init()

        client = SamplerClient(config, dp_rank=0, dp_size=1)
        num_clusters = client.svr_cluster_num_per_sampler[0]

        # ---- Phase 1: idle load ----
        loads = await client.get_all_loads(sampler_idx=0)
        _assert_load_schema(loads, num_clusters)
        for i, ld in enumerate(loads):
            assert ld["num_reqs"] == 0, f"cluster {i}: expected 0 reqs, got {ld['num_reqs']}"
            assert ld["num_tokens"] == 0, f"cluster {i}: expected 0 tokens, got {ld['num_tokens']}"
        _print_loads("Phase 1 — Idle Load", loads)

        # ---- Phase 2: load under generation ----
        # Fire test_generate on cluster 0 with max_tokens=16384 so the
        # generation takes long enough for us to observe non-zero load.
        actor_0 = sampler_group._actor_handlers[0][0]
        gen_ref = actor_0.test_generate.remote(
            {
                "prompts": [_LONG_PROMPT] * 8,
                "max_tokens": 16384,
            }
        )

        await asyncio.sleep(2.0)

        loads = await client.get_all_loads(sampler_idx=0)
        _assert_load_schema(loads, num_clusters)
        _print_loads("Phase 2 — Load Under Generation", loads)

        ld0 = loads[0]
        assert ld0["num_reqs"] > 0, (
            f"cluster 0: expected num_reqs > 0 during generation, got {ld0['num_reqs']}"
        )
        assert ld0["num_tokens"] > 0, (
            f"cluster 0: expected num_tokens > 0 during generation, got {ld0['num_tokens']}"
        )

        await gen_ref
        print("  Generation completed.\n")

    async def test_train_async_rollout_load_aware(self):
        """Full async-rollout training with load-aware sampler routing."""
        # 加一个 check 保证 workload 散出去了
        config = load_config("test_math_rl_async_rollout_rule_only", RlConfig)
        config.sampler.backend = "sglang"
        assert config.placement_type == "disaggregated"
        assert config.training.async_rollout

        config.training.load_aware_sampler_routing = True
        config.training.sampling_repeat_n = 8
        config.training.sampling_keep_n = 8
        config.training.train_gbs = 128

        tmp_dir = "tests/test_gpatch_v4/gsm8k_small"
        os.makedirs(tmp_dir, exist_ok=True)
        src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
        dst = os.path.join(tmp_dir, "train.jsonl")
        with open(src, "r") as fin, open(dst, "w") as fout:
            for i, line in enumerate(fin):
                if i >= 64:
                    break
                fout.write(line)

        shutil.rmtree(config.checkpoint.load_ckpt_path, ignore_errors=True)
        try:
            trainer = GrpoSingleCtrlTrainer()
            metrics = await trainer.launch_then_run_with_recovery(config)
        finally:
            shutil.rmtree(config.checkpoint.save_ckpt_path, ignore_errors=True)

        assert metrics is not None
        assert len(metrics) >= 1

        for dp_metrics in metrics:
            assert len(dp_metrics) >= 1
            for step_metric in dp_metrics:
                assert "policy/loss" in step_metric
                assert "policy/grad_norm" in step_metric
                assert "policy/ppo_ratio" in step_metric

                loss = step_metric["policy/loss"]
                grad_norm = step_metric["policy/grad_norm"]
                ppo_ratio = step_metric["policy/ppo_ratio"]

                assert -0.01 < loss < 0.01, f"policy/loss out of range: {loss}"
                assert 0 <= grad_norm < 3.0, f"policy/grad_norm out of range: {grad_norm}"
                assert 0.99 < ppo_ratio < 1.01, f"policy/ppo_ratio out of range: {ppo_ratio}"


class LoadAwareRoutingUnitTest(unittest.IsolatedAsyncioTestCase):
    """Unit tests for load-aware routing logic in SamplerClient.

    Mocks get_all_loads and RPC internals — no GPU, no sglang, no Ray.
    """
    def _make_client(self, num_clusters: int, default_ep_idx: int):
        """Build a minimal SamplerClient mock with controllable routing."""
        from unittest.mock import AsyncMock, MagicMock

        client = object.__new__(SamplerClient)
        client.dp_rank = 0
        client.dp_size = 1
        client.svr_cluster_num_per_sampler = [num_clusters]
        client._standalone = True

        # Minimal config mock so _get_load_aware_dispatch_delay can read the
        # fields it accesses:
        #   - training.load_aware_sampler_dispatch_stagger_s  (stagger delay)
        #   - training.rollout_gbs  (used to compute stagger_slots)
        # MagicMock is intentional here: any unset attribute silently returns
        # another MagicMock rather than raising, which keeps the unit test
        # self-contained and focused on routing logic only.
        mock_training = MagicMock()
        mock_training.load_aware_sampler_dispatch_stagger_s = 0.0
        mock_training.rollout_gbs = 1
        mock_config = MagicMock()
        mock_config.training = mock_training
        client.config = mock_config

        rpc = MagicMock()
        rpc._pick_endpoint_idx = MagicMock(return_value=default_ep_idx)
        rpc.get_target_endpoint = MagicMock(side_effect=lambda sample_idx, ep_idx: ep_idx)

        async def fake_call(target_ep, action, data):
            return {"ok": True}

        rpc.call = AsyncMock(side_effect=fake_call)
        client.rpc_client_lst = [rpc]

        client._chosen_ep = None
        orig_get_target = rpc.get_target_endpoint

        def capture_ep(sample_idx, ep_idx):
            client._chosen_ep = ep_idx
            return orig_get_target(sample_idx=sample_idx, ep_idx=ep_idx)

        rpc.get_target_endpoint = MagicMock(side_effect=capture_ep)
        return client

    async def _run_generate(self, client, loads, sidx=0):
        from unittest.mock import AsyncMock

        client.get_all_loads = AsyncMock(return_value=loads)
        await client.generate(
            sampler_idx=0,
            ppo_step=0,
            sidx=sidx,
            batched_data={"x": [1]},
            repeat_n=1,
            load_aware=True,
            busy_sleep_s=0.0,
        )
        return client._chosen_ep

    async def test_default_target_idle_no_switch(self):
        """Default target has no waiting → use it as-is."""
        client = self._make_client(num_clusters=4, default_ep_idx=1)
        loads = [
            {
                "num_reqs": 5,
                "num_tokens": 100,
                "num_waiting_reqs": 0
            },
            {
                "num_reqs": 3,
                "num_tokens": 50,
                "num_waiting_reqs": 0
            },
            {
                "num_reqs": 8,
                "num_tokens": 200,
                "num_waiting_reqs": 0
            },
            {
                "num_reqs": 2,
                "num_tokens": 30,
                "num_waiting_reqs": 0
            },
        ]
        chosen = await self._run_generate(client, loads)
        assert chosen == 1, f"expected default ep 1, got {chosen}"

    async def test_default_busy_switch_to_idle(self):
        """Default target waiting > 0, cluster 3 is idle → switch to 3."""
        client = self._make_client(num_clusters=4, default_ep_idx=0)
        loads = [
            {
                "num_reqs": 10,
                "num_tokens": 500,
                "num_waiting_reqs": 5
            },
            {
                "num_reqs": 8,
                "num_tokens": 400,
                "num_waiting_reqs": 3
            },
            {
                "num_reqs": 6,
                "num_tokens": 300,
                "num_waiting_reqs": 1
            },
            {
                "num_reqs": 2,
                "num_tokens": 50,
                "num_waiting_reqs": 0
            },
        ]
        chosen = await self._run_generate(client, loads)
        assert chosen == 3, f"expected idle cluster 3, got {chosen}"

    async def test_default_busy_multiple_idle_pick_least_reqs(self):
        """Default busy, clusters 1 and 3 both idle → pick the one with fewer num_reqs."""
        client = self._make_client(num_clusters=4, default_ep_idx=0)
        loads = [
            {
                "num_reqs": 10,
                "num_tokens": 500,
                "num_waiting_reqs": 2
            },
            {
                "num_reqs": 5,
                "num_tokens": 100,
                "num_waiting_reqs": 0
            },
            {
                "num_reqs": 8,
                "num_tokens": 300,
                "num_waiting_reqs": 1
            },
            {
                "num_reqs": 2,
                "num_tokens": 30,
                "num_waiting_reqs": 0
            },
        ]
        chosen = await self._run_generate(client, loads)
        assert chosen == 3, f"expected cluster 3 (fewest reqs among idle), got {chosen}"

    async def test_all_busy_pick_least_waiting_then_reqs(self):
        """All clusters waiting > 0 → pick (waiting, num_reqs) minimum."""
        client = self._make_client(num_clusters=4, default_ep_idx=0)
        loads = [
            {
                "num_reqs": 10,
                "num_tokens": 500,
                "num_waiting_reqs": 5
            },
            {
                "num_reqs": 8,
                "num_tokens": 400,
                "num_waiting_reqs": 2
            },
            {
                "num_reqs": 6,
                "num_tokens": 300,
                "num_waiting_reqs": 2
            },
            {
                "num_reqs": 9,
                "num_tokens": 450,
                "num_waiting_reqs": 3
            },
        ]
        chosen = await self._run_generate(client, loads)
        # waiting: [5, 2, 2, 3] → min waiting is 2 (clusters 1, 2)
        # tie-break by num_reqs: cluster 2 has 6 < cluster 1 has 8
        assert chosen == 2, f"expected cluster 2 (least waiting+reqs), got {chosen}"

    async def test_load_aware_false_uses_default(self):
        """load_aware=False → always use default endpoint, no load query."""
        from unittest.mock import AsyncMock

        client = self._make_client(num_clusters=4, default_ep_idx=2)
        client.get_all_loads = AsyncMock()
        await client.generate(
            sampler_idx=0,
            ppo_step=0,
            sidx=0,
            batched_data={"x": [1]},
            repeat_n=1,
            load_aware=False,
        )
        assert client._chosen_ep == 2, f"expected default ep 2, got {client._chosen_ep}"
        client.get_all_loads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
