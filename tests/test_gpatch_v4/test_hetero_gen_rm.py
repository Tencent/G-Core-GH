import asyncio
import importlib.util
import time
import unittest
from pathlib import Path

import torch

from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.infer_engine_config import InferEngineConfig
from gpatch_v4.configs.reward_config import GenRewardConfig, RewardModelInfo
from gpatch_v4.orches.placement_group import (
    compute_gen_rm_config_placement,
    compute_gen_rm_placement,
)
from gpatch_v4.trainer.helper import set_nnodes_default
from gpatch_v4_test_helper import requires_sglang

_GB = 2**30

try:
    from tests.test_gpatch_v4.custom_py.stub_gen_rm_actor import StubGrpoGenRmActor
except ModuleNotFoundError:
    _stub_actor_path = Path(__file__).resolve().parent / "custom_py" / "stub_gen_rm_actor.py"
    _spec = importlib.util.spec_from_file_location("stub_gen_rm_actor", _stub_actor_path)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_module)
    StubGrpoGenRmActor = _module.StubGrpoGenRmActor


def _cluster_max_gpu_mem():
    """Return max GPU memory used across live Ray nodes."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    def gpu_max() -> int:
        import pynvml
        pynvml.nvmlInit()
        try:
            n = pynvml.nvmlDeviceGetCount()
            if n == 0:
                return 0
            return max(
                pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used
                for i in range(n)
            )
        finally:
            pynvml.nvmlShutdown()

    task = ray.remote(num_cpus=0)(gpu_max)
    refs = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node["NodeID"]
        addr = node.get("NodeManagerAddress", node_id)
        ref = task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote()
        refs.append((addr, ref))
    per_node = {addr: ray.get(ref) for addr, ref in refs}
    return (max(per_node.values()) if per_node else 0), per_node


class TestComputeGenRmPlacement(unittest.TestCase):
    def test_single_rm(self):
        alloc = compute_gen_rm_placement(8, [(0, 4)])
        self.assertEqual(alloc[0], 8)

    def test_round_robin_priority(self):
        # total=16, A(MP=4), B(MP=2), C(MP=1)
        # Phase1: A=4, B=2, C=1 (rem=9)
        # Round1: A+4=8(rem=5), B+2=4(rem=3), C+1=2(rem=2)
        # Round2: A skip(4>2), B+2=6(rem=0)
        alloc = compute_gen_rm_placement(16, [(0, 4), (1, 2), (2, 1)])
        self.assertEqual(alloc[0], 8)
        self.assertEqual(alloc[1], 6)
        self.assertEqual(alloc[2], 2)

    def test_exact_fit(self):
        # total=6, A(MP=2), B(MP=2) -> Phase1: A=2,B=2 (rem=2)
        # Round1: A+2=4(rem=0)
        alloc = compute_gen_rm_placement(6, [(0, 2), (1, 2)])
        self.assertEqual(alloc[0], 4)
        self.assertEqual(alloc[1], 2)

    def test_equal_mp_sizes(self):
        # total=12, A(MP=4), B(MP=4) -> Phase1: A=4,B=4 (rem=4)
        # Round1: A+4=8(rem=0)
        alloc = compute_gen_rm_placement(12, [(0, 4), (1, 4)])
        self.assertEqual(alloc[0], 8)
        self.assertEqual(alloc[1], 4)

    def test_equal_mp_more_gpus(self):
        # total=20, A(MP=4), B(MP=4) -> Phase1: A=4,B=4 (rem=12)
        # Round1: A+4=8(rem=8), B+4=8(rem=4)
        # Round2: A+4=12(rem=0)
        alloc = compute_gen_rm_placement(20, [(0, 4), (1, 4)])
        self.assertEqual(alloc[0], 12)
        self.assertEqual(alloc[1], 8)

    def test_all_gpus_used(self):
        alloc = compute_gen_rm_placement(8, [(0, 2), (1, 2)])
        total = sum(alloc.values())
        self.assertEqual(total, 8)

    def test_insufficient_gpus_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(3, [(0, 4)])

    def test_insufficient_gpus_multi_rm_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(5, [(0, 4), (1, 4)])

    def test_no_remainder(self):
        # total=8, A(TP=4), B(TP=4) -> each exactly 1 replica
        alloc = compute_gen_rm_placement(8, [(0, 4), (1, 4)])
        self.assertEqual(alloc[0], 4)
        self.assertEqual(alloc[1], 4)

    def test_manual_allocation_is_exact(self):
        alloc = compute_gen_rm_placement(20, [(0, 4), (1, 4)], {1: 8})
        self.assertEqual(alloc[0], 12)
        self.assertEqual(alloc[1], 8)

    def test_manual_allocation_mixed_with_auto(self):
        alloc = compute_gen_rm_placement(
            32,
            [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
            {6: 8},
        )
        for rm_idx in range(6):
            self.assertEqual(alloc[rm_idx], 4)
        self.assertEqual(alloc[6], 8)
        self.assertEqual(sum(alloc.values()), 32)

    def test_manual_allocation_less_than_mp_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(8, [(0, 4), (1, 4)], {0: 2})

    def test_manual_allocation_not_divisible_by_mp_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(12, [(0, 4), (1, 4)], {0: 6})

    def test_manual_allocation_non_positive_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(8, [(0, 4), (1, 4)], {0: 0})

    def test_manual_allocation_non_integer_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(8, [(0, 4), (1, 4)], {0: 4.0})

    def test_manual_allocation_unknown_rm_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(8, [(0, 4), (1, 4)], {2: 4})

    def test_manual_allocation_total_exceeds_budget_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(8, [(0, 4), (1, 4)], {0: 12})

    def test_manual_allocation_leaves_insufficient_auto_budget_raises(self):
        with self.assertRaises(AssertionError):
            compute_gen_rm_placement(8, [(0, 4), (1, 4), (2, 4)], {0: 4})

    def test_config_placement_extracts_manual_allocations(self):
        config = GenRewardConfig(
            reward_model_info=[RewardModelInfo(), RewardModelInfo()],
            infer_engine_configs=[
                InferEngineConfig(
                    dist_config=DistConfig(
                        tensor_model_parallel_size=2,
                        nnodes=1,
                        num_gpus_per_node=4,
                    )
                ),
                InferEngineConfig(
                    dist_config=DistConfig(
                        tensor_model_parallel_size=1,
                        nnodes=1,
                        num_gpus_per_node=4,
                    ),
                    allocated_gpus=2,
                ),
            ],
        )
        alloc = compute_gen_rm_config_placement(config, 4)
        self.assertEqual(alloc[0], 2)
        self.assertEqual(alloc[1], 2)

    def test_config_placement_requires_one_to_one_lists(self):
        config = GenRewardConfig(
            reward_model_info=[RewardModelInfo()],
            infer_engine_configs=[
                InferEngineConfig(),
                InferEngineConfig(),
            ],
        )
        with self.assertRaises(AssertionError):
            compute_gen_rm_config_placement(config, 8)


class TestGenRewardConfigBasic(unittest.TestCase):
    def test_input_token_key_none_by_default(self):
        config = GenRewardConfig(
            reward_model_info=[RewardModelInfo()],
            infer_engine_configs=[InferEngineConfig()],
        )
        self.assertIsNone(config.reward_model_info[0].input_token_key)

    def test_input_token_key_set(self):
        rm = RewardModelInfo(input_token_key=["math_tokens", "code_tokens"])
        self.assertEqual(rm.input_token_key, ["math_tokens", "code_tokens"])

    def test_none_reward_model_info(self):
        config = GenRewardConfig()
        self.assertIsNone(config.reward_model_info)

    def test_one_to_one_config(self):
        """Verify reward_model_info and infer_engine_configs can be 1:1."""
        config = GenRewardConfig(
            reward_model_info=[RewardModelInfo(), RewardModelInfo()],
            infer_engine_configs=[
                InferEngineConfig(dist_config=DistConfig(nnodes=1, num_gpus_per_node=4)),
                InferEngineConfig(dist_config=DistConfig(nnodes=1, num_gpus_per_node=2)),
            ],
        )
        self.assertEqual(len(config.reward_model_info), 2)
        self.assertEqual(len(config.infer_engine_configs), 2)


@requires_sglang
class TestHeteroGenRmEndToEnd(unittest.IsolatedAsyncioTestCase):
    """End-to-end test for heterogeneous gen-rm with independent placement.

    Requires: 4 GPUs (colocate mode, policy + gen-rm share GPUs), Ray, sglang.

    Config (test_hetero_gen_rm_e2e.yaml):
      - RM0: Qwen2.5-3B, TP=2, input_token_key=["math_tokens"]
      - RM1: Qwen2.5-3B, TP=1, input_token_key=["code_tokens"]
      - 2 infer_engine_configs (one-to-one with RMs)
    """
    def setUp(self):
        # Use orches.init(config) instead of raw ray.init() so that PYTHONPATH
        # and other env vars are propagated to Ray workers via runtime_env.
        # Without this, workers fail with "No module named 'megatron'"
        # because the import chain (base_actor → utils → core → …)
        # reaches megatron which is only on PYTHONPATH, not installed.
        from gpatch_v4 import orches
        orches.init()

    def tearDown(self):
        from gpatch_v4 import orches
        orches.shutdown()

    async def test_placement_group_creation(self):
        """Verify gen-rm PG creation."""
        from gpatch_v4.configs.config import RlConfig
        from gpatch_v4.orches.placement_group import create_placement_groups
        from gpatch_v4_test_helper import load_config

        config = load_config('test_hetero_gen_rm_e2e', RlConfig)
        set_nnodes_default(config, nnodes=1)

        self.assertEqual(len(config.gen_rm.reward_model_info), 2)
        self.assertEqual(config.gen_rm.reward_model_info[0].input_token_key, ["math_tokens"])
        self.assertEqual(config.gen_rm.reward_model_info[1].input_token_key, ["code_tokens"])

        pgs = create_placement_groups(config)

        gen_rm_pg = pgs["gen_rm"]
        self.assertIsInstance(gen_rm_pg, tuple)
        self.assertEqual(len(gen_rm_pg), 2)  # (pg, bundle_indices)

    async def test_group_creation(self):
        """Verify create_gen_rm_group returns one RayGenRmGroup per RM."""
        from gpatch_v4.configs.config import RlConfig
        from gpatch_v4.orches.gen_rm_group import RayGenRmGroup
        from gpatch_v4.orches.placement_group import (
            create_gen_rm_group,
            create_placement_groups,
        )
        from gpatch_v4_test_helper import load_config

        config = load_config('test_hetero_gen_rm_e2e', RlConfig)
        set_nnodes_default(config, nnodes=1)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)

        self.assertIsInstance(gen_rm_groups, list)
        num_rms = len(config.gen_rm.reward_model_info)
        self.assertEqual(len(gen_rm_groups), num_rms)

        for grp in gen_rm_groups:
            self.assertIsInstance(grp, RayGenRmGroup)

    async def test_per_rm_gpu_allocation(self):
        """Verify each group gets correct GPU count and non-overlapping actors.

        Config: 4 GPU pool, RM0 (TP=2), RM1 (TP=1).
        Expected: RM0 gets 2 GPUs (1 engine), RM1 gets 2 GPUs (2 engines).
        """
        import ray

        from gpatch_v4.configs.config import RlConfig
        from gpatch_v4.orches.custom_actor_registry import CustomActorRegistry
        from gpatch_v4.orches.gen_rm_group import RayGenRmGroup
        from gpatch_v4.orches.placement_group import (
            create_gen_rm_group,
            create_placement_groups,
        )
        from gpatch_v4_test_helper import load_config

        if CustomActorRegistry.get("stub_gen_rm") is None:
            CustomActorRegistry.register("stub_gen_rm", StubGrpoGenRmActor)

        config = load_config('test_hetero_gen_rm_e2e', RlConfig)
        set_nnodes_default(config, nnodes=1)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)

        self.assertEqual(len(gen_rm_groups), 2)

        grp0 = gen_rm_groups[0]  # RM0, TP=2
        grp1 = gen_rm_groups[1]  # RM1, TP=1

        self.assertEqual(grp0.rm_idx, 0)
        self.assertEqual(grp1.rm_idx, 1)
        self.assertEqual(grp0.allocated_num_gpus, 2)
        self.assertEqual(grp1.allocated_num_gpus, 2)

        # RM0 (TP=2): 2 GPUs / 2 per engine = 1 engine
        _, num_engines_0, _, _, _ = grp0.get_rm_engine_info()
        self.assertEqual(num_engines_0, 1)
        self.assertEqual(len(grp0._actor_handlers), 1)

        # RM1 (TP=1): 2 GPUs / 1 per engine = 2 engines
        _, num_engines_1, _, _, _ = grp1.get_rm_engine_info()
        self.assertEqual(num_engines_1, 2)
        self.assertEqual(len(grp1._actor_handlers), 2)

        # Verify actor names don't overlap
        actor_names_0 = set()
        for rank in range(num_engines_0):
            name = f"gen_rm_0_{rank}"
            actor = ray.get_actor(name)
            self.assertIsNotNone(actor)
            actor_names_0.add(name)

        actor_names_1 = set()
        for rank in range(num_engines_1):
            name = f"gen_rm_1_{rank}"
            actor = ray.get_actor(name)
            self.assertIsNotNone(actor)
            actor_names_1.add(name)

        self.assertTrue(actor_names_0.isdisjoint(actor_names_1))

    async def test_ppo_step_gen_rm_flow(self):
        """Integration test: one complete PPO step gen-rm reward flow.

        Exercises the full server-side lifecycle:
        1. Placement group creation
        2. Actor creation with correct Ray names (gen_rm_{rm_idx}_{rank})
        3. Group init (init → init_infer_engine → sleep)
        4. PPO step: mark_ppo_step_begin → broadcast all batches to all RMs
           → generate_rewards → mark_ppo_step_end
        5. Verify reward keys/shapes and broadcast correctness
        """
        import ray

        from gpatch_v4.configs.config import RlConfig
        from gpatch_v4.orches.custom_actor_registry import CustomActorRegistry
        from gpatch_v4.orches.placement_group import (
            create_gen_rm_group,
            create_placement_groups,
        )
        from gpatch_v4_test_helper import load_config

        if CustomActorRegistry.get("stub_gen_rm") is None:
            CustomActorRegistry.register("stub_gen_rm", StubGrpoGenRmActor)

        config = load_config('test_hetero_gen_rm_e2e', RlConfig)
        set_nnodes_default(config, nnodes=1)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)

        self.assertEqual(len(gen_rm_groups), 2)
        for grp in gen_rm_groups:
            await grp.init()

        # -- Verify actors are discoverable via Ray naming convention --
        actor_0_0 = ray.get_actor("gen_rm_0_0")
        actor_1_0 = ray.get_actor("gen_rm_1_0")
        self.assertIsNotNone(actor_0_0)
        self.assertIsNotNone(actor_1_0)

        # -- Prepare rollout batches (simulating sampler output) --
        n_samples = config.training.rollout_mbs * config.training.sampling_repeat_n
        rbs = [
            {
                "prompt": ["Solve: 2+2"] * n_samples,
                "math_tokens": [[101, 102]] * n_samples,
            },
            {
                "prompt": ["Write hello world"] * n_samples,
                "code_tokens": [[201, 202]] * n_samples,
            },
        ]

        # -- PPO Step: gen-rm reward generation --
        num_rms = len(config.gen_rm.reward_model_info)
        ppo_step = 0

        for rm_idx in range(num_rms):
            grp = gen_rm_groups[rm_idx]
            _, num_engines, _, _, _ = grp.get_rm_engine_info()

            # Step 1: mark_ppo_step_begin (wake up all engines)
            futs = []
            for ep_idx in range(num_engines):
                actor = grp._actor_handlers[ep_idx]
                futs.append(actor.wake_up.remote({"tag_names": ["weights", "kv_cache"]}))
            for fut in futs:
                await fut

            # Step 2: Send all batches to every RM (broadcast mode)
            for idx, rb in enumerate(rbs):
                master_actor = grp._actor_handlers[0]
                ret = await master_actor.generate_rewards.remote({"batched_data": rb})
                reward_key = f"reward_gen_rm_{rm_idx}"
                self.assertIn(reward_key, ret, f"RM{rm_idx} must return {reward_key}")
                rbs[idx][reward_key] = ret[reward_key]

            # Step 3: mark_ppo_step_end (sleep all engines)
            futs = []
            for ep_idx in range(num_engines):
                actor = grp._actor_handlers[ep_idx]
                futs.append(actor.sleep.remote({"tag_names": ["weights", "kv_cache"]}))
            for fut in futs:
                await fut

        # -- Verify broadcast: all batches get rewards from all RMs --
        for rbi in range(len(rbs)):
            for rm_idx in range(num_rms):
                reward_key = f"reward_gen_rm_{rm_idx}"
                self.assertIn(reward_key, rbs[rbi])
                self.assertEqual(len(rbs[rbi][reward_key]), n_samples)
                for t in rbs[rbi][reward_key]:
                    self.assertIsInstance(t, torch.Tensor)

        # -- Verify reward values (deterministic stub) --
        # RM0: 1.0 / (0+1) = 1.0, RM1: 1.0 / (1+1) = 0.5
        self.assertAlmostEqual(rbs[0]["reward_gen_rm_0"][0].item(), 1.0)
        self.assertAlmostEqual(rbs[0]["reward_gen_rm_1"][0].item(), 0.5)
        self.assertAlmostEqual(rbs[1]["reward_gen_rm_0"][0].item(), 1.0)
        self.assertAlmostEqual(rbs[1]["reward_gen_rm_1"][0].item(), 0.5)

    async def test_lazy_gen_rm_engine_lifecycle(self):
        """Verify lazy gen-rm config starts engines per phase and stops them after."""
        from gpatch_v4.configs.config import RlConfig
        from gpatch_v4.orches.custom_actor_registry import CustomActorRegistry
        from gpatch_v4.orches.placement_group import (
            create_gen_rm_group,
            create_placement_groups,
        )
        from gpatch_v4_test_helper import load_config

        if CustomActorRegistry.get("stub_gen_rm") is None:
            CustomActorRegistry.register("stub_gen_rm", StubGrpoGenRmActor)

        config = load_config('test_hetero_gen_rm_e2e', RlConfig)
        config.gen_rm.destroy_engine_after_generation = True
        set_nnodes_default(config, nnodes=1)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)

        for grp in gen_rm_groups:
            await grp.init_setup()
        for grp in gen_rm_groups:
            await grp.init_load()

        all_actors = []
        for grp in gen_rm_groups:
            _, num_engines, _, _, _ = grp.get_rm_engine_info()
            all_actors.extend(grp._actor_handlers[:num_engines])

        for actor in all_actors:
            state = await actor.get_lifecycle_state.remote()
            self.assertTrue(state["configured"])
            self.assertFalse(state["engine_started"])
            self.assertEqual(state["start_count"], 0)
            self.assertEqual(state["stop_count"], 0)

        async def run_phase(ppo_step):
            for actor in all_actors:
                await actor.mark_ppo_step_begin.remote(
                    {"ppo_step": ppo_step, "tag_names": ["weights", "kv_cache"]}
                )
            for actor in all_actors:
                state = await actor.get_lifecycle_state.remote()
                self.assertTrue(state["engine_started"])

            n_samples = config.training.rollout_mbs * config.training.sampling_repeat_n
            for rm_idx, grp in enumerate(gen_rm_groups):
                ret = await grp._actor_handlers[0].generate_rewards.remote(
                    {"batched_data": {"prompt": [f"batch-{ppo_step}"] * n_samples}}
                )
                self.assertIn(f"reward_gen_rm_{rm_idx}", ret)

            for actor in all_actors:
                await actor.mark_ppo_step_end.remote({"ppo_step": ppo_step})
            for actor in all_actors:
                state = await actor.get_lifecycle_state.remote()
                self.assertFalse(state["engine_started"])

        await run_phase(0)
        for actor in all_actors:
            state = await actor.get_lifecycle_state.remote()
            self.assertEqual(state["start_count"], 1)
            self.assertEqual(state["stop_count"], 1)

        await run_phase(1)
        for actor in all_actors:
            state = await actor.get_lifecycle_state.remote()
            self.assertEqual(state["start_count"], 2)
            self.assertEqual(state["stop_count"], 2)

    async def test_lazy_gen_rm_destroy_releases_gpu_memory(self):
        """Real sglang gen-RM should return close to baseline after lazy destroy."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for GPU memory release test")

        from gpatch_v4.configs.config import RlConfig
        from gpatch_v4.orches.placement_group import (
            create_gen_rm_group,
            create_placement_groups,
        )
        from gpatch_v4_test_helper import load_config

        config = load_config('test_hetero_gen_rm_e2e', RlConfig)
        config.gen_rm.destroy_engine_after_generation = True
        config.gen_rm.reward_model_info = [config.gen_rm.reward_model_info[0]]
        config.gen_rm.infer_engine_configs = [config.gen_rm.infer_engine_configs[0]]
        config.gen_rm.reward_model_info[0].reward_py_path = "tasks/math_rl_v4/gen_rm_reward.py"
        config.gen_rm.reward_model_info[0].gen_reward_fn_name = "generate_rewards"
        config.gen_rm.infer_engine_configs[0].custom_rm_actor_impl = None
        set_nnodes_default(config, nnodes=1)

        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)
        self.assertEqual(len(gen_rm_groups), 1)
        grp = gen_rm_groups[0]

        await grp.init_setup()
        await grp.init_load()

        baseline_mem, baseline_per_node = _cluster_max_gpu_mem()
        _, num_engines, _, _, _ = grp.get_rm_engine_info()
        actors = grp._actor_handlers[:num_engines]

        await asyncio.gather(
            *[
                actor.mark_ppo_step_begin.remote(
                    {"ppo_step": 0, "tag_names": ["weights", "kv_cache"]}
                ) for actor in actors
            ]
        )
        after_begin_mem, after_begin_per_node = _cluster_max_gpu_mem()
        self.assertGreater(
            after_begin_mem,
            baseline_mem + 1 * _GB,
            f"expected gen-rm lazy begin to load >1GiB, "
            f"baseline={baseline_mem / _GB:.2f}GiB {baseline_per_node=}, "
            f"after_begin={after_begin_mem / _GB:.2f}GiB {after_begin_per_node=}",
        )

        await asyncio.gather(
            *[actor.mark_ppo_step_end.remote({"ppo_step": 0}) for actor in actors]
        )

        deadline = time.time() + 30.0
        final_mem = after_begin_mem
        final_per_node = after_begin_per_node
        threshold = baseline_mem + int(0.5 * _GB)
        while time.time() < deadline:
            final_mem, final_per_node = _cluster_max_gpu_mem()
            if final_mem <= threshold:
                break
            time.sleep(0.5)

        self.assertLessEqual(
            final_mem,
            threshold,
            f"gen-rm GPU memory did not return within 0.5GiB of baseline: "
            f"baseline={baseline_mem / _GB:.2f}GiB {baseline_per_node=}, "
            f"final={final_mem / _GB:.2f}GiB {final_per_node=}",
        )


class TestGenerateGenRmRewardParallel(unittest.IsolatedAsyncioTestCase):
    """Unit test for the parallel 3-phase gen-rm reward flow in BaseRolloutGenerator.

    Mocks all external dependencies (gen_rm_client, config, cpu_barrier, etc.)
    to verify:
    1. mark_ppo_step_begin is called for ALL RMs before any reward generation
    2. Reward requests are dispatched concurrently (via asyncio.gather)
    3. All batches are broadcast to every RM
    4. mark_ppo_step_end is called for ALL RMs after all rewards collected
    5. Results are correctly merged back into rbs
    """
    async def test_parallel_reward_flow_broadcast_multi_rm(self):
        """Two RMs, verify all batches are broadcast to every RM."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        # -- Build mock reward_model_info --
        rm_info_0 = MagicMock()
        rm_info_0.input_token_key = ["math_tokens"]
        rm_info_1 = MagicMock()
        rm_info_1.input_token_key = ["code_tokens"]

        # -- Build mock config --
        config = MagicMock()
        config.placement_type = "colocate"
        config.gen_rm.reward_model_info = [rm_info_0, rm_info_1]

        # -- Build mock gen_rm_client --
        gen_rm_client = MagicMock()
        gen_rm_client.num_rms = 2
        gen_rm_client.mark_ppo_step_begin = AsyncMock()
        gen_rm_client.mark_ppo_step_end = AsyncMock()

        async def mock_generate_rewards(rm_idx, ppo_step, sample_idx, batched_data):
            reward_key = f"reward_gen_rm_{rm_idx}"
            return {reward_key: [torch.tensor(1.0 / (rm_idx + 1))]}

        gen_rm_client.generate_rewards = AsyncMock(side_effect=mock_generate_rewards)

        # -- Build rollout batches: batch0 has math_tokens, batch1 has code_tokens --
        rbs = [
            {
                "prompt": ["Solve 2+2"],
                "math_tokens": [[101, 102]]
            },
            {
                "prompt": ["Write hello"],
                "code_tokens": [[201, 202]]
            },
        ]

        # -- Construct generator with mocks --
        from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator

        generator = object.__new__(BaseRolloutGenerator)
        generator.config = config
        generator.gen_rm_client = gen_rm_client
        generator.is_mp_and_cp_head = True
        generator.sample_idx = 0

        with patch("gpatch_v4.rollout_generator.base_generator.cpu_barrier"), \
             patch("gpatch_v4.rollout_generator.base_generator.logging_memory_usage"), \
             patch("gpatch_v4.rollout_generator.base_generator.check_rollout_batches", return_value=True):
            result = await generator.generate_gen_rm_reward(
                rbs, num_microbatches=2, curr_ppo_step=0
            )

        # -- Verify Phase 1: both RMs got mark_ppo_step_begin --
        self.assertEqual(gen_rm_client.mark_ppo_step_begin.call_count, 2)

        # -- Verify Phase 2: broadcast — all batches get rewards from all RMs --
        for i in range(2):
            self.assertIn("reward_gen_rm_0", result[i])
            self.assertIn("reward_gen_rm_1", result[i])

        # generate_rewards called 4 times (2 batches × 2 RMs)
        self.assertEqual(gen_rm_client.generate_rewards.call_count, 4)

        # -- Verify Phase 3: both RMs got mark_ppo_step_end --
        self.assertEqual(gen_rm_client.mark_ppo_step_end.call_count, 2)

        # -- Verify reward values --
        self.assertAlmostEqual(result[0]["reward_gen_rm_0"][0].item(), 1.0)
        self.assertAlmostEqual(result[0]["reward_gen_rm_1"][0].item(), 0.5)
        self.assertAlmostEqual(result[1]["reward_gen_rm_0"][0].item(), 1.0)
        self.assertAlmostEqual(result[1]["reward_gen_rm_1"][0].item(), 0.5)

    async def test_parallel_reward_flow_broadcast(self):
        """Single RM with input_token_key=None (broadcast), verify all batches are sent."""
        from unittest.mock import AsyncMock, MagicMock, patch

        rm_info_0 = MagicMock()
        rm_info_0.input_token_key = None

        config = MagicMock()
        config.placement_type = "colocate"
        config.gen_rm.reward_model_info = [rm_info_0]

        gen_rm_client = MagicMock()
        gen_rm_client.num_rms = 1
        gen_rm_client.mark_ppo_step_begin = AsyncMock()
        gen_rm_client.mark_ppo_step_end = AsyncMock()

        async def mock_generate_rewards(rm_idx, ppo_step, sample_idx, batched_data):
            return {f"reward_gen_rm_{rm_idx}": [torch.tensor(0.42)]}

        gen_rm_client.generate_rewards = AsyncMock(side_effect=mock_generate_rewards)

        rbs = [
            {
                "prompt": ["batch0"]
            },
            {
                "prompt": ["batch1"]
            },
            {
                "prompt": ["batch2"]
            },
        ]

        from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator

        generator = object.__new__(BaseRolloutGenerator)
        generator.config = config
        generator.gen_rm_client = gen_rm_client
        generator.is_mp_and_cp_head = True
        generator.sample_idx = 0

        with patch("gpatch_v4.rollout_generator.base_generator.cpu_barrier"), \
             patch("gpatch_v4.rollout_generator.base_generator.logging_memory_usage"), \
             patch("gpatch_v4.rollout_generator.base_generator.check_rollout_batches", return_value=True):
            result = await generator.generate_gen_rm_reward(
                rbs, num_microbatches=3, curr_ppo_step=0
            )

        # All 3 batches should get reward from RM0
        for i in range(3):
            self.assertIn("reward_gen_rm_0", result[i])

        # generate_rewards called 3 times (one per batch)
        self.assertEqual(gen_rm_client.generate_rewards.call_count, 3)

    async def test_reward_error_still_marks_ppo_step_end(self):
        """Generation errors should still close the gen-rm phase."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config = MagicMock()
        config.placement_type = "colocate"
        config.gen_rm.reward_model_info = [MagicMock(), MagicMock()]

        gen_rm_client = MagicMock()
        gen_rm_client.num_rms = 2
        gen_rm_client.mark_ppo_step_begin = AsyncMock()
        gen_rm_client.mark_ppo_step_end = AsyncMock()
        gen_rm_client.generate_rewards = AsyncMock(side_effect=RuntimeError("gen-rm failed"))

        from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator

        generator = object.__new__(BaseRolloutGenerator)
        generator.config = config
        generator.gen_rm_client = gen_rm_client
        generator.is_mp_and_cp_head = True
        generator.sample_idx = 0

        with patch("gpatch_v4.rollout_generator.base_generator.cpu_barrier"), \
             patch("gpatch_v4.rollout_generator.base_generator.logging_memory_usage"), \
             patch("gpatch_v4.rollout_generator.base_generator.check_rollout_batches", return_value=True):
            with self.assertRaises(RuntimeError):
                await generator.generate_gen_rm_reward(
                    [{"prompt": ["batch0"]}], num_microbatches=1, curr_ppo_step=7
                )

        self.assertEqual(gen_rm_client.mark_ppo_step_begin.call_count, 2)
        self.assertEqual(gen_rm_client.mark_ppo_step_end.call_count, 2)

    async def test_non_mp_head_skips_reward_dispatch(self):
        """When is_mp_and_cp_head=False, no reward requests should be sent."""
        from unittest.mock import AsyncMock, MagicMock, patch

        rm_info_0 = MagicMock()
        rm_info_0.input_token_key = None

        config = MagicMock()
        config.placement_type = "disaggregated"
        config.gen_rm.reward_model_info = [rm_info_0]

        gen_rm_client = MagicMock()
        gen_rm_client.num_rms = 1
        gen_rm_client.mark_ppo_step_begin = AsyncMock()
        gen_rm_client.mark_ppo_step_end = AsyncMock()
        gen_rm_client.generate_rewards = AsyncMock()

        rbs = [{"prompt": ["batch0"]}]

        from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator

        generator = object.__new__(BaseRolloutGenerator)
        generator.config = config
        generator.gen_rm_client = gen_rm_client
        generator.is_mp_and_cp_head = False
        generator.sample_idx = 0

        with patch("gpatch_v4.rollout_generator.base_generator.cpu_barrier"), \
             patch("gpatch_v4.rollout_generator.base_generator.logging_memory_usage"), \
             patch("gpatch_v4.rollout_generator.base_generator.check_rollout_batches", return_value=True):
            result = await generator.generate_gen_rm_reward(
                rbs, num_microbatches=1, curr_ppo_step=0
            )

        # mark_ppo_step_begin/end should still be called (they handle barrier logic)
        self.assertEqual(gen_rm_client.mark_ppo_step_begin.call_count, 1)
        self.assertEqual(gen_rm_client.mark_ppo_step_end.call_count, 1)
        # But generate_rewards should NOT be called
        gen_rm_client.generate_rewards.assert_not_called()


class TestAsyncGenRmPhase(unittest.IsolatedAsyncioTestCase):
    async def test_gen_rm_all_phase_marks_end_on_error(self):
        """Async colocate phase already uses finally semantics for cleanup."""
        from unittest.mock import AsyncMock, MagicMock

        from gpatch_v4.rollout_generator.async_rollout.mixin import ColocateAgentMixin

        phase_owner = object.__new__(ColocateAgentMixin)
        phase_owner.agents_pg = None
        phase_owner.agent_rank = 0
        phase_owner.gen_rm_client = MagicMock()
        phase_owner.gen_rm_client.num_rms = 2
        phase_owner.gen_rm_client.mark_ppo_step_begin = AsyncMock()
        phase_owner.gen_rm_client.mark_ppo_step_end = AsyncMock()
        phase_owner.agent_cpu_barrier = AsyncMock()
        phase_owner._log_policy_memory = AsyncMock()

        with self.assertRaises(RuntimeError):
            async with phase_owner.gen_rm_all_phase(ppo_step=9):
                raise RuntimeError("phase failed")

        self.assertEqual(phase_owner.gen_rm_client.mark_ppo_step_begin.call_count, 2)
        self.assertEqual(phase_owner.gen_rm_client.mark_ppo_step_end.call_count, 2)


if __name__ == '__main__':
    unittest.main()
