# coding=utf-8
"""DSpark THD CP=1 vs CP>1：loss / grad 数值对照（Ray worker）。

两边共用 ``_dspark_thd_worker``：同 token、同 pack 对齐、同 global
anchor seed；CP=1 捕获 backbone + DSpark TopK 路由，CP>1 按 contiguous
chunk replay。比较 CE / L1 / confidence / total DSpark loss 与 grad norm。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 \\
      tests/test_gfused/test_deepseek_v4_dspark_cp_vs_cp1.py::TestDeepseekV4DSparkThdCp::test_dspark_thd_cp1_vs_cp2
"""

from __future__ import annotations

import math
import os
import unittest

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from test_gfused.test_deepseek_v4_dspark_bshd_vs_thd import (
    EP_SIZE,
    HF_MODEL_PATH,
    NUM_GPUS,
    _dspark_thd_worker,
    _get_node_ip,
)
from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

CP_SIZE = 2
LOSS_RTOL = 0.005
GRAD_RTOL = 0.01


class TestDeepseekV4DSparkThdCp(unittest.TestCase):
    def setUp(self):
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        if not env_pythonpath:
            rcdir = "/work/wepsdl"
            env_pythonpath = ":".join(
                [
                    f"{rcdir}/gcore-dev",
                    f"{rcdir}/gcore-dev/tests",
                    f"{rcdir}/gcore-dev/tests/test_gpatch_v4",
                    f"{rcdir}/Megatron-LM",
                    f"{rcdir}/mbridge",
                    f"{rcdir}/Megatron-Bridge/src",
                ]
            )
        ray.init(
            address="auto",
            runtime_env={"env_vars": {"PYTHONPATH": env_pythonpath}},
        )
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _run_thd_workers(
        self,
        *,
        hf_model_path: str,
        world_size: int,
        ep_size: int,
        master_port: int,
        fake_seq_lens: list[int],
        cp_size: int,
        pack_cp_size: int,
        capture_routing: bool,
        replay_indices: list | None,
    ) -> list[dict]:
        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * world_size,
            strategy="PACK",
        )
        ray.get(pg.ready())
        try:
            master_addr = ray.get(
                _get_node_ip.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,
                    )
                ).remote()
            )
            futures = [
                _dspark_thd_worker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=r,
                    )
                ).remote(
                    hf_model_path,
                    rank=r,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    ep_size=ep_size,
                    fake_seq_lens=fake_seq_lens,
                    cp_size=cp_size,
                    pack_cp_size=pack_cp_size,
                    capture_routing=capture_routing,
                    replay_indices=replay_indices,
                )
                for r in range(world_size)
            ]
            return ray.get(futures)
        finally:
            remove_placement_group(pg)

    def test_dspark_thd_cp1_vs_cp2(
        self,
        world_size: int = NUM_GPUS,
        ep_size: int = EP_SIZE,
        cp_size: int = CP_SIZE,
        master_port: int = 13800,
    ):
        """THD cp=1 vs cp=2：DSpark CE/L1/confidence/total loss 与 grad_norm 应接近。"""
        hf_model_path = os.path.abspath(HF_MODEL_PATH)
        if not os.path.isdir(hf_model_path):
            raise unittest.SkipTest(
                f"DSpark checkpoint not found: {hf_model_path}; "
                "set DSPARK_HF_MODEL_PATH"
            )
        assert world_size % ep_size == 0
        assert world_size % cp_size == 0

        fake_seq_lens = [100, 80, 120]
        pack_cp_size = cp_size

        print("=" * 60)
        print(f"Running THD DSpark ep={ep_size} cp=1 (baseline + routing capture)")
        cp1_results = self._run_thd_workers(
            hf_model_path=hf_model_path,
            world_size=world_size,
            ep_size=ep_size,
            master_port=master_port,
            fake_seq_lens=fake_seq_lens,
            cp_size=1,
            pack_cp_size=pack_cp_size,
            capture_routing=True,
            replay_indices=None,
        )

        replay = cp1_results[0]["recorded_routing"]
        assert replay is not None and len(replay) > 0
        self.tearDown()
        self.setUp()

        print("=" * 60)
        print(
            f"Running THD DSpark ep={ep_size} cp={cp_size} "
            f"(contiguous CP + router replay, {len(replay)} TopK layers)"
        )
        cp_results = self._run_thd_workers(
            hf_model_path=hf_model_path,
            world_size=world_size,
            ep_size=ep_size,
            master_port=master_port + 1,
            fake_seq_lens=fake_seq_lens,
            cp_size=cp_size,
            pack_cp_size=pack_cp_size,
            capture_routing=False,
            replay_indices=replay,
        )

        for res in cp1_results + cp_results:
            self.assertTrue(math.isfinite(res["dspark_loss"]))
            self.assertTrue(math.isfinite(res["ce_loss"]))
            self.assertTrue(math.isfinite(res["l1_loss"]))
            self.assertTrue(math.isfinite(res["confidence_loss"]))
            self.assertTrue(math.isfinite(res["total_grad_norm"]))

        r0_cp1 = cp1_results[0]
        r0_cp = cp_results[0]
        self.assertEqual(r0_cp1["packed_seq_len"], r0_cp["packed_seq_len"])
        self.assertAlmostEqual(r0_cp1["global_denom"], r0_cp["global_denom"], places=5)

        keys = (
            ("dspark_loss", LOSS_RTOL),
            ("ce_loss", LOSS_RTOL),
            ("l1_loss", LOSS_RTOL),
            ("confidence_loss", LOSS_RTOL),
            ("total_grad_norm", GRAD_RTOL),
        )
        print("\n" + "=" * 60)
        print(f"DSpark THD CP=1 vs CP={cp_size}")
        for name, rtol in keys:
            left = r0_cp1[name]
            right = r0_cp[name]
            rel = abs(right - left) / (abs(left) + 1e-8)
            print(f"  {name}: cp1={left:.6f}  cp{cp_size}={right:.6f}  rel={rel:.6e}")
            self.assertLess(rel, rtol, f"{name} rel_diff {rel:.6e} > {rtol}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    unittest.main()
