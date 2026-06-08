"""End-to-end sanity test: actor.shutdown() actually releases GPU memory and
reaps sglang subprocesses, cluster-wide.

Contract tested
---------------

1. Before any actor starts, every node's max GPU memory is < 1 GiB and
   no sglang scheduler / TP worker / detokenizer processes exist on any
   node.
2. After creating a ``GrpoGenRmActor`` group and calling ``init()``, at
   least one GPU (somewhere in the cluster) shows non-trivial memory
   usage AND at least one sglang subprocess is alive somewhere.
3. After each actor's ``actor.shutdown.remote()`` returns (triggering
   sglang's ``kill_process_tree`` on the actor's host), the cluster
   returns to baseline: cluster-max GPU mem < 1 GiB AND zero sglang
   processes on every node.

Queries are dispatched as zero-CPU ray tasks pinned to each live node via
``NodeAffinitySchedulingStrategy(soft=False)``. The probe functions are
defined **inside** ``_run_on_every_node`` as closures so cloudpickle
serializes them by value — remote ray workers do NOT need this test
module on their ``sys.path``.
"""

import time
import unittest

import ray

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4.orches.placement_group import (
    create_gen_rm_group,
    create_placement_groups,
)
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_GB = 2**30


def _run_on_every_node(probe_name: str) -> dict:
    """Dispatch a named probe as a zero-CPU ray task on every live node.

    Probes are defined as closures so cloudpickle ships them by value; ray
    workers do NOT need this test module importable.

    Parameters
    ----------
    probe_name : {"gpu_max", "sglang_pids"}
        Which of the two probe closures to run.

    Returns
    -------
    dict
        ``{addr: result}`` keyed by ``NodeManagerAddress``.
    """
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

    def sglang_pids() -> list:
        import os
        import subprocess
        uid = os.getuid()
        # Include comm (proctitle) because sglang sets proctitle via
        # ``setproctitle("sglang::scheduler-N")`` / ``sglang::detokenizer``
        # / ``sglang::tp_worker-N`` — the original cmdline can be short.
        out = subprocess.check_output(
            ["ps", "-eo", "pid,uid,comm,cmd", "--no-headers"],
            text=True,
            errors="replace",
        )
        pids = []
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) != 4:
                continue
            pid_s, uid_s, comm, cmd = parts
            try:
                if int(uid_s) != uid:
                    continue
            except ValueError:
                continue
            # Check both comm (may be truncated to 15 chars like
            # "sglang::detoken") and cmd. The ``sglang::`` prefix is the
            # canonical setproctitle marker.
            haystack = f"{comm} {cmd}"
            needles = (
                "sglang::",  # catches sglang::scheduler / detokenizer / tp_worker
                "sglang.srt.managers.scheduler",
                "sglang.srt.managers.detokenizer",
                "sglang_temp_file_",
            )
            if any(n in haystack for n in needles):
                pids.append(int(pid_s))
        return pids

    probes = {"gpu_max": gpu_max, "sglang_pids": sglang_pids}
    func = probes[probe_name]

    task = ray.remote(num_cpus=0)(func)
    refs = []
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        node_id = n["NodeID"]
        addr = n.get("NodeManagerAddress", node_id)
        ref = task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote()
        refs.append((addr, ref))
    return {addr: ray.get(ref) for addr, ref in refs}


def _cluster_max_gpu_mem():
    per_node = _run_on_every_node("gpu_max")
    return (max(per_node.values()) if per_node else 0), per_node


def _cluster_sglang_pids() -> dict:
    return _run_on_every_node("sglang_pids")


@requires_sglang
class ActorCleanupSglangTest(unittest.IsolatedAsyncioTestCase):
    """Regression guard for sglang subprocess leaks after actor teardown."""
    def setUp(self):
        ray.init()

    def tearDown(self):
        # If a test threw mid-way, still clean up.
        kill_all_actors_and_shutdown_ray()

    async def test_gen_rm_actor_cleanup_releases_gpu_and_kills_sglang(self):
        # 1) Baseline: cluster-wide GPU mem low, no sglang procs anywhere.
        baseline, baseline_per_node = _cluster_max_gpu_mem()
        assert baseline < 1 * _GB, (
            f"baseline cluster-max GPU mem {baseline / _GB:.2f}GiB too high "
            f"(per-node: {baseline_per_node}) — prior test probably leaked"
        )
        baseline_sglang = _cluster_sglang_pids()
        for addr, pids in baseline_sglang.items():
            assert pids == [], (f"sglang procs leaked from a prior test on node {addr}: {pids}")

        # 2) Create a gen-rm actor group that owns an sglang engine.
        config = load_config("test_t2i_grpo_gen_rm", T2iRlConfig)
        pgs = create_placement_groups(config)
        gen_rm_groups = create_gen_rm_group(config, pgs)
        assert config.training.use_gen_rm_reward
        for grp in gen_rm_groups:
            await grp.init()

        # 3) After init: cluster-max GPU mem must climb, sglang procs exist.
        after_init, after_init_per_node = _cluster_max_gpu_mem()
        self.assertGreater(
            after_init,
            1 * _GB,
            f"expected gen-rm to load weights (>1GiB) but cluster-max GPU mem "
            f"is {after_init / _GB:.2f}GiB (per-node: {after_init_per_node}) — "
            "init may have silently failed",
        )
        running_sglang = _cluster_sglang_pids()
        total_sglang = sum(len(v) for v in running_sglang.values())
        self.assertGreater(
            total_sglang,
            0,
            f"expected at least one sglang subprocess after gen-rm init "
            f"(per-node: {running_sglang})",
        )

        # 4) Core assertion: invoke graceful shutdown on every named actor
        #    (triggers sglang's kill_process_tree on each actor's host),
        #    then poll cluster-wide until mem returns to baseline AND no
        #    sglang processes remain, or give up after the grace window.
        for info in ray.util.list_named_actors(all_namespaces=True):
            name = info["name"] if isinstance(info, dict) else info
            ns = info.get("namespace") if isinstance(info, dict) else None
            try:
                h = ray.get_actor(name, namespace=ns) if ns else ray.get_actor(name)
            except Exception:
                continue
            method = getattr(h, "shutdown", None)
            if method is None:
                continue
            try:
                ray.get(method.remote(), timeout=30.0)
            except Exception:
                pass  # isolated — tearDown will ray.kill these

        deadline = time.time() + 15.0
        final_mem = after_init
        final_per_node = after_init_per_node
        final_sglang = running_sglang
        while time.time() < deadline:
            final_mem, final_per_node = _cluster_max_gpu_mem()
            final_sglang = _cluster_sglang_pids()
            total = sum(len(v) for v in final_sglang.values())
            if final_mem < 1 * _GB and total == 0:
                break
            time.sleep(0.5)

        self.assertLess(
            final_mem,
            1 * _GB,
            f"cluster-max GPU memory did NOT return to baseline after "
            f"actor.shutdown (max used = {final_mem / _GB:.2f}GiB, "
            f"per-node = {final_per_node})",
        )
        for addr, pids in final_sglang.items():
            self.assertEqual(
                pids,
                [],
                f"sglang processes leaked on node {addr} after shutdown: {pids}",
            )


if __name__ == "__main__":
    unittest.main()
