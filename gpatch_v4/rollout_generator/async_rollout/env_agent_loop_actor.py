"""EnvAgentLoopActor – manages repeat_n TrajEnvManager instances for agentic async rollout.

Each ``agent_loop`` call constructs fresh env managers from setup-time templates
and runs ``repeat_n`` trajectories concurrently in worker threads.
"""

import asyncio
import contextvars
import copy
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Type

from transformers import AutoTokenizer

from gpatch_v4.agentic.env import register_custom_env_for_agentic
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import AgentLoopActor
from gpatch_v4.utils import log, safe_import_class


def assign_workers_to_templates(weights: List[float], num_workers: int) -> List[int]:
    """Map each agent-loop worker to exactly one env-template index.

    Each env is guaranteed at least one worker; the remaining workers are split
    by ``weights`` using the largest-remainder method, so the fraction of
    workers bound to an env (hence the fraction of rollout samples it produces)
    tracks its relative weight. Assignments are interleaved across worker ids so
    that, when only a subset of workers fire in a step, the env mix stays
    representative.

    Parameters
    ----------
    weights : list of float
        Per-template relative weights (``EnvTemplateConfig.env_weight``). Length is
        the number of envs.
    num_workers : int
        ``training.num_agent_loop_workers``. Must be ``>= len(weights)``.

    Returns
    -------
    list of int
        ``assignment[worker_id]`` = template index that worker runs. Length
        ``num_workers``.
    """
    num_envs = len(weights)
    assert num_envs > 0, "at least one env template required"
    assert num_workers >= num_envs, (
        f"num_agent_loop_workers ({num_workers}) must be >= number of env "
        f"templates ({num_envs}) so every env gets at least one worker"
    )
    if num_envs == 1:
        return [0] * num_workers

    w = [float(x) for x in weights]
    assert all(x >= 0.0 for x in w), (f"EnvTemplateConfig.env_weight must be non-negative, got {w}")
    total = sum(w)
    if total <= 0.0:
        w = [1.0] * num_envs
        total = float(num_envs)

    # Base 1 worker each, distribute the rest by weight (largest remainder).
    extra = num_workers - num_envs
    ideal = [x / total * extra for x in w]
    counts = [1 + int(v) for v in ideal]  # floor
    remainder = num_workers - sum(counts)
    if remainder > 0:
        frac_order = sorted(range(num_envs), key=lambda e: ideal[e] - int(ideal[e]), reverse=True)
        for e in frac_order[:remainder]:
            counts[e] += 1

    # Interleave so envs are spread across worker ids.
    assignment: List[int] = []
    remaining = counts[:]
    while len(assignment) < num_workers:
        for e in range(num_envs):
            if remaining[e] > 0:
                assignment.append(e)
                remaining[e] -= 1
    return assignment


class EnvAgentLoopActor(AgentLoopActor):
    """Agent-loop actor that drives ``TrajEnvManager`` instances.

    For GRPO, one worker holds ``repeat_n`` env-entry templates; each
    ``agent_loop`` builds new manager instances, runs them in a thread pool,
    and merges the results.
    """
    def __init__(self, config: RlConfig, worker_id: int):
        super().__init__(config, worker_id)
        self._env_manager_cls: Type = None  # type: ignore[assignment]
        self._env_entry_templates: List[Dict[str, Any]] = []
        self._env_run_sem: Optional[asyncio.Semaphore] = None
        self._env_per_actor_slots: int = 0
        self._env_executor: Optional[ThreadPoolExecutor] = None
        self._engine_index_counter: Iterator[int] = itertools.count(start=worker_id)

    async def setup(self):
        await super().setup()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.policy.hf_tokenizer_path,
                use_fast=getattr(self.training_config, "use_fast_tokenizer", True),
                trust_remote_code=True,
            )

        agentic_cfg = self.config.training.agentic
        register_custom_env_for_agentic(agentic_cfg)

        repeat_n = self.training_config.sampling_repeat_n
        num_workers = self.training_config.num_agent_loop_workers

        # Bind this worker to exactly one env template. Single-env runs resolve
        # to a one-element list, so this worker always picks index 0.
        templates = agentic_cfg.resolved_env_templates()
        weights = [t.env_weight for t in templates]
        assignment = assign_workers_to_templates(weights, num_workers)
        my_tmpl_idx = assignment[self.worker_id]
        my_template = templates[my_tmpl_idx]

        env_config = {**my_template.env_config}
        base_entry = {**my_template}
        base_entry.pop("env_config", None)
        base_entry["config"] = env_config
        base_entry["env_name"] = my_template.env_type

        env_manager_cls_path = my_template.env_manager_cls
        env_manager_cls = safe_import_class(env_manager_cls_path)
        assert env_manager_cls is not None, (
            f"Cannot import env_manager_cls: {env_manager_cls_path}"
        )
        self._env_manager_cls = env_manager_cls

        self._env_entry_templates = []
        for i in range(repeat_n):
            entry = deepcopy(base_entry)
            entry["env_id"] = i
            entry["group_id"] = self.worker_id
            entry["tag"] = f"tag_{self.worker_id}"
            self._env_entry_templates.append(entry)

        total_cap = getattr(agentic_cfg, "max_concurrency", 0) or 0
        if total_cap > 0:
            per_actor = max(1, total_cap // max(num_workers, 1))
            self._env_run_sem = asyncio.Semaphore(per_actor)
            self._env_per_actor_slots = per_actor
            if total_cap < num_workers:
                log(
                    f"[EnvAgentLoopActor-{self.worker_id}] agentic.max_concurrency={total_cap} "
                    f"< num_agent_loop_workers={num_workers}: each actor still gets at least "
                    f"1 slot; cluster-wide traj concurrency may exceed max_concurrency."
                )
        else:
            self._env_run_sem = nullcontext()
            self._env_per_actor_slots = 0

        if self._env_per_actor_slots > 0:
            self._env_executor = ThreadPoolExecutor(
                max_workers=self._env_per_actor_slots,
                thread_name_prefix=f"env-{self.worker_id}",
            )

        log(
            f"[EnvAgentLoopActor-{self.worker_id}] setup done, "
            f"env='{my_template.env_type}' (tmpl_idx={my_tmpl_idx}/{len(templates)}, "
            f"worker_id={self.worker_id}), "
            f"{repeat_n} env entry templates x {env_manager_cls.__name__}, "
            f"traj_sem_per_actor={self._env_per_actor_slots} "
            f"(agentic.max_concurrency={total_cap}, num_agent_loop_workers={num_workers})"
        )

    def _build_managers_for_step(self) -> List[Any]:
        """Construct one TrajEnvManager (or subclass) per repeat slot."""
        managers = []
        for entry_template in self._env_entry_templates:
            entry = deepcopy(entry_template)
            entry["engine_index"] = next(self._engine_index_counter)
            mgr = self._env_manager_cls(
                rl_config=self.config,
                env_config=entry,
                tokenizer=copy.deepcopy(self.tokenizer),
                sampler_client=self.sampler_client,
            )
            managers.append(mgr)
        return managers

    async def _run_env_in_thread(
        self,
        mgr: Any,
        seed: int,
        ppo_step: int,
        cleaned_data: Dict[str, Any],
    ) -> Dict[str, List[Any]]:
        """Run ``mgr.run`` in a worker thread, optionally under process-wide cap."""
        async with self._env_run_sem:
            loop = asyncio.get_running_loop()
            context = contextvars.copy_context()
            executor_future = loop.run_in_executor(
                self._env_executor,
                context.run,
                mgr.run,
                seed,
                ppo_step,
                cleaned_data,
            )
            try:
                return await asyncio.shield(executor_future)
            except asyncio.CancelledError:
                await executor_future
                raise

    @staticmethod
    def _merge_env_results(results: list, cleaned_data: dict, repeat_n: int, group_id: int) -> dict:
        """Merge ``repeat_n`` env trajectory results into one rollout batch.

        Layout is traj-major: traj_0 segments, then traj_1, ...
        Assigns ``group_id`` / ``traj_id`` / ``segment_id`` for each segment.
        """
        assert len(results) == repeat_n, f"len(results) expect {repeat_n}, but get {len(results)}"
        merged: dict = {}
        first_key = list(results[0].keys())[0]
        unique_id = cleaned_data.get("unique_id", [group_id])[0]
        for traj_id in range(len(results)):
            num_segments = len(results[traj_id][first_key])
            assert num_segments > 0, f"traj {traj_id} has zero segments"
            results[traj_id]["unique_id"] = [unique_id] * num_segments
            results[traj_id]["group_id"] = [unique_id] * num_segments
            results[traj_id]["traj_id"] = [traj_id] * num_segments
            results[traj_id]["segment_id"] = list(range(num_segments))
        for key in results[0]:
            merged[key] = []
            for r in results:
                merged[key].extend(r[key])
        return merged

    async def _run_prompt_group(
        self,
        cleaned_data: Dict[str, Any],
        sample_idx: int,
        ppo_step: int,
    ) -> Dict[str, List[Any]]:
        managers = self._build_managers_for_step()
        repeat_n = len(managers)
        group_seed = ppo_step * 10000 + sample_idx * repeat_n
        started_at = time.time()
        log(
            f"[env_rollout] BEGIN worker_id={self.worker_id} "
            f"ppo_step={ppo_step} sample_idx={sample_idx} "
            f"repeat_n={repeat_n} group_seed={group_seed}"
        )

        results = await asyncio.gather(
            *[
                self._run_env_in_thread(
                    manager,
                    group_seed,
                    ppo_step,
                    cleaned_data,
                ) for manager in managers
            ]
        )
        merged = self._merge_env_results(
            results,
            cleaned_data,
            repeat_n,
            group_seed,
        )
        log(
            f"[env_rollout] END worker_id={self.worker_id} "
            f"ppo_step={ppo_step} sample_idx={sample_idx} "
            f"repeat_n={repeat_n} elapsed_s={time.time() - started_at:.3f} "
            f"num_segments={len(merged['tokens'])} "
            f"traj_id={merged['traj_id']} segment_id={merged['segment_id']} "
            f"group_id={merged['group_id']}"
        )
        return merged

    async def generate_batches(
        self,
        cleaned_batches: List[Dict[str, Any]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
        on_ready=None,
    ) -> List[Dict[str, List[Any]]]:
        """Run env trajectories for all microbatches.

        Wrapped in ``sampler_phase`` when colocate.  Microbatch groups run
        concurrently here, so ``on_ready(rbi, rb)`` fires as each one merges --
        in completion order, not index order -- letting its reward overlap the
        groups still running.  The returned list stays in index order.
        """
        async def run_group(rbi, cleaned_data, sample_idx):
            merged = await self._run_prompt_group(cleaned_data, sample_idx, ppo_step)
            if on_ready is not None:
                on_ready(rbi, merged)
            return merged

        ctx = self.sampler_phase(ppo_step) if use_colocate else nullcontext()
        async with ctx:
            group_tasks = []
            for rbi, (cleaned_data, sample_idx) in enumerate(zip(cleaned_batches, sample_indices)):
                group_tasks.append(asyncio.create_task(run_group(rbi, cleaned_data, sample_idx)))
            return list(await asyncio.gather(*group_tasks))
