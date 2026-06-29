"""EnvAgentLoopActor – manages repeat_n TrajEnvManager instances for agentic async rollout.

Each ``agent_loop`` call constructs fresh env managers from setup-time templates
and runs ``repeat_n`` trajectories concurrently via ``asyncio.to_thread``.
"""

import asyncio
import copy
import itertools
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Type

from transformers import AutoTokenizer

from gpatch_v4.agentic.env import register_custom_env_for_agentic
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import AgentLoopActor
from gpatch_v4.utils import log, safe_import_class


class EnvAgentLoopActor(AgentLoopActor):
    """Agent-loop actor that drives ``TrajEnvManager`` instances.

    For GRPO, one worker holds ``repeat_n`` env-entry templates; each
    ``agent_loop`` builds new manager instances, runs them in a thread pool
    via ``asyncio.to_thread``, and merges the results.
    """
    def __init__(self, config: RlConfig, worker_id: int):
        super().__init__(config, worker_id)
        self._env_manager_cls: Type = None  # type: ignore[assignment]
        self._env_entry_templates: List[Dict[str, Any]] = []
        self._env_run_sem: Optional[asyncio.Semaphore] = None
        self._env_per_actor_slots: int = 0
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
        env_config = {**agentic_cfg.env_cfg_template.env_config}
        base_entry = {**agentic_cfg.env_cfg_template}
        base_entry.pop("env_config", None)
        base_entry["config"] = env_config

        env_manager_cls_path = agentic_cfg.env_cfg_template.env_manager_cls
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

        num_workers = self.training_config.num_agent_loop_workers
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

        log(
            f"[EnvAgentLoopActor-{self.worker_id}] setup done, "
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
            return await asyncio.to_thread(mgr.run, seed, ppo_step, cleaned_data)

    @staticmethod
    def _merge_env_results(results: list, cleaned_data: dict, repeat_n: int) -> dict:
        """Merge ``repeat_n`` env trajectory results into one rollout batch."""
        merged: dict = {}
        for key in results[0]:
            merged[key] = []
            for r in results:
                merged[key].extend(r[key])
        if "unique_id" in cleaned_data:
            uid = cleaned_data["unique_id"][0]
            merged["unique_id"] = [uid] * repeat_n
        return merged

    async def generate_batches(
        self,
        cleaned_batches: List[Dict[str, Any]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Run env trajectories for all microbatches.

        Wrapped in ``sampler_phase`` when colocate; plain execution otherwise.
        """
        ctx = self.sampler_phase(ppo_step) if use_colocate else nullcontext()
        async with ctx:
            all_merged = []
            for cd, sidx in zip(cleaned_batches, sample_indices):
                managers = self._build_managers_for_step()
                repeat_n = len(managers)
                #注意：同 group（同 ppo_step + sidx）内所有轨迹共享同一 seed，不同 group seed 不同，不依赖数据文件。
                group_seed = ppo_step * 10000 + sidx * repeat_n
                _t0 = time.time()
                log(
                    f"[env_rollout] BEGIN worker_id={self.worker_id} "
                    f"ppo_step={ppo_step} sample_idx={sidx} repeat_n={repeat_n} group_seed={group_seed}"
                )
                results = await asyncio.gather(
                    *[
                        self._run_env_in_thread(mgr, group_seed, ppo_step, cd)
                        for j, mgr in enumerate(managers)
                    ]
                )
                all_merged.append(self._merge_env_results(results, cd, repeat_n))
                log(
                    f"[env_rollout] END worker_id={self.worker_id} "
                    f"ppo_step={ppo_step} sample_idx={sidx} repeat_n={repeat_n} "
                    f"elapsed_s={time.time() - _t0:.3f}"
                )
        return all_merged
