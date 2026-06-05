import asyncio
import copy
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, Optional

from codetiming import Timer
from transformers import PreTrainedTokenizer, ProcessorMixin

from gpatch_v4.agentic.env import register_custom_env_for_agentic
from gpatch_v4.agentic.env_manager.base_env_manager import BaseEnvManager
from gpatch_v4.agentic.proto import DataProto
from gpatch_v4.agentic.utils import (
    default_processor_provider,
    default_tokenizer_provider,
    get_extra_data_provider,
)
from gpatch_v4.configs.config import AgenticRlConfig
from gpatch_v4.utils import log, safe_import_class
from megatron.core import mpu

"""
environment worker will reside in rl training actor for synchronous train
maybe wrapped in a separate actor for asynchronous train
"""


class EnvironmentWorker:
    """
      Within a group, all environments share identical states by using the same seed.
      To reduce the overhead of dedicating one process per environment, parallelism is redesigned as **process + threads** :
      - One `EnvironmentWorker` holds multiple `EnvStateManager`s.
      - Each `EnvStateManager` manages the rollout loop for a single environment.
      - `EnvStateManager.run_rollout_loop` runs inside dedicated threads.
        TODO: GiGPO: https://arxiv.org/abs/2505.10978

      Rollout base seed: ``ppo_step * stride`` with
      ``stride = dp_size * group_per_worker * max_traj_per_env`` so consecutive PPO steps
      do not reuse the same env RNG / problem indices (no sliding-window overlap).
    """
    def __init__(self, rl_config: AgenticRlConfig):
        self.rl_config: AgenticRlConfig = rl_config
        self.env_managers: Dict[int, BaseEnvManager] = {}
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[ProcessorMixin] = None
        self.thread_lock = threading.Lock()
        self.output_queue = None

    async def initialize(
        self,
        sampler_client,
        output_queue=None,
        mode: str = "train"
    ):

        self.output_queue = queue.Queue()
        model_name_or_path = self.rl_config.sampler.model_info[0].hf_model_path
        self.tokenizer = default_tokenizer_provider(model_name_or_path)
        self.processor = default_processor_provider(model_name_or_path)

        agentic_cfg = self.rl_config.training.agentic
        register_custom_env_for_agentic(agentic_cfg)
        cfg_template = agentic_cfg.env_cfg_template
        group_per_worker = agentic_cfg.train_env_manager.group_per_worker
        group_replicate = agentic_cfg.train_env_manager.group_replicate
        assert group_per_worker > 0
        assert group_replicate > 0

        env_config = {**cfg_template.env_config}
        entry = {}
        entry.update(cfg_template)
        entry.pop("env_config", None)

        dp_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()

        entry.update(
            {
                "tag": "tag",
                "config": env_config,
                "env_class": "sokoban",
                "env_manager_cls": cfg_template.get("env_manager_cls", "gpatch_v4.agentic.env_manager.step_vl_traj_env_manager.StepVLTrajEnvManager"),
            }
        )

        self.env_configs = {}
        for i in range(group_per_worker* group_replicate* dp_rank, group_per_worker * group_replicate * (dp_rank + 1)):
            group_id = i // group_replicate
            entry["env_id"] = i
            entry["group_id"] = group_id
            entry["group_seed"] = group_id
            entry["group_num"] = dp_size * group_per_worker // group_replicate
            self.env_configs[i] = deepcopy(entry)

        def create_env_manager(env_id, env_config):
            if env_id == 0:
                log(f"use env_manager_cls: {env_config['env_manager_cls']}")
            env_manager_cls = safe_import_class(env_config["env_manager_cls"])

            assert env_manager_cls is not None
            tokenizer = copy.deepcopy(self.tokenizer)
            processor = copy.deepcopy(self.processor)
            extra_data_provider = None
            if processor is not None and isinstance(processor, ProcessorMixin):
                extra_data_provider = get_extra_data_provider(
                    model_name_or_path, processor=processor
                )
            return env_id, env_manager_cls(
                rl_config=self.rl_config,
                manager_config=self.rl_config.training.agentic.train_env_manager,
                env_config=env_config,
                sampler_client=sampler_client,
                tokenizer=tokenizer,  # https://github.com/huggingface/tokenizers/issues/537
                processor=processor,
                output_queue=self.output_queue,
                thread_lock=self.thread_lock,
                mode=mode,
                extra_data_provider=extra_data_provider,
            )

        with ThreadPoolExecutor(max_workers=min(len(self.env_configs), 64)) as executor:
            futures = [
                executor.submit(create_env_manager, env_id, env_config)
                for env_id, env_config in self.env_configs.items()
            ]
            for future in as_completed(futures):
                try:
                    env_id, env_manager = future.result()
                    self.env_managers[env_id] = env_manager
                except Exception as e:
                    raise e

        max_traj = max(1, int(self.rl_config.training.agentic.train_env_manager.max_traj_per_env))
        self._rollout_seed_stride = dp_size * group_per_worker * max_traj

    async def run_rollout_loop(self, current_ppo_step, seed):
        loop = asyncio.get_event_loop()
        base_seed = int(seed) * self._rollout_seed_stride
        with ThreadPoolExecutor(max_workers=len(self.env_managers)) as pool:
            try:
                await asyncio.gather(
                    *[
                        loop.run_in_executor(
                            pool, env_manager.run_rollout_loop,
                            DataProto(meta_info={
                                "current_step": current_ppo_step,
                                "seed": base_seed
                            })
                        ) for env_manager in self.env_managers.values()
                    ]
                )
            except Exception as e:
                raise e

    def get_output_data(self):
        data = []
        while not self.output_queue.empty():
            data.append(self.output_queue.get_nowait()[-1])
        return data
