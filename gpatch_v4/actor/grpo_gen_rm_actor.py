import copy
import gc
import inspect
import os
from typing import Any, Dict, List, Optional

import ray
import torch
from typing_extensions import override

from gpatch_v4.actor.mixin import TokenizerMixin
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.parallel_state import init_pg, initlize_parallel_state
from gpatch_v4.generation_backend import InferEngine
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.orches.utils import ray_noset_visible_devices
from gpatch_v4.patch.sglang_patch import sglang_hack
from gpatch_v4.utils import (
    import_fn_from_path,
    kill_process_tree,
    log,
    logging_memory_usage_details,
)
from gpatch_v4.utils.logging_utils import write_infer_engine_log_marker


class GrpoGenRmActor(BaseActor, TokenizerMixin):
    """Ray actor wrapping an inference engine for LLM generative reward model.

    This actor handles text-only generative reward model inference,
    unlike T2iGrpoGenRmActor which also handles multi-modal (image) inputs.
    """
    def write_engine_log_marker(self, ppo_step: int, phase: str = "begin"):
        write_infer_engine_log_marker(ppo_step, phase)

    @override
    def shutdown(self) -> None:
        """Reap sglang subprocess tree owned by this actor's infer engine.

        See ``GrpoSamplerActor.shutdown`` for the rationale.
        """
        infer_engine = self.infer_engine
        if infer_engine is not None:
            infer_engine.shutdown()
            self.infer_engine = None
            return
        kill_process_tree(os.getpid(), include_parent=False)

    async def init(self, config):
        super().init(config, pg_backend='nccl')
        fake_dist_config = DistConfig()
        initlize_parallel_state(config, fake_dist_config)
        init_pg(fake_dist_config)
        # 推理引擎只需在主节点上控制
        self._is_master_node = None
        self.infer_engine = None
        self._infer_engine_kwargs: Optional[Dict[str, Any]] = None
        self._engine_idx = None

    async def init_infer_engine(
        self,
        config,
        dist_init_addr,
        rm_idx,
        engine_idx,
        tp_rank,
        is_master_node,
        pg_bundle_indices=None,
    ):
        await self._configure_infer_engine(
            config,
            dist_init_addr,
            rm_idx,
            engine_idx,
            tp_rank,
            is_master_node,
            pg_bundle_indices=pg_bundle_indices,
        )
        if not self._destroy_engine_after_generation():
            await self.ensure_infer_engine({})

    async def _configure_infer_engine(
        self,
        config,
        dist_init_addr,
        rm_idx,
        engine_idx,
        tp_rank,
        is_master_node,
        pg_bundle_indices=None,
    ):
        if config.gen_rm.backend == "vllm":
            import os
            os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        else:
            sglang_hack()
        self._is_master_node = is_master_node
        self.rm_idx = rm_idx
        self._engine_idx = engine_idx
        infer_engine_config = config.gen_rm.infer_engine_configs[rm_idx]
        dist_config = infer_engine_config.dist_config
        g_rank = torch.distributed.get_rank()
        self.model_arch = config.gen_rm.reward_model_info[rm_idx].model_arch
        hf_model_path = config.gen_rm.reward_model_info[rm_idx].hf_model_path

        base_gpu_id = None
        if ray_noset_visible_devices():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                base_gpu_id = int(gpu_ids[0])

        log(
            f'GrpoGenRmActor.init create infer engine {rm_idx=} {g_rank=} {engine_idx=} {tp_rank=} {base_gpu_id=}'
        )

        model_specific_kwargs = {}
        if self.model_arch == MODEL_ARCH.WELMV4_MOE:
            model_specific_kwargs = {
                "disable_overlap_schedule": True,
                "enable_over_encoding": True,
                "disable_radix_cache": True,
                "disable_piecewise_cuda_graph": True,
            }
            assert config.gen_rm.backend == "sglang", "disable_piecewise_cuda_graph is only supported for sglang backend"
        extra_infer_engine_config = model_specific_kwargs
        override_infer_engine_config = infer_engine_config.override_infer_engine_config or {}
        extra_infer_engine_config.update(override_infer_engine_config)

        self._infer_engine_kwargs = dict(
            infer_engine_impl=config.gen_rm.backend,
            model_path=hf_model_path,
            dtype=infer_engine_config.dtype,
            tensor_parallel_size=dist_config.tensor_model_parallel_size,
            pipeline_parallel_size=dist_config.pipeline_model_parallel_size,
            expert_parallel_size=dist_config.expert_model_parallel_size,
            enable_deepep=infer_engine_config.enable_deepep_moe,
            gpu_memory_utilization=infer_engine_config.gpu_memory_utilization,
            enforce_eager=False,
            tp_rank=tp_rank,
            engine_idx=engine_idx,
            dist_init_addr=dist_init_addr,
            num_gpus_per_node=dist_config.num_gpus_per_node,
            infer_engine_role='gen-rm',
            rm_idx=rm_idx,
            use_fast=infer_engine_config.use_fast_tokenizer,
            max_running_requests=infer_engine_config.max_running_requests,
            load_format=infer_engine_config.load_format,
            log_level='info',
            allow_auto_truncate=infer_engine_config.allow_auto_truncate,
            enable_custom_logit_processor=False,
            attention_backend=config.infer_result.attention_backend
            if hasattr(config, 'infer_result') else infer_engine_config.attention_backend,
            mm_per_request_timeout=infer_engine_config.mm_per_request_timeout,
            sgl_chunked_prefill_size=infer_engine_config.sgl_chunked_prefill_size,
            sgl_schedule_conservativeness=infer_engine_config.sgl_schedule_conservativeness,
            sgl_sleep_on_idle=infer_engine_config.sgl_sleep_on_idle,
            sgl_mamba_full_memory_ratio=infer_engine_config.sgl_mamba_full_memory_ratio,
            sgl_mamba_scheduler_strategy=infer_engine_config.sgl_mamba_scheduler_strategy,
            sgl_enable_spec_v2=infer_engine_config.sgl_enable_spec_v2,
            apply_deterministic_mode=getattr(config.training, "apply_deterministic_mode", False),
            placement_type=config.placement_type,
            pg_bundle_indices=pg_bundle_indices,
            base_gpu_id=base_gpu_id,
            seed=infer_engine_config.engine_seed,
            **extra_infer_engine_config,
        )
        self.build_tokenizer()
        assert rm_idx < len(self.gen_rm_tokenizers)
        self.tokenizer = self.gen_rm_tokenizers[rm_idx]
        self.post_init(self.rm_idx)

    async def ensure_infer_engine(self, req_dict=None):
        """Create the sglang gen-RM engine if this actor does not own one."""
        del req_dict
        if self.infer_engine is not None:
            return {"ret": True}
        assert self._infer_engine_kwargs is not None, "gen-RM infer engine is not configured"
        infer_engine_impl = self._infer_engine_kwargs["infer_engine_impl"]
        kwargs = {k: v for k, v in self._infer_engine_kwargs.items() if k != "infer_engine_impl"}
        log(
            f"GrpoGenRmActor.ensure create infer engine rm_idx={self.rm_idx} "
            f"engine_idx={kwargs['engine_idx']} tp_rank={kwargs['tp_rank']}"
        )
        self.infer_engine = InferEngine.from_engine_args(
            infer_engine_impl,
            **kwargs,
        )
        return {"ret": True}

    async def destroy_infer_engine(self, req_dict=None):
        """Destroy this actor's gen-RM engine and reap sglang child processes."""
        del req_dict
        mem_tag = f"gen-rm destroy infer engine rm_idx={self.rm_idx} engine_idx={self._engine_idx}"
        logging_memory_usage_details(f"{mem_tag} before")
        infer_engine = self.infer_engine
        self.infer_engine = None
        if infer_engine is not None:
            infer_engine.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
        logging_memory_usage_details(f"{mem_tag} after")
        return {"ret": True}

    def _destroy_engine_after_generation(self) -> bool:
        return self.config.gen_rm.destroy_engine_after_generation

    def post_init(self, rm_idx):
        reward_model_info = self.config.gen_rm.reward_model_info[rm_idx]
        self.generate_rewards_fn = import_fn_from_path(
            reward_model_info.reward_py_path, reward_model_info.gen_reward_fn_name
        )
        self.reward_repeat_n = reward_model_info.gen_reward_repeat_n
        fn_kwargs = inspect.signature(self.generate_rewards_fn).parameters
        cond1 = all(
            [
                len(fn_kwargs) == 7,
                'config' in fn_kwargs,
                'rm_infer_engine' in fn_kwargs,
                'rm_idx' in fn_kwargs,
                'rm_tokenizer' in fn_kwargs,
                'actor_tokenizer' in fn_kwargs,
                'batched_data' in fn_kwargs,
                'reward_repeat_n' in fn_kwargs,
            ]
        )
        assert cond1, f"unexpected prompt_fn signature: {fn_kwargs}"

    async def sleep(self, req_dict):
        """Put the inference engine to sleep."""
        assert self.config.placement_type != "disaggregated"
        if not self._is_master_node:
            return {"ret": True}
        if self.infer_engine is None:
            return {"ret": True}
        await self.infer_engine.sleep()
        return {"ret": True}

    async def wake_up(self, req_dict):
        """Wake up the inference engine for specified tag groups."""
        assert self.config.placement_type != "disaggregated"
        if not self._is_master_node:
            return {"ret": True}
        if self.infer_engine is None:
            return {"ret": True}
        tags = req_dict["tag_names"]
        await self.infer_engine.wake_up(tags=tags)
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict):
        """Wake up the engine at the start of a PPO step."""
        ppo_step = req_dict.get("ppo_step", -1)
        if self._destroy_engine_after_generation():
            await self.ensure_infer_engine({})
            if self._is_master_node:
                write_infer_engine_log_marker(ppo_step, phase="begin")
            return {"ret": True}

        if not self._is_master_node:
            return {"ret": True}

        write_infer_engine_log_marker(ppo_step, phase="begin")
        if self.config.placement_type != "disaggregated":
            tags = req_dict["tag_names"]
            await self.infer_engine.wake_up(tags=tags)
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict):
        """Put the engine to sleep at the end of a PPO step."""
        ppo_step = req_dict.get("ppo_step", -1)
        if self._destroy_engine_after_generation():
            if self._is_master_node:
                write_infer_engine_log_marker(ppo_step, phase="end")
            await self.destroy_infer_engine({})
            return {"ret": True}

        if not self._is_master_node:
            return {"ret": True}

        if self.config.placement_type != "disaggregated":
            await self.infer_engine.sleep()
        write_infer_engine_log_marker(ppo_step, phase="end")
        return {"ret": True}

    async def generate_rewards(self, req_dict: Dict[str, Any]):
        if not self._is_master_node:
            return {"ret": True}
        batched_data: Dict[str, List[Any]] = req_dict["batched_data"]
        resp_dict = await self.generate_rewards_fn(
            config=self.config,
            rm_infer_engine=self.infer_engine,
            rm_idx=self.rm_idx,
            rm_tokenizer=self.tokenizer,
            actor_tokenizer=self.actor_tokenizer,
            batched_data=batched_data,
            reward_repeat_n=self.reward_repeat_n,
        )
        ret_rb = copy.deepcopy(batched_data)
        for k, v in resp_dict.items():
            ret_rb[k] = v

        return ret_rb

    async def flush_cache(self, req_dict):
        """Flush the inference engine KV cache."""
        if not self._is_master_node:
            return {"ret": True}
        if self.infer_engine is None:
            return {"ret": True}
        await self.infer_engine.flush_cache()
        return {"ret": True}
