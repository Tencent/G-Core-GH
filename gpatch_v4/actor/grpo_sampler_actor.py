import asyncio
import inspect
import os
import uuid
from typing import Any, Dict, List

import ray
import torch
from typing_extensions import override

from gpatch_v4.actor.mixin import TokenizerMixin
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.parallel_state import init_pg, initlize_parallel_state
from gpatch_v4.extended_model import SamplerGenerateFuncFactory
from gpatch_v4.generation_backend import InferEngine
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.orches.utils import ray_noset_visible_devices
from gpatch_v4.patch.sglang_patch import sglang_hack
from gpatch_v4.utils import import_fn_from_path, kill_process_tree, log, perf_time
from gpatch_v4.utils.logging_utils import write_infer_engine_log_marker


class GrpoSamplerActor(BaseActor, TokenizerMixin):
    """Ray actor wrapping an inference engine for GRPO sampling.

    References
    ----------
    .. [1] https://huggingface.co/docs/transformers/main/en/chat_templating#addgenerationprompt
    .. [2] https://github.com/QwenLM/Qwen3-VL
    """
    @override
    def shutdown(self) -> None:
        """Reap sglang subprocess tree owned by this actor's infer engine.

        Called explicitly by tearDown before ``ray.kill(actor)``. sglang
        spawns scheduler / TP worker / detokenizer as ``multiprocessing``
        children of this actor process; if we do nothing, ``ray.kill``
        SIGKILLs the ray worker and those children are reparented to PID 1
        as orphans still holding GPU memory.

        For vllm backend, this is still called but ``kill_process_tree`` is
        safe: it simply walks our children and SIGKILLs any that exist,
        which is cheap and idempotent when there are none. The import is
        deferred into the method body so that importing this module on a
        vllm-only install (where ``sglang`` may not be present) does not
        fail.
        """
        kill_process_tree(os.getpid(), include_parent=False)

    async def init(self, config):
        """Initialize parallel state.

        Parameters
        ----------
        config : RlConfig
        """
        super().init(config, pg_backend='nccl')  # TODO make it gloo to save gpu mem
        fake_dist_config = DistConfig()
        initlize_parallel_state(config, fake_dist_config)
        init_pg(fake_dist_config)
        self._is_master_node = None

    def write_engine_log_marker(self, ppo_step: int, phase: str = "begin"):
        write_infer_engine_log_marker(ppo_step, phase)

    async def init_infer_engine(
        self,
        config,
        dist_init_addr,
        idx,
        engine_idx,
        tp_rank,
        is_master_node,
        pg_bundle_indices=None,
    ):
        """Create and configure the inference engine.

        Parameters
        ----------
        config : RlConfig
        dist_init_addr : str
        idx : int
        engine_idx : int
        tp_rank : int
        is_master_node : bool
        pg_bundle_indices : list[int] or None
            Placement group bundle indices for this engine's TP workers.
        """
        self._is_master_node = is_master_node
        if config.sampler.backend == "vllm":
            import os
            os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        else:
            sglang_hack()
        self.idx = idx
        g_rank = torch.distributed.get_rank()
        infer_engine_config = config.sampler.infer_engine_configs[idx]
        dist_config = infer_engine_config.dist_config
        self.model_arch = config.sampler.model_info[idx].model_arch
        hf_model_path = config.sampler.model_info[idx].hf_model_path

        base_gpu_id = None
        if ray_noset_visible_devices():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                base_gpu_id = int(gpu_ids[0])

        log(
            f'{self.__class__.__name__} init create infer engine {idx=} {g_rank=} {engine_idx=} {tp_rank=} {base_gpu_id=}'
        )

        load_format = infer_engine_config.load_format
        if hasattr(self.config, 'debug') and self.config.debug.debug_engine_update_weight:
            load_format = "auto"

        model_specific_kwargs = {}
        if self.model_arch == MODEL_ARCH.WELMV4_MOE:
            model_specific_kwargs = {
                "disable_overlap_schedule": True,
                "enable_over_encoding": True,
                "disable_radix_cache": True,
                "disable_piecewise_cuda_graph": True,
            }
            assert config.sampler.backend == "sglang", "disable_piecewise_cuda_graph is only supported for sglang backend"
        extra_infer_engine_config = model_specific_kwargs
        override_infer_engine_config = infer_engine_config.override_infer_engine_config or {}
        extra_infer_engine_config.update(override_infer_engine_config)

        self.infer_engine = InferEngine.from_engine_args(
            config.sampler.backend,
            model_path=hf_model_path,
            dtype=infer_engine_config.dtype,
            tensor_parallel_size=dist_config.tensor_model_parallel_size,
            pipeline_parallel_size=dist_config.pipeline_model_parallel_size,
            expert_parallel_size=dist_config.expert_model_parallel_size,
            enable_deepep=infer_engine_config.enable_deepep_moe,
            gpu_memory_utilization=infer_engine_config.gpu_memory_utilization,
            enforce_eager=infer_engine_config.disable_cuda_graph,
            tp_rank=tp_rank,
            engine_idx=engine_idx,
            dist_init_addr=dist_init_addr,
            num_gpus_per_node=dist_config.num_gpus_per_node,
            infer_engine_role=config.sampler.sampler_type,
            use_fast=infer_engine_config.use_fast_tokenizer,
            max_running_requests=infer_engine_config.max_running_requests,
            load_format=load_format,
            log_level='info',
            allow_auto_truncate=infer_engine_config.allow_auto_truncate,
            enable_custom_logit_processor=False,
            attention_backend=config.infer_result.attention_backend
            if hasattr(config, 'infer_result') else infer_engine_config.attention_backend,
            mm_per_request_timeout=infer_engine_config.mm_per_request_timeout,
            sgl_crash_dump_folder=infer_engine_config.sgl_crash_dump_folder,
            sgl_chunked_prefill_size=infer_engine_config.sgl_chunked_prefill_size,
            sgl_schedule_conservativeness=infer_engine_config.sgl_schedule_conservativeness,
            sgl_sleep_on_idle=infer_engine_config.sgl_sleep_on_idle,
            sgl_mamba_full_memory_ratio=infer_engine_config.sgl_mamba_full_memory_ratio,
            sgl_mamba_scheduler_strategy=infer_engine_config.sgl_mamba_scheduler_strategy,
            sgl_enable_spec_v2=infer_engine_config.sgl_enable_spec_v2,
            enable_return_routed_experts=config.training.moe_router_replay
            if hasattr(config, 'training') else False,
            apply_deterministic_mode=getattr(config.training, "apply_deterministic_mode", False),
            placement_type=config.placement_type,
            pg_bundle_indices=pg_bundle_indices,
            base_gpu_id=base_gpu_id,
            enable_mtp=(
                self.config.training.enable_mtp and
                getattr(self.config.training, "online_mtp_sft", False)
            ),
            **extra_infer_engine_config,
        )
        self.build_tokenizer()
        self.tokenizer = self.sampler_tokenizers[idx]
        self.load_hf_config()
        self.post_init(self.idx)

    def post_init(self, idx):
        """Load custom generation function or fall back to factory default.

        Parameters
        ----------
        idx : int
        """
        model_info = self.config.sampler.model_info[idx]
        if model_info.gen_rollout_py_path is not None and model_info.gen_rollout_fn_name is not None:
            generate_func = import_fn_from_path(
                model_info.gen_rollout_py_path, model_info.gen_rollout_fn_name
            )
            fn_kwargs = inspect.signature(generate_func).parameters
            required_keys = {
                'config',
                'infer_engine',
                'idx',
                'tokenizer',
                'batched_data',
                'sampling_repeat_n',
            }
            cond1 = all(
                [len(fn_kwargs) >= len(required_keys)] + [k in fn_kwargs for k in required_keys]
            )
            assert cond1, f"unexpected {cond1} {fn_kwargs}"
            self._extra_gen_args = set(fn_kwargs.keys()) - required_keys
            self.generate_func = generate_func
        else:
            self._extra_gen_args = set()
            self.generate_func = SamplerGenerateFuncFactory.get_gen_func(self.config, idx)

    async def sleep(self, req_dict):
        """Put the inference engine to sleep."""
        assert self.config.placement_type != "disaggregated"
        log(f"sleep called on rank {self._rank=}")
        if not self._is_master_node:
            return {"ret": True}
        await self.infer_engine.sleep()
        return {"ret": True}

    async def wake_up(self, req_dict):
        """Wake up the inference engine for specified tag groups."""
        assert self.config.placement_type != "disaggregated"
        if not self._is_master_node:
            return {"ret": True}
        tags = req_dict["tag_names"]
        await self.infer_engine.wake_up(tags=tags)
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        ppo_step = req_dict.get("ppo_step", -1)
        write_infer_engine_log_marker(ppo_step, phase="begin")
        if self.config.placement_type != "disaggregated":
            tags = req_dict["tag_names"]
            await self.infer_engine.wake_up(tags=tags)
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        if self.config.placement_type != "disaggregated":
            await self.infer_engine.sleep()
        ppo_step = req_dict.get("ppo_step", -1)
        write_infer_engine_log_marker(ppo_step, phase="end")
        return {"ret": True}

    async def update_weights(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        with perf_time(f"update_weight", rank=0):
            ret = await self.infer_engine.update_weights(**req_dict)
            log(f"sampler update weight result {ret}", rank=0)
        return {"ret": ret}

    async def update_weights_bucketed(self, req_dict):
        """Deliver one flat-IPC weight bucket to every colocated vLLM worker.

        ``req_dict`` is the per-bucket payload produced by
        :class:`FlatIpcBucketBuilder` (see
        ``gpatch_v4/generation_backend/bucketed_ipc_transfer.py``) plus the
        merged ``{gpu_uuid: ipc_handle}`` dict built by the trainer-side
        all_gather.
        """
        if not self._is_master_node:
            return {"ret": True}
        with perf_time("update_weight_bucketed", rank=0):
            await self.infer_engine.update_weights_bucketed(req_dict)
        return {"ret": True}

    async def init_weights_update_group(self, req_dict):
        """Create an NCCL group for distributed weight updates."""
        if not self._is_master_node:
            return {"ret": True}
        ret = await self.infer_engine.init_weights_update_group(
            master_address=req_dict["master_address"],
            master_port=req_dict["master_port"],
            rank_offset=req_dict["rank_offset"],
            world_size=req_dict["world_size"],
            group_name=req_dict["group_name"],
            backend=req_dict.get("backend", "nccl"),
        )
        log(f"sampler init_weights_update_group result {ret}", rank=0)
        return {"ret": ret}

    async def update_weights_from_distributed(self, req_dict):
        """Receive weights from the training rank."""
        if not self._is_master_node:
            return {"ret": True}
        with perf_time("update_weight_from_distributed", rank=0):
            ret = await self.infer_engine.update_weights_from_distributed(req_dict)
            log(f"sampler update_weights_from_distributed result {ret}", rank=0)
        return {"ret": ret}

    async def finalize_weights_update(self, req_dict=None):
        """Run ``process_weights_after_loading`` once per full weight update.

        Transport-agnostic finalize -- serves both the bucketed-IPC and
        NCCL-distributed paths; the trainer calls this exactly once after
        all buckets of an update have been acked.
        """
        if not self._is_master_node:
            return {"ret": True}
        with perf_time("finalize_weight_update", rank=0):
            await self.infer_engine.finalize_weights_update()
        return {"ret": True}

    def get_gpu_uuids(self):
        """Return the physical GPU UUIDs visible on this node."""
        uuids = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            uuids.append(str(props.uuid))
        return uuids

    async def destroy_weights_update_group(self, req_dict):
        """Destroy the NCCL group for distributed weight updates."""
        if not self._is_master_node:
            return {"ret": True}
        ret = await self.infer_engine.destroy_weights_update_group(
            group_name=req_dict["group_name"],
        )
        log(f"sampler destroy_weights_update_group result {ret}", rank=0)
        return {"ret": ret}

    async def flush_cache(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        await self.infer_engine.flush_cache()
        log("sampler flush_cache done", rank=0)
        return {"ret": True}

    async def get_load(self, req_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """Return scheduler load metrics for this engine.

        Returns
        -------
        dict
            ``num_reqs``: running + waiting request count.
            ``num_running_reqs``: running-only request count.
            ``num_waiting_reqs``: waiting-only request count.
        """
        assert self._is_master_node, ("get_load must only be called on master-node actors")
        return await self.infer_engine.get_load()

    async def generate(self, req_dict: Dict[str, Any]):
        """Generate completions for a batch of inputs.

        Parameters
        ----------
        req_dict : dict
            Request with ``'batched_data'`` and ``'sampling_repeat'``.

        Returns
        -------
        dict
            Rollout batch with generated tokens, logprobs, etc.
        """
        if not self._is_master_node:
            return {"ret": True}
        batched_data: Dict[str, List[Any]] = req_dict["batched_data"]
        repeat_n = req_dict["sampling_repeat"]
        if hasattr(self.config, 'training'):
            rollout_mbs = self.config.training.rollout_mbs
            for k, v in batched_data.items():
                assert isinstance(
                    v, list
                ) and len(v) == rollout_mbs, f'unexpected {k=} {v=} {rollout_mbs=} {len(v)=}'

        gen_fn_kwargs = dict(
            config=self.config,
            infer_engine=self.infer_engine,
            idx=self.idx,
            tokenizer=self.tokenizer,
            batched_data=batched_data,
            sampling_repeat_n=repeat_n
        )
        for k in self._extra_gen_args:
            if hasattr(self, k):
                gen_fn_kwargs[k] = getattr(self, k)

        rollout_batch = await self.generate_func(**gen_fn_kwargs)

        return rollout_batch

    async def test_generate(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        raw_prompts = req_dict.pop("prompts")
        if not isinstance(raw_prompts, list):
            raw_prompts = [raw_prompts]
        prompts = []
        for prompt in raw_prompts:
            chat = [
                {
                    'role': 'user',
                    'content': prompt
                },
            ]
            text = self.tokenizer.apply_chat_template(
                chat,
                add_special_tokens=False,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
            prompts.append(text)

        res_gens = []
        max_tokens = req_dict.get("max_tokens", 128)
        sampling_params = self.infer_engine.get_sampling_params(
            temperature=0., top_k=1, seed=123, n=1, max_tokens=max_tokens
        )
        for prompt in prompts:
            input_dict = {
                'prompt_token_ids': self.tokenizer(prompt, add_special_tokens=False).input_ids,
            }
            res_gens.append(
                self.infer_engine.async_generate(
                    input_dict, sampling_params, str(uuid.uuid4().hex)
                )
            )
        outputs = await self.infer_engine.wait_and_get_async_generate_output(res_gens)

        if self.config.sampler.backend == 'sglang':
            for output in outputs:
                output.outputs[0].text = self.tokenizer.decode(
                    output.outputs[0].token_ids, skip_special_tokens=False
                )
        text_outputs = [prompt + output.outputs[0].text for prompt, output in zip(prompts, outputs)]
        output_token_ids = [output.outputs[0].token_ids for output in outputs]
        log(f"test_generate {text_outputs=} {output_token_ids=}")
        return {"ret": "ok", "text_outputs": text_outputs, "output_token_ids": output_token_ids}

    async def save_engine_ckpt(self, req_dict):
        """Save the inference engine checkpoint to disk."""
        if not self._is_master_node:
            return {"ret": True}
        import os
        save_ckpt_dir = f"{req_dict['save_ckpt_dir']}_{torch.distributed.get_rank()}"
        if not os.path.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir, exist_ok=True)

        ret = await self.infer_engine.save_engine_ckpt(save_ckpt_dir)
        log(f"saved engine ckpt to {save_ckpt_dir}", rank=0)
        return {"save_ckpt": ret}
