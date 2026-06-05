import inspect
import os
import uuid
from typing import Any, Dict, List

import PIL
import ray
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override

from gpatch_v4.actor.mixin import T2iTokenizerMixin
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.parallel_state import cpu_barrier, init_pg, initlize_parallel_state
from gpatch_v4.generation_backend import InferEngine
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.orches.utils import ray_noset_visible_devices
from gpatch_v4.patch.sglang_patch import sglang_hack
from gpatch_v4.utils import (
    import_fn_from_path,
    kill_process_tree,
    log,
    logging_rank0,
    unbind_tensor_to_list,
)


def find_images_recursively(d):
    l = []
    if isinstance(d, PIL.Image.Image):
        return [d]
    elif isinstance(d, list) or isinstance(d, tuple):
        for x in d:
            l.extend(find_images_recursively(x))
    elif isinstance(d, dict):
        for x in d.values():
            l.extend(find_images_recursively(x))
    return l


class T2iGrpoGenRmActor(BaseActor, T2iTokenizerMixin):
    '''
    关于 ``prompt_fn``：
    messages 是 huggingface 的**标准**对话格式：

    ```python
    messages1 = [
        {
            "role":
                "user",
            "content":
                [
                    { "type": "image", "image": "file:///path/to/image1.jpg" },
                    { "type": "image", "image": "file:///path/to/image2.jpg" },
                    { "type": "text", "text": "What are the common elements in these pictures?" },
                ],
        }
    ]
    messages2 = [
        {
            "role": "system",
            "content": [{ "type": "text", "text": "You are a helpful assistant." }]
        },
        {
            "role": "user",
            "content": [{ "type": "text", "text": "Who are you?" }]
        },
    ]
    messages = [
        messages1,
        messages2,
    ]
    ```

    Docs:
      1. https://huggingface.co/docs/transformers/main/en/chat_templating#addgenerationprompt
      2. https://github.com/QwenLM/Qwen3-VL
    '''
    @override
    def shutdown(self) -> None:
        """Reap sglang subprocess tree owned by this actor's infer engine.

        See ``GrpoSamplerActor.shutdown`` for the rationale.
        """
        kill_process_tree(os.getpid(), include_parent=False)

    async def init(self, config):
        super().init(config, pg_backend='nccl')  # TODO make it gloo to save gpu mem
        fake_dist_config = DistConfig()
        initlize_parallel_state(config, fake_dist_config)
        init_pg(fake_dist_config)
        self._is_master_node = None

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
        if config.gen_rm.backend == "vllm":
            import os
            os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        else:
            sglang_hack()
        self._is_master_node = is_master_node
        self.rm_idx = rm_idx
        g_rank = torch.distributed.get_rank()
        infer_engine_config = config.gen_rm.infer_engine_configs[rm_idx]
        dist_config = infer_engine_config.dist_config
        self.model_arch = config.gen_rm.reward_model_info[rm_idx].model_arch
        hf_model_path = config.gen_rm.reward_model_info[rm_idx].hf_model_path

        base_gpu_id = None
        if ray_noset_visible_devices():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                base_gpu_id = int(gpu_ids[0])

        log(
            f'T2iGrpoGenRmActor.init create infer engine {rm_idx=} {g_rank=} {engine_idx=} {tp_rank=} {base_gpu_id=}'
        )
        self.infer_engine = InferEngine.from_engine_args(
            config.gen_rm.backend,
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
            load_format='auto',
            log_level='info',
            allow_auto_truncate=infer_engine_config.allow_auto_truncate,
            enable_custom_logit_processor=False,
            attention_backend=None,
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
        )
        self.setup_tokenizer(hf_model_path)
        self.setup_processor(hf_model_path)
        self.post_init(self.rm_idx)

    def post_init(self, rm_idx):
        reward_model_info = self.config.gen_rm.reward_model_info[rm_idx]
        self.prompt_fn = import_fn_from_path(
            reward_model_info.reward_py_path, reward_model_info.prompt_fn_name
        )
        fn_kwargs = inspect.signature(self.prompt_fn).parameters
        cond1 = all([
            len(fn_kwargs) == 1,
            'batched_data' in fn_kwargs,
        ])
        assert cond1, f"unexpected {cond1}"

        self.parse_reward_fn = import_fn_from_path(
            reward_model_info.reward_py_path, reward_model_info.parse_reward_fn_name
        )
        fn_kwargs = inspect.signature(self.parse_reward_fn).parameters
        cond1 = all([
            len(fn_kwargs) == 1,
            'resp_texts' in fn_kwargs,
        ])
        assert cond1, f"unexpected {cond1}"

    async def sleep(self, req_dict):
        assert self.config.placement_type != "disaggregated"
        if not self._is_master_node:
            return {"ret": True}
        await self.infer_engine.sleep()
        return {"ret": True}

    async def wake_up(self, req_dict):
        assert self.config.placement_type != "disaggregated"
        if not self._is_master_node:
            return {"ret": True}
        tags = req_dict["tag_names"]
        await self.infer_engine.wake_up(tags=tags)
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        if self.config.placement_type != "disaggregated":
            tags = req_dict["tag_names"]
            await self.infer_engine.wake_up(tags=tags)
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict):
        if not self._is_master_node:
            return {"ret": True}
        if self.config.placement_type != "disaggregated":
            await self.infer_engine.sleep()
        return {"ret": True}

    async def generate_rewards(self, req_dict: Dict[str, Any]):
        batched_data: Dict[str, List[Any]] = req_dict["batched_data"]
        repeat_n = self.config.training.sampling_repeat_n
        rollout_mbs = self.config.training.rollout_mbs
        for k, v in batched_data.items():
            assert isinstance(v, list) and len(
                v
            ) == rollout_mbs * repeat_n, f'unexpected {k=} {v=} {rollout_mbs=} {repeat_n=} {len(v)=}'

        sampling_params = self.infer_engine.get_sampling_params_from_config(
            self.config.gen_rm.infer_engine_configs[self.rm_idx], self.tokenizer.eos_token_id
        )
        chats = self.prompt_fn(batched_data=batched_data)
        prompts = self.processor.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer(prompts)["input_ids"]

        async_gens = []
        for i in range(rollout_mbs * repeat_n):
            chat_i = chats[i]
            images = find_images_recursively(chat_i)
            vlm_inputs = {
                'prompt_token_ids': token_ids[i],
                'multi_modal_data': {
                    "image": images
                },
            }
            tmp_sampling_params = self.infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i
            )
            gen = self.infer_engine.async_generate(
                vlm_inputs, tmp_sampling_params, str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

        gen_outputs = await self.infer_engine.wait_and_get_async_generate_output(async_gens)

        output_token_ids = []
        for gi in range(len(async_gens)):
            sample_output = gen_outputs[gi]
            assert len(sample_output.outputs) == 1
            sample_output = sample_output.outputs[0]
            output_token_ids.append(list(sample_output.token_ids))

        resp_texts = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=False)
        reward_tensors = self.parse_reward_fn(resp_texts=resp_texts).view(-1, 1)
        ret_dict = {
            f"reward_gen_rm_{self.rm_idx}": unbind_tensor_to_list(reward_tensors),
        }
        return ret_dict
