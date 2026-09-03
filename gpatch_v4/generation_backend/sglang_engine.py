# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import asyncio
import copy
import io

import numpy as np

try:
    import soundfile as sf
except ImportError:
    pass

import types
from typing import Dict, List, Optional

import sglang as sgl
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from typing_extensions import override

from gpatch_v4.generation_backend.infer_engine import InferEngine
from gpatch_v4.generation_backend.routed_experts_utils import extract_routed_experts
from gpatch_v4.utils import GenerationAborted, log


def raise_if_aborted(meta_info: dict) -> None:
    """Signal a cancelled request before its missing logprobs trip an assert."""
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        aborted = finish_reason.get("type") == "abort"
    elif isinstance(finish_reason, str):
        # Native generate uses ``{"type": "abort"}``; OpenAI-style / some
        # sglang versions emit the bare string.
        aborted = finish_reason == "abort"
    else:
        aborted = False
    if aborted:
        raise GenerationAborted(f"sglang aborted a generation request: {finish_reason}")


def cuda_graph_max_bs_args(value):
    """Return the right kwarg dict for ServerArgs across sglang versions.

    Older sglang uses ``cuda_graph_max_bs``; newer versions (>=0.5.5) split it
    into ``cuda_graph_max_bs_decode`` and ``cuda_graph_max_bs_prefill``.
    """
    import dataclasses
    fields = {f.name for f in dataclasses.fields(sgl.ServerArgs)}
    if "cuda_graph_max_bs_decode" in fields:
        return {"cuda_graph_max_bs_decode": value, "cuda_graph_max_bs_prefill": value}
    return {"cuda_graph_max_bs": value}


def filter_server_args_kwargs(kwargs: Dict) -> Dict:
    """Drop keys that the installed ``sgl.ServerArgs`` does not accept."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(sgl.ServerArgs)}
    return {k: v for k, v in kwargs.items() if k in fields}


def save_sharded_model_kwargs(path, pattern=None, max_size=None):
    """Return kwargs for ``Engine.save_sharded_model`` across sglang versions.

    - sglang (e.g. 0.5.14): RPC target takes ``path`` / ``pattern`` /
      ``max_size`` directly, so pass them as top-level kwargs.
    - sglang (e.g. 0.5.10 with ``SchedulerUpdateWeightsMixin``): target is
      ``save_sharded_model(self, params)`` and expects a single ``params`` dict.
    """
    import inspect

    save_args = {"path": path, "pattern": pattern, "max_size": max_size}
    try:
        from sglang.srt.managers.scheduler_update_weights_mixin import (
            SchedulerUpdateWeightsMixin,
        )
        params = inspect.signature(SchedulerUpdateWeightsMixin.save_sharded_model).parameters
        if "params" in params and "path" not in params:
            return {"params": save_args}
    except Exception:
        pass
    return save_args


# sglang engine 这里有一些 TODO：
# 1. 输入从 prompt_ids 换成 text
# 2. 减少支持的版本，比如 0.4.6.post5 直接不支持了
# 3. 0.5.2 版本的 sglang 可以不设置 skip_tokenizer_init 并且返回 text 和 ouput_ids


class SglangEngine(InferEngine):
    """Inference engine wrapper for sglang backend.

    Parameters
    ----------
    infer_engine : object
    model_path : str or list[str]
    infer_engine_role : str or None
    placement_type : str
        Placement mode for lifecycle behavior.
    """

    SEED_ATTR = "sampling_seed"

    def __init__(self, infer_engine, model_path, infer_engine_role, placement_type):
        super().__init__(infer_engine, model_path, infer_engine_role, placement_type)
        self.sglang_version = sgl.__version__
        # tokenizer_manager.get_loads() is not safe to call concurrently; all
        # callers must go through this lock.
        self.get_load_lock = asyncio.Lock()

    @override
    def shutdown(self) -> None:
        """Shutdown sglang engine through its own lifecycle hook."""
        if self.infer_engine is None:
            return
        self.infer_engine.shutdown()
        self.infer_engine = None

    @override
    def get_sampling_params(
        self,
        n=1,
        temperature=1.,
        top_k=-1,
        top_p=1.,
        max_tokens=128,
        stop_token_ids=None,
        seed=None,
        repetition_penalty=1.0,
        logit_bias=None,
        penalty_token_ids=None,
        custom_logit_processor=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        min_p=0.0,
    ):
        assert n == 1
        extra_args = {}
        if logit_bias is not None:
            extra_args["logit_bias"] = logit_bias
        if penalty_token_ids is not None:
            extra_args["penalty_token_ids"] = penalty_token_ids
        if custom_logit_processor is not None:
            extra_args["custom_logit_processor"] = custom_logit_processor
        return types.SimpleNamespace(
            n=n,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            sampling_seed=seed,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            min_p=min_p,
            **extra_args,
        )

    @override
    def async_generate(self, inp, sampling_params, request_id: str, return_routed_experts=False):
        """Submit an asynchronous generation request.

        Parameters
        ----------
        inp : dict
            Input dict with ``'prompt_token_ids'`` and optional ``'multi_modal_data'``.
        sampling_params : object or list
        request_id : str
        return_routed_experts : bool, optional

        Returns
        -------
        coroutine
            Async generation output.
        """
        #TODO: 将 prompt 直接用 text 输入
        assert self.infer_engine != None, "Not initialized InferEngine class"

        sampling_params_lst = sampling_params
        if not isinstance(sampling_params, list):
            sampling_params_lst = [sampling_params]
        custom_logit_processor_l = []
        for i, x in enumerate(sampling_params_lst):
            sampling_params_lst[i] = x.__dict__

            if "max_new_tokens" in sampling_params_lst[i]:
                # sglang 有一个限制:
                # input_prompt_lens + max_new_tokens < max position embedding
                sampling_params_lst[i]["max_new_tokens"] -= 1
            if 'penalty_token_ids' in sampling_params_lst[i]:
                sampling_params_lst[i]['custom_params'] = \
                    {"token_id": copy.deepcopy(sampling_params_lst[i]['penalty_token_ids'])}
                sampling_params_lst[i].pop('penalty_token_ids')

            if 'custom_logit_processor' in sampling_params_lst[i]:
                custom_logit_processor_l.append(
                    sampling_params_lst[i]['custom_logit_processor'].to_str()
                )
                sampling_params_lst[i].pop('custom_logit_processor')
        if len(sampling_params_lst) == 1:
            sampling_params_lst = sampling_params_lst[0]

        input_kwargs = dict(
            sampling_params=sampling_params_lst,
            input_ids=inp['prompt_token_ids'],
            return_logprob=True,
        )
        if "multi_modal_data" in inp:
            if "image" in inp["multi_modal_data"]:
                # list of image pil
                input_kwargs["image_data"] = inp["multi_modal_data"]["image"]
            if "audio" in inp["multi_modal_data"]:
                audio_list = []
                for item in inp["multi_modal_data"]["audio"]:
                    if isinstance(item, tuple):
                        arr, sr = item
                        buf = io.BytesIO()
                        sf.write(buf, np.asarray(arr), sr, format="WAV", subtype="FLOAT")
                        audio_list.append(buf.getvalue())
                    else:
                        audio_list.append(item)
                input_kwargs["audio_data"] = audio_list
        if isinstance(custom_logit_processor_l, str) or len(custom_logit_processor_l) > 0:
            input_kwargs["custom_logit_processor"] = custom_logit_processor_l

        if return_routed_experts:
            input_kwargs["return_routed_experts"] = True

        async_output = self.infer_engine.async_generate(**input_kwargs)
        return async_output

    @override
    async def wait_and_get_async_generate_output(self, async_generators):
        """Await all async generation outputs and parse them.

        Parameters
        ----------
        async_generators : list

        Returns
        -------
        list
            Parsed generation outputs with token IDs, logprobs, etc.
        """
        fns = []
        for gi, gen in enumerate(async_generators):
            fns.append(gen)
        async_outputs = await asyncio.gather(*fns)

        outputs = []
        for async_out in async_outputs:
            if isinstance(async_out, list):  # n>1
                rep_outs = []
                for rep_out in async_out:
                    meta = rep_out['meta_info']
                    raise_if_aborted(meta)
                    assert 'output_token_logprobs' in meta, \
                        f"Missing key: output_token_logprobs, meta_info keys = {meta.keys()}"
                    rep_outs.append(
                        types.SimpleNamespace(
                            token_ids=rep_out['output_ids'],
                            prompt_len=meta['prompt_tokens'],
                            routed_experts=extract_routed_experts(rep_out),
                            output_logprobs=[it[0] for it in meta['output_token_logprobs']],
                            # sglang 特有，后面最好干掉它，要用什么直接加上
                            meta_info=meta,
                        )
                    )
            else:
                meta = async_out['meta_info']
                raise_if_aborted(meta)
                assert 'output_token_logprobs' in meta, \
                    f"Missing key: output_token_logprobs, meta_info keys = {meta.keys()}"
                rep_outs = [
                    types.SimpleNamespace(
                        token_ids=async_out['output_ids'],
                        prompt_len=async_out['meta_info']['prompt_tokens'],
                        routed_experts=extract_routed_experts(async_out),
                        output_logprobs=[it[0] for it in meta['output_token_logprobs']],
                        # sglang 特有，后面最好干掉它，要用什么直接加上
                        meta_info=meta,
                    )
                ]
            output = types.SimpleNamespace(outputs=rep_outs)
            outputs.append(output)
        return outputs

    @override
    async def flush_cache(self):
        """Flush the KV cache of the sglang engine."""
        await self.infer_engine.tokenizer_manager.flush_cache()

    @override
    async def abort_all_requests(self):
        """Abort every in-flight request on this sglang engine."""
        log("[SglangEngine] abort_all_requests called", rank=0)
        ret = self.infer_engine.tokenizer_manager.abort_request(abort_all=True)
        if asyncio.iscoroutine(ret):
            ret = await ret
        return ret

    @override
    async def get_load(self) -> Dict[str, int]:
        tm = self.infer_engine.tokenizer_manager
        # sglang 0.5.14 renamed TokenizerManager.get_load -> get_loads and
        # switched the payload to LoadSnapshot (num_running/waiting/used_tokens).
        async with self.get_load_lock:
            if hasattr(tm, "get_loads"):
                loads = await tm.get_loads()
            else:
                loads = await tm.get_load()
        if not loads:
            # sglang's IPC communicator can return None / [] when the scheduler
            # is saturated; treat as "busy" so the caller routes elsewhere.
            log(
                "[WARN] tokenizer_manager.get_loads() returned empty, "
                "reporting max load so caller avoids this cluster"
            )
            return {
                "num_reqs": 999,
                "num_tokens": 999999,
                "num_running_reqs": 0,
                "num_waiting_reqs": 999,
            }
        ld = loads[0]
        if hasattr(ld, "num_running_reqs"):
            num_running_reqs = ld.num_running_reqs
            num_waiting_reqs = ld.num_waiting_reqs
            num_reqs = num_running_reqs + num_waiting_reqs
            num_tokens = ld.num_used_tokens
        else:
            # Pre-0.5.14 GetLoad schema.
            num_reqs = ld.num_reqs
            num_waiting_reqs = ld.num_waiting_reqs
            num_running_reqs = num_reqs - num_waiting_reqs
            num_tokens = ld.num_tokens
        return {
            "num_reqs": num_reqs,
            "num_tokens": num_tokens,
            "num_running_reqs": num_running_reqs,
            "num_waiting_reqs": num_waiting_reqs,
        }

    @override
    async def wake_up(self, *args, **kwargs):
        """Resume memory occupation for specified tag groups."""
        assert self.placement_type != "disaggregated"
        want_to_wake_up_tags: List[str] = kwargs.get("tags", self.all_supported_tags)

        # check if the tags are already awake
        should_wake_up_tags: List[str] = [
            tag for tag in want_to_wake_up_tags if not self.wake_up_tag[tag]
        ]
        if len(should_wake_up_tags) == 0:
            return

        kwargs["tags"] = should_wake_up_tags
        for tag in should_wake_up_tags:
            self.wake_up_tag[tag] = True

        obj = ResumeMemoryOccupationReqInput(*args, **kwargs)

        if self.infer_engine_role in ["gen-rm", "off-policy-sampler"]:
            await self.infer_engine.tokenizer_manager.resume_memory_occupation(obj, None)

            # 设置 enable_weights_cpu_backup 之后，这个应该就不需要用了
            # model_path = self.model_path
            # obj = UpdateWeightFromDiskReqInput(model_path=model_path)
            # # 针对 gen_rm，这里 wake_up 起来后权重要重新读取
            # await self.infer_engine.tokenizer_manager.update_weights_from_disk(obj, None)
        else:
            await self.infer_engine.tokenizer_manager.resume_memory_occupation(obj, None)

    async def _wait_for_idle(self, timeout_s: float = 30):
        """Flush cache and wait for the scheduler to become idle.
        """
        await self.infer_engine.tokenizer_manager.flush_cache(timeout_s=timeout_s)

    @override
    async def sleep(self, *args, **kwargs):
        """Release memory occupation for specified tag groups."""
        assert self.placement_type != "disaggregated"
        want_to_sleep_tags = kwargs.get("tags", self.all_supported_tags)

        # check if the tags are already asleep
        should_sleep_tags: List[str] = [tag for tag in want_to_sleep_tags if self.wake_up_tag[tag]]
        if len(should_sleep_tags) == 0:
            return

        kwargs["tags"] = should_sleep_tags
        for tag in should_sleep_tags:
            self.wake_up_tag[tag] = False

        await self._wait_for_idle()
        obj = ReleaseMemoryOccupationReqInput(*args, **kwargs)
        await self.infer_engine.tokenizer_manager.release_memory_occupation(obj, None)

    async def release_kv_cache_for_weight_update(self):
        """Release KV cache only before distributed weight update."""
        if not self.wake_up_tag["kv_cache"]:
            log("release_kv_cache_for_weight_update: kv_cache already released", rank=0)
            return
        self.wake_up_tag["kv_cache"] = False
        await self._wait_for_idle()
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        await self.infer_engine.tokenizer_manager.release_memory_occupation(obj, None)
        log("SglangEngine release_kv_cache_for_weight_update done", rank=0)

    async def resume_kv_cache_after_weight_update(self):
        """Resume KV cache after distributed weight update."""
        if self.wake_up_tag["kv_cache"]:
            log("resume_kv_cache_after_weight_update: kv_cache already active", rank=0)
            return
        self.wake_up_tag["kv_cache"] = True
        obj = ResumeMemoryOccupationReqInput(tags=["kv_cache"])
        await self.infer_engine.tokenizer_manager.resume_memory_occupation(obj, None)
        log("SglangEngine resume_kv_cache_after_weight_update done", rank=0)

    @override
    def update_engine_weight_by_model_idx(self, rm_model_idx):
        """Switch to a different reward model's weights.

        Parameters
        ----------
        rm_model_idx : int
            Index into ``self.model_path`` list.

        Returns
        -------
        coroutine
            Async update result.
        """
        assert isinstance(self.model_path, list) and len(self.model_path) > rm_model_idx

        async def update_rm_idx_fn():
            obj = UpdateWeightFromDiskReqInput(model_path=self.model_path[rm_model_idx])
            return await self.infer_engine.tokenizer_manager.update_weights_from_disk(obj, None)

        ret = update_rm_idx_fn()
        self.model_index = rm_model_idx
        return ret

    @override
    async def update_weights(self, serialized_named_tensors, load_format, **kwargs):
        """Update engine weights from serialized tensors.

        Parameters
        ----------
        serialized_named_tensors : list
        load_format : str

        Returns
        -------
        object
            Update result from sglang.
        """
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=False,
        )
        ret = await self.infer_engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        return ret

    async def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Create an NCCL process group for receiving distributed weight updates."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        return await self.infer_engine.tokenizer_manager.init_weights_update_group(obj, None)

    async def update_weights_from_distributed(
        self,
        update_info: dict,
    ):
        """Receive weights via NCCL broadcast from the training rank."""
        obj = UpdateWeightsFromDistributedReqInput(**update_info)
        ret = await self.infer_engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        return ret

    async def destroy_weights_update_group(self, group_name):
        """Destroy the NCCL process group created by ``init_weights_update_group``."""
        obj = DestroyWeightsUpdateGroupReqInput(group_name=group_name)
        return await self.infer_engine.tokenizer_manager.destroy_weights_update_group(obj, None)

    @override
    async def save_engine_ckpt(self, save_ckpt_path):
        """Save the engine checkpoint to disk.

        Parameters
        ----------
        save_ckpt_path : str

        Returns
        -------
        object
            Save result.
        """
        ret = self.infer_engine.save_sharded_model(
            **save_sharded_model_kwargs(
                path=save_ckpt_path,
                pattern=None,
                max_size=(16 * 1024**3),
            ),
        )
        return ret
