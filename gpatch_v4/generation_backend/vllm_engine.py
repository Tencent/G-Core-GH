# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import asyncio
import os
import types
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import vllm
from typing_extensions import override

from gpatch_v4.generation_backend.infer_engine import InferEngine
from gpatch_v4.utils import gcore_save_vllm_checkpoint, log


def merge_vllm_routed_experts(gen_output, completion_output):
    """Merge vLLM v0.21+ split routing back to a full-sequence array.

    vLLM puts prompt routing on ``RequestOutput.prompt_routed_experts`` and
    generation routing on ``CompletionOutput.routed_experts``.  gcore's
    ``process_routed_experts`` expects the concatenated full sequence.
    """
    prompt_re = getattr(gen_output, "prompt_routed_experts", None)
    gen_re = completion_output.routed_experts
    if prompt_re is None and gen_re is None:
        return None
    if prompt_re is not None and gen_re is not None:
        return np.concatenate([prompt_re, gen_re], axis=0)
    return gen_re if gen_re is not None else prompt_re


class VllmEngine(InferEngine):
    """Inference engine wrapper for vLLM backend.

    Parameters
    ----------
    infer_engine : object
    model_path : str
    infer_engine_role : str or None
    placement_type : str
        Placement mode for lifecycle behavior.
    """

    SEED_ATTR = "seed"

    def __init__(self, infer_engine, model_path, infer_engine_role, placement_type):
        super().__init__(infer_engine, model_path, infer_engine_role, placement_type)
        self.patch_output_processor_scheduler_stats_cache()
        self._stats_task: Optional[asyncio.Task] = None
        if placement_type == "disaggregated":
            self.start_stats_logging()

    def patch_output_processor_scheduler_stats_cache(self) -> None:
        """Cache vLLM scheduler stats observed by the frontend output loop.

        ``OutputProcessor.update_scheduler_stats`` is the only place where the
        frontend sees live scheduler snapshots; vLLM does not expose them
        afterwards.  We wrap the method so the last non-``None`` snapshot is
        kept on ``output_processor._gcore_last_scheduler_stats`` and consumed
        by :meth:`get_load`.
        """
        output_processor = self.infer_engine.output_processor
        if hasattr(output_processor, "_gcore_original_update_scheduler_stats"):
            return

        original = output_processor.update_scheduler_stats
        output_processor._gcore_original_update_scheduler_stats = original

        def update_scheduler_stats_with_cache(scheduler_stats: Any) -> Any:
            if scheduler_stats is None:
                if hasattr(output_processor, "_gcore_last_scheduler_stats"):
                    delattr(output_processor, "_gcore_last_scheduler_stats")
            else:
                output_processor._gcore_last_scheduler_stats = scheduler_stats
            return original(scheduler_stats)

        output_processor.update_scheduler_stats = update_scheduler_stats_with_cache

    @override
    async def get_load(self) -> Dict[str, int]:
        output_processor = self.infer_engine.output_processor
        frontend_num_reqs = output_processor.get_num_unfinished_requests()
        scheduler_stats = getattr(output_processor, "_gcore_last_scheduler_stats", None)

        if scheduler_stats is None:
            # No scheduler snapshot yet (engine still warming up): approximate
            # running = min(frontend_num_reqs, max_running), rest is waiting.
            max_running_reqs = self.infer_engine.vllm_config.scheduler_config.max_num_seqs
            num_running_reqs = min(frontend_num_reqs, max_running_reqs)
            num_waiting_reqs = frontend_num_reqs - num_running_reqs
            return {
                "num_reqs": frontend_num_reqs,
                "num_running_reqs": num_running_reqs,
                "num_waiting_reqs": num_waiting_reqs,
            }

        num_running_reqs = scheduler_stats.num_running_reqs
        num_waiting_reqs = (
            scheduler_stats.num_waiting_reqs +
            getattr(scheduler_stats, "num_skipped_waiting_reqs", 0)
        )
        scheduler_num_reqs = num_running_reqs + num_waiting_reqs
        # Requests that the frontend accepted but that have not yet reached
        # the scheduler: count them as waiting so they are visible to the
        # routing decision.
        num_waiting_reqs += max(0, frontend_num_reqs - scheduler_num_reqs)
        return {
            "num_reqs": max(frontend_num_reqs, num_running_reqs + num_waiting_reqs),
            "num_running_reqs": num_running_reqs,
            "num_waiting_reqs": num_waiting_reqs,
        }

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
        assert n == 1  # vllm async llm bug
        return vllm.SamplingParams(
            n=n,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            seed=seed,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            min_p=min_p,
            logit_bias=logit_bias,
            logprobs=1,
        )

    @override
    def async_generate(self, inp, sampling_params, request_id: str, return_routed_experts=False):
        """Submit an async generation request to vLLM.

        Parameters
        ----------
        inp : dict
        sampling_params : vllm.SamplingParams
        request_id : str
        return_routed_experts : bool, optional
            Not used by vLLM, by default *False*.

        Returns
        -------
        async generator
            Async generation stream.
        """
        assert self.infer_engine is not None, "Not initialized InferEngine class"
        async_gen = self.infer_engine.generate(inp, sampling_params, request_id)
        return async_gen

    @override
    async def wait_and_get_async_generate_output(self, async_generators):
        """Await all async vLLM generation streams and collect outputs.

        Parameters
        ----------
        async_generators : list

        Returns
        -------
        list
            Parsed generation outputs with token IDs, logprobs, etc.
            Format matches :class:`SglangEngine` output.
        """
        fns = []
        for gi, gen in enumerate(async_generators):

            async def fn(_gen):
                output = None
                async for _output in _gen:
                    output = _output
                assert output is not None, "vllm generate output is None"
                return output

            fns.append(fn(gen))
        gen_outputs = await asyncio.gather(*fns)

        outputs = []
        for gen_output in gen_outputs:
            if gen_output is None:
                outputs.append(None)
                continue
            rep_outs = []
            for completion_output in gen_output.outputs:
                assert completion_output.logprobs is not None, \
                    "Missing logprobs in CompletionOutput"
                output_logprobs = [
                    logprob_dict[token_id].logprob for token_id, logprob_dict in
                    zip(completion_output.token_ids, completion_output.logprobs)
                ]
                rep_outs.append(
                    types.SimpleNamespace(
                        token_ids=completion_output.token_ids,
                        prompt_len=len(gen_output.prompt_token_ids),
                        routed_experts=merge_vllm_routed_experts(gen_output, completion_output),
                        output_logprobs=output_logprobs,
                        text=completion_output.text,
                        # vllm 特有，sglang 后面如果有最好加上，对多模态验证比较有用
                        prompt_token_ids=gen_output.prompt_token_ids,
                    )
                )
            outputs.append(types.SimpleNamespace(outputs=rep_outs))
        return outputs

    @override
    async def wake_up(self, *args, **kwargs):
        """Resume vLLM engine memory for specified tag groups.

        Uses vLLM's CuMemAllocator-backed sleep mode to selectively restore
        weights and/or KV cache memory on GPU.
        """
        want_tags: List[str] = kwargs.get("tags", self.all_supported_tags)
        should_wake = [t for t in want_tags if not self.wake_up_tag[t]]
        if not should_wake:
            return

        self.model_index = 0
        await self.infer_engine.wake_up(tags=should_wake)
        await self.infer_engine.collective_rpc("gcore_restore_moe_after_wakeup")
        for t in should_wake:
            self.wake_up_tag[t] = True
        log(f"VllmEngine wake_up tags={should_wake}", rank=0)
        # Start stats logging only when kv_cache is woken (= about to generate),
        # not during weights-only wake_up (= update_weights phase).
        if "kv_cache" in should_wake:
            self.start_stats_logging()

    @override
    async def sleep(self, *args, **kwargs):
        """Release vLLM engine memory via CuMemAllocator sleep.

        Uses level=1 so that weight tensors are offloaded to CPU (backed up)
        before GPU memory is freed.  This is critical because IPC-based weight
        updates write directly to GPU memory; level=2 would discard those
        updated weights without backup, causing all-zero weights on the next
        ``wake_up``.
        """
        self.stop_stats_logging()
        await self.infer_engine.do_log_stats()

        want_tags = kwargs.get("tags", self.all_supported_tags)
        should_sleep = [t for t in want_tags if self.wake_up_tag[t]]
        if not should_sleep:
            return

        # vllm 这里如果有一些 tag 在睡，一些 tag 在醒，vllm 就默认睡了，不会再将醒的 tag 也睡下去
        # 所以在这种情况下，让所有 tag 都醒，再睡下去，这样才能释放全部的显存
        awake_tags = [t for t in self.all_supported_tags if self.wake_up_tag[t]]
        sleeping_tags = [t for t in self.all_supported_tags if not self.wake_up_tag[t]]

        if awake_tags and sleeping_tags:
            await self.infer_engine.wake_up(tags=sleeping_tags)
            for t in sleeping_tags:
                self.wake_up_tag[t] = True
            should_sleep = [t for t in want_tags if self.wake_up_tag[t]]

        await self.infer_engine.collective_rpc("gcore_save_moe_for_sleep")
        await self.infer_engine.sleep(level=1)
        # EngineCore._reset_caches() clears mm_receiver_cache during sleep,
        # but the renderer-side mm hash cache is left stale. Clear both sides
        # so the next generation re-sends full multimodal features.
        await self.infer_engine.reset_mm_cache()
        for t in should_sleep:
            self.wake_up_tag[t] = False
        log(f"VllmEngine sleep tags={should_sleep}", rank=0)

    def start_stats_logging(self) -> None:
        """Start periodic throughput logging while the engine is awake."""
        if self._stats_task is not None:
            return

        async def _periodic_log_stats():
            try:
                interval = int(os.getenv("VLLM_LOG_STATS_INTERVAL", "5"))
                while True:
                    await asyncio.sleep(interval)
                    if self.infer_engine.output_processor.get_num_unfinished_requests() > 0:
                        await self.infer_engine.do_log_stats()
            except asyncio.CancelledError:
                pass

        self._stats_task = asyncio.create_task(_periodic_log_stats())

    def stop_stats_logging(self) -> None:
        """Stop the periodic throughput logging task."""
        if self._stats_task is not None:
            self._stats_task.cancel()
            self._stats_task = None

    @override
    async def flush_cache(self):
        """Flush the KV cache of the vLLM engine."""
        await self.infer_engine.reset_prefix_cache()

    async def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize vLLM's weight transfer engine.

        For colocated placement, the IPC backend needs no init info.
        For disaggregated placement, the NCCL backend requires address/port/rank
        so each TP worker can join the same NCCL group as the training rank.
        """
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
        }
        await self.infer_engine.collective_rpc(
            "init_weight_transfer_engine",
            args=(init_info, ),
        )
        self._weight_transfer_group_name = group_name
        log(f"VllmEngine init_weights_update_group done (nccl)", rank=0)

    @override
    async def update_weights(self, **update_info):
        """Receive weights via vLLM's ``IPCWeightTransferEngine`` (colocated).

        Called by the sampler actor when the client sent per-GPU CUDA IPC
        handles produced by ``torch.multiprocessing.reductions.reduce_tensor``.
        The worker-side ``IPCWeightTransferEngine.receive_weights`` looks up
        each tensor's handle by physical GPU UUID and rebuilds it via
        ``rebuild_cuda_tensor``.

        We dispatch to ``gcore_update_weights_ipc`` (injected onto every vLLM
        worker via ``worker_extension_cls``) instead of the built-in
        ``update_weights``. The extension skips vLLM's per-bucket
        ``initialize/finalize_layerwise_reload`` and instead runs
        ``process_weights_after_loading`` once, driven by the
        ``is_last_bucket`` flag in ``update_info`` emitted by
        :meth:`_update_weights_by_ipc_handle_vllm._flush`. This avoids the
        MoE ``w2_weight`` corruption observed on Qwen3-30B-MoE when a layer's
        experts straddle multiple IPC buckets.

        Parameters
        ----------
        **update_info
            Keyword arguments repacked by
            :meth:`GrpoSamplerActor.update_weights`. Expected keys:
            ``names``, ``dtype_names``, ``shapes``, ``ipc_handles``,
            ``is_checkpoint_format``, ``is_last_bucket``.
        """
        await self.infer_engine.collective_rpc(
            "gcore_update_weights_ipc",
            args=(update_info, ),
        )
        if self.wake_up_tag.get("kv_cache", False):
            await self.infer_engine.reset_prefix_cache()
        log("VllmEngine update_weights (ipc) done", rank=0)

    async def update_weights_bucketed(self, update_info: dict):
        """Deliver one flat-IPC bucket to every colocated vLLM TP worker.

        Dispatches :meth:`GCoreVllmWorkerExtension.gcore_update_weights_bucketed`
        via ``collective_rpc``. ``update_info`` already carries the per-GPU
        CUDA IPC handle dict for the flat bucket -- each worker picks out
        the handle matching its own physical GPU UUID.
        """
        await self.infer_engine.collective_rpc(
            "gcore_update_weights_bucketed",
            args=(update_info, ),
        )

    async def update_weights_from_distributed(self, update_info):
        """Receive one flat NCCL-broadcast weight bucket.

        Called by the sampler actor when the client drives a bucketed NCCL
        broadcast via
        :meth:`VllmUpdateWeightFactory._broadcast_vllm_bucket`.
        Each bucket is one ``uint8`` flat tensor that is split into per-param
        views on the worker side and handed to ``model.load_weights``.
        Finalization (``process_weights_after_loading``) is done once, in
        :meth:`finalize_weights_update`, after all buckets are acked.

        Parameters
        ----------
        update_info : dict
            Expected keys: ``names``, ``dtype_names``, ``shapes``,
            ``total_bytes``, ``group_name``.
        """
        await self.infer_engine.collective_rpc(
            "gcore_update_weights_distributed",
            args=(update_info, ),
        )
        log("VllmEngine update_weights_from_distributed (nccl) done", rank=0)

    async def finalize_weights_update(self):
        """Run worker finalize (MegaMoE + verl-style post-process) once.

        Transport-agnostic finalize. Called once per full weight update,
        after all buckets have been acked -- serves both the bucketed-IPC
        path (:meth:`update_weights_bucketed`) and the NCCL-distributed
        path (:meth:`update_weights_from_distributed`). Also flushes the
        KV prefix cache when KV memory is live, matching
        :meth:`update_weights`.
        """
        await self.infer_engine.collective_rpc("gcore_finalize_weights_update")
        if self.wake_up_tag.get("kv_cache", False):
            await self.infer_engine.reset_prefix_cache()
        log("VllmEngine finalize_weights_update done", rank=0)

    async def start_weights_update(self):
        """Prepare vLLM workers to receive checkpoint-format weights.

        Must be called once before the first bucket of a multi-bucket
        update. Pairs with :meth:`finalize_weights_update`.
        """
        await self.infer_engine.collective_rpc("gcore_start_weights_update")
        log("VllmEngine start_weights_update done", rank=0)

    async def update_weights_from_file(self, weight_file: str):
        """Reload checkpoint-format weights from a safetensors file.

        This avoids CUDA IPC entirely, sidestepping memory pinning issues
        that occur when ``reduce_tensor()`` is used with vLLM's CuMemAllocator.

        vLLM's ``reload_weights`` expects a directory containing safetensors
        files, so we pass the parent directory of the weight file.
        """
        import os

        weights_dir = os.path.dirname(weight_file)
        await self.infer_engine.collective_rpc(
            "reload_weights",
            kwargs={
                "weights_path": weights_dir,
                "is_checkpoint_format": True
            },
        )
        if self.wake_up_tag.get("kv_cache", False):
            await self.infer_engine.reset_prefix_cache()
        log(f"VllmEngine update_weights_from_file done ({weight_file})", rank=0)

    async def destroy_weights_update_group(self, group_name):
        """Clean up weight transfer resources."""
        self._weight_transfer_group_name = None
        log("VllmEngine destroy_weights_update_group done", rank=0)

    @override
    def update_engine_weight_by_model_idx(self, rm_model_idx):
        raise NotImplementedError("Vllm engine has not impl update_engine_weight_by_model_idx")

    @override
    async def save_engine_ckpt(self, save_ckpt_path):
        """Dump each TP worker's current model ``state_dict`` as safetensors.

        Used by the ``debug_update_weight`` harness to snapshot engine
        weights at three points (src / zero / real) for offline diff.
        We avoid depending on any specific vLLM checkpoint writer by
        going through ``collective_rpc`` and letting each worker write
        its own shard keyed by rank. Failures are logged but non-fatal
        so the smoke test can still proceed even if the snapshot cannot
        be materialized (the real correctness signal comes from
        ``test_generate`` before vs. after zero-replace/real-replace).
        """
        import os
        try:
            os.makedirs(save_ckpt_path, exist_ok=True)
        except Exception as exc:
            log(
                f"VllmEngine save_engine_ckpt mkdir failed path={save_ckpt_path} err={exc}",
                rank=0,
            )
            return {"ret": False, "error": str(exc)}

        ret = await self.infer_engine.collective_rpc(
            gcore_save_vllm_checkpoint,
            args=(save_ckpt_path, ),
        )
        log(f"VllmEngine save_engine_ckpt done path={save_ckpt_path}", rank=0)
        return {"ret": True, "per_rank": ret}
