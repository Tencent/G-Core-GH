# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import asyncio
import copy
import math
import os
import sys
import types
from contextlib import nullcontext
from typing import List, Optional

import torch
from packaging import version

from gpatch_v4.configs.infer_engine_config import InferEngineConfig
from gpatch_v4.utils.common_utils import log
from gpatch_v4.utils.logging_utils import (
    configure_third_party_logging,
    redirect_stdio_fds_to_file,
)


class FakeSignal:
    # 路子还得是 hess 野，有点牛逼。

    SIGQUIT = None

    @staticmethod
    def signal(*args):
        pass


def patch_import_processors():
    """Monkey-patch sglang 0.5.4.post3 to avoid ``log_info_on_rank0`` assert.

    When ``torch.distributed`` is initialized in the driver process, sglang
    unexpectedly fails.  This patch swaps ``get_tensor_model_parallel_rank``
    temporarily during processor import.
    """

    import sys
    from contextlib import contextmanager

    def fake_get_tensor_model_parallel_rank():
        return None

    @contextmanager
    def tmp_patch_get_tensor_model_parallel_rank():
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        try:
            setattr(
                sys.modules["sglang.srt.distributed"], 'get_tensor_model_parallel_rank',
                fake_get_tensor_model_parallel_rank
            )
            yield
        except:
            pass
        finally:
            setattr(
                sys.modules["sglang.srt.distributed"], 'get_tensor_model_parallel_rank',
                get_tensor_model_parallel_rank
            )

    try:
        from sglang.srt.managers.tokenizer_manager import import_processors

        def import_processors_patch(*args, **kwargs):
            # in engine should not log_info_on_rank0
            with tmp_patch_get_tensor_model_parallel_rank():
                return import_processors(*args, **kwargs)

        setattr(
            sys.modules["sglang.srt.managers.tokenizer_manager"], 'import_processors',
            import_processors_patch
        )
    except:
        pass


class InferEngine:
    """Unified wrapper around inference engines (vLLM / sglang).

    Parameters
    ----------
    infer_engine : object
    model_path : str
    infer_engine_role : str or None
        Role identifier (``'sampler'``, ``'gen-rm'``, etc.).
    placement_type : str
        Placement mode (``'colocate'`` or ``'disaggregated'``).
    """
    def __init__(self, infer_engine, model_path, infer_engine_role, placement_type):
        self.infer_engine = infer_engine
        self.model_path = model_path
        self.infer_engine_role = infer_engine_role
        self.placement_type = placement_type
        self.wake_up_tag = {
            "weights": True,
            "kv_cache": True,
        }
        self.all_supported_tags = list(self.wake_up_tag.keys())
        log(
            f"Init InferEngine {self.infer_engine_role=}"
            f"infer_engine_type {self.__class__.__name__}"
        )

    def get_engine(self):
        """Return the underlying inference engine.

        Returns
        -------
        object
            The engine instance.

        Raises
        ------
        AssertionError
            If the engine is not initialized.
        """
        assert self.infer_engine is not None, "Not initialized InferEngine class"
        return self.infer_engine

    SEED_ATTR: str = NotImplemented  # "seed" for vLLM, "sampling_seed" for sglang

    def copy_sampling_params_with_seed_offset(self, sampling_params, offset: int):
        """Return a deep copy of *sampling_params* with ``seed += offset``."""
        params = copy.deepcopy(sampling_params)
        attr = self.SEED_ATTR
        setattr(params, attr, getattr(params, attr) + offset)
        return params

    def set_sampling_params_seed(self, sampling_params, seed: int):
        """Set the seed of *sampling_params*."""
        attr = self.SEED_ATTR
        setattr(sampling_params, attr, seed)
        return sampling_params

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
        """Build sampling parameters (must be overridden by subclasses).

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(f"infer_engine does not implement get_sampling_params")

    def get_sampling_params_from_config(
        self,
        infer_engine_config: InferEngineConfig,
        stop_at_token_id: int,
        logit_bias: Optional[dict[str | int, int]] = None,
    ):
        """Build sampling parameters from an InferEngineConfig.

        Parameters
        ----------
        infer_engine_config : InferEngineConfig
        stop_at_token_id : int
            Token ID that signals end of generation.
        logit_bias : dict, optional

        Returns
        -------
        object
            Sampling parameters.
        """
        # logit_bias: 尽可能不要使用这个东西。如果用了，这里打一个 warning，方便我们 debug 问题。
        return self.get_sampling_params(
            n=1,  # by passing vllm async llm issues
            temperature=infer_engine_config.temperature,
            top_k=infer_engine_config.top_k if infer_engine_config.top_k > 0 else -1,
            top_p=infer_engine_config.top_p,
            max_tokens=infer_engine_config.generate_max_tokens,
            stop_token_ids=[stop_at_token_id],
            seed=infer_engine_config.seed,
            repetition_penalty=infer_engine_config.repetition_penalty,
            frequency_penalty=infer_engine_config.frequency_penalty,
            presence_penalty=infer_engine_config.presence_penalty,
            min_p=infer_engine_config.min_p,
            logit_bias=logit_bias,
        )

    def async_generate(self, inp, sampling_params, request_id: str, return_routed_experts=False):
        raise NotImplementedError(f"infer_engine does not implement async_generate")

    async def wait_and_get_async_generate_output(self, async_generators):
        raise NotImplementedError(
            f"infer_engine does not implement wait_and_get_async_generate_output"
        )

    async def flush_cache(self):
        raise NotImplementedError(f"infer_engine does not implement flush_cache")

    async def get_load(self):
        """Return scheduler load metrics for this engine.

        Returns
        -------
        dict
            ``num_reqs``: running + waiting request count.
            ``num_running_reqs``: running-only request count.
            ``num_waiting_reqs``: waiting-only request count.
        """
        raise NotImplementedError(f"infer_engine does not implement get_load")

    async def wake_up(self, *args, **kwargs):
        raise NotImplementedError(f"infer_engine does not implement wake_up")

    async def sleep(self, *args, **kwargs):
        raise NotImplementedError(f"infer_engine does not implement sleep")

    def update_engine_weight_by_model_idx(self, rm_model_idx):
        raise NotImplementedError(
            f"infer_engine does not implement update_gen_rm_weight_by_model_idx"
        )

    async def update_weights(self, *args, **kwargs):
        raise NotImplementedError(f"infer_engine does not implement update_weights")

    async def save_engine_ckpt(self, save_ckpt_path):
        raise NotImplementedError(f"infer_engine does not implement save_engine_ckpt")

    @staticmethod
    def _map_sgl_attention_backend_to_vllm_attention_backend(sgl_attention_backend):
        """Map an sglang attention backend name to the vLLM equivalent.

        Parameters
        ----------
        sgl_attention_backend : str or None
            sglang-style backend name (lowercase), registered via
            ``see: sglang/python/sglang/srt/layers/attention/attention_registry.py:register_attention_backend``.

        Returns
        -------
        str
            vLLM-style backend name (uppercase), from
            ``vllm/vllm/v1/attention/backends/registry.py: AttentionBackendEnum``.
        """
        _SGL_TO_VLLM = {
            "flashinfer": "FLASHINFER",
            "triton": "TRITON_ATTN",
            "torch_native": "TORCH_SDPA",
            "flex_attention": "FLEX_ATTENTION",
            "fa3": "FLASH_ATTN",
            "fa4": "FLASH_ATTN",
            "flashmla": "FLASHMLA",
            "cutlass_mla": "CUTLASS_MLA",
            "trtllm_mla": "TRITON_MLA",
            "torch_sdpa": "TORCH_SDPA",
        }
        if sgl_attention_backend is None:
            return None
        key = sgl_attention_backend.lower()
        if key not in _SGL_TO_VLLM:
            raise ValueError(
                f"Unknown sglang attention backend '{sgl_attention_backend}'. "
                f"Supported mappings: {list(_SGL_TO_VLLM.keys())}"
            )
        return _SGL_TO_VLLM[key]

    def shutdown(self) -> None:
        """Graceful shutdown hook.

        Default no-op. vLLM relies on ray's automatic cleanup so vllm's
        ``VllmEngine`` keeps this default. ``SglangEngine`` overrides this
        hook and delegates to sglang's own shutdown sequence so its subprocess
        watchdog is stopped before child processes are reaped.
        """
        return

    @staticmethod
    def from_engine_args(
        infer_engine_impl,
        model_path=None,
        dtype='bfloat16',
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        enable_deepep=False,
        gpu_memory_utilization=0.67,
        enforce_eager=True,
        tp_rank=None,
        engine_idx=None,
        dist_init_addr=None,
        num_gpus_per_node=8,
        infer_engine_role=None,
        rm_idx=None,
        use_fast=False,
        max_running_requests=None,
        load_format="auto",
        log_level="info",
        allow_auto_truncate=False,
        enable_custom_logit_processor=False,
        attention_backend=None,
        enable_weights_cpu_backup=True,
        sgl_crash_dump_folder=None,
        mm_per_request_timeout=10,  # sglang default value
        sgl_chunked_prefill_size=None,
        sgl_schedule_conservativeness=None,
        sgl_sleep_on_idle=False,
        sgl_mamba_full_memory_ratio=None,
        sgl_mamba_scheduler_strategy=None,
        sgl_enable_spec_v2=False,
        enable_return_routed_experts=False,
        apply_deterministic_mode=False,
        placement_type=None,
        pg_bundle_indices=None,
        base_gpu_id: Optional[int] = None,
        enable_mtp: bool = False,
        **extra_infer_engine_config
    ):
        """Factory method to create an InferEngine from engine arguments.

        Supports ``'vllm'`` and ``'sglang'`` backends.

        Parameters
        ----------
        infer_engine_impl : str
            Backend implementation (``'vllm'`` or ``'sglang'``).
        model_path : str
        dtype : str, optional
        tensor_parallel_size : int, optional
        gpu_memory_utilization : float, optional
        enforce_eager : bool, optional
            Disable CUDA graphs, by default *True*.
        tp_rank : int, optional
        engine_idx : int, optional
        dist_init_addr : str, optional
            Distributed init address (``'host:port'``).
        infer_engine_role : str, optional
            Role (``'sampler'``, ``'gen-rm'``, etc.).
        rm_idx : int, optional
            Reward-model index for gen-rm log sharding.
        placement_type : str, optional
            Placement mode for lifecycle behavior.

        Returns
        -------
        InferEngine
            Concrete engine instance (VllmEngine or SglangEngine).
        """
        assert infer_engine_role in [None, "sampler", "gen-rm", "off-policy-sampler"]
        assert model_path is not None
        assert tp_rank is not None and engine_idx is not None
        assert placement_type is not None

        if infer_engine_impl == "vllm":
            assert not enable_mtp, "enable mtp is not supported by vllm"
            import vllm
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            from gpatch_v4.generation_backend.vllm_engine import VllmEngine

            # Must be called AFTER import vllm, because vllm's module-level
            # dictConfig() calls _clearExistingHandlers() which removes all
            # logging handlers.
            infer_engine_log_file = configure_third_party_logging(
                infer_engine_role or "infer_engine",
                "vllm",
                rank=tp_rank,
                engine_idx=engine_idx,
                rm_idx=rm_idx,
            )

            if apply_deterministic_mode:
                os.environ["VLLM_BATCH_INVARIANT"] = "1"

            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.01"
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
            # Specify which bundles this engine's TP workers should use.
            if pg_bundle_indices is not None:
                bundle_indices = [str(b) for b in pg_bundle_indices]
            else:
                base_bundle = engine_idx * min(tensor_parallel_size, num_gpus_per_node)
                bundle_indices = [str(base_bundle + i) for i in range(tensor_parallel_size)]
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(bundle_indices)

            # Prefer "mp" (lower overhead) when all TP/PP workers fit on a
            # single node and we can pin them via CUDA_VISIBLE_DEVICES.
            mp_size = tensor_parallel_size * pipeline_parallel_size
            use_mp = (
                base_gpu_id is not None and mp_size <= 8 and
                base_gpu_id + mp_size <= num_gpus_per_node
            )
            if use_mp:
                gpu_ids_str = ",".join(str(base_gpu_id + i) for i in range(mp_size))
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
                log(f"vllm mp backend: CUDA_VISIBLE_DEVICES={gpu_ids_str}")

            attention_backend = InferEngine._map_sgl_attention_backend_to_vllm_attention_backend(
                attention_backend
            )
            log(f"vllm use attention backend: {attention_backend}")
            mm_processor_kwargs = {}
            if use_fast:
                mm_processor_kwargs = dict(use_fast=True)
            engine_args = AsyncEngineArgs(
                model=model_path,
                dtype=dtype,
                distributed_executor_backend="mp" if use_mp else "ray",
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                trust_remote_code=True,
                enable_sleep_mode=True,
                mm_processor_kwargs=mm_processor_kwargs,
                load_format=load_format,
                enable_return_routed_experts=enable_return_routed_experts,
                max_num_seqs=max_running_requests,
                attention_backend=attention_backend,
                weight_transfer_config=(
                    {
                        "backend": "nccl"
                    } if placement_type == "disaggregated" else {
                        "backend": "ipc"
                    }
                ),
                # Inject G-Core worker extension so ``gcore_update_weights_ipc``
                # becomes an RPC-callable method on every vLLM Worker.
                # See ``gpatch_v4.generation_backend.vllm_worker_extension`` for
                # why we bypass vLLM's built-in per-bucket layerwise reload.
                worker_extension_cls=(
                    "gpatch_v4.generation_backend.vllm_worker_extension."
                    "GCoreVllmWorkerExtension"
                ),
                **extra_infer_engine_config,
            )
            if infer_engine_log_file:
                engine_context = redirect_stdio_fds_to_file(infer_engine_log_file)
            else:
                engine_context = nullcontext()
            with engine_context:
                infer_engine = AsyncLLM.from_engine_args(engine_args)

            log(
                f"vLLM engine logging to level logs, infer engine log to {infer_engine_log_file}",
                rank=0,
            )
            return VllmEngine(infer_engine, model_path, infer_engine_role, placement_type)
        else:
            if sgl_enable_spec_v2:
                os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"

            import sglang as sgl

            assert version.parse(sgl.__version__
                                ) >= version.parse('0.5.3'), f"{sgl.__version__=} must be >= 0.5.3"
            assert enable_weights_cpu_backup, "enable_weights_cpu_backup must be True in sglang"
            if sgl.__version__ in ['0.5.4.post3', '0.5.7']:
                patch_import_processors()

            import sglang.srt.entrypoints.engine

            from gpatch_v4.generation_backend.sglang_engine import SglangEngine

            # Must be called AFTER import sglang, in case sglang's module-level
            # logging setup clears existing handlers.
            infer_engine_log_file = configure_third_party_logging(
                infer_engine_role or "infer_engine",
                "sglang",
                rank=tp_rank,
                engine_idx=engine_idx,
                rm_idx=rm_idx,
            )
            setattr(sys.modules["sglang.srt.entrypoints.engine"], 'signal', FakeSignal)

            assert infer_engine_impl == "sglang"
            assert dist_init_addr is not None
            assert pipeline_parallel_size == 1
            assert expert_parallel_size <= tensor_parallel_size
            extra_args = {}
            extra_args["watchdog_timeout"] = 600
            extra_args["crash_dump_folder"] = sgl_crash_dump_folder
            if sgl_chunked_prefill_size is not None:
                extra_args["chunked_prefill_size"] = sgl_chunked_prefill_size
            if sgl_schedule_conservativeness is not None:
                extra_args["schedule_conservativeness"] = sgl_schedule_conservativeness
            if sgl_sleep_on_idle:
                extra_args["sleep_on_idle"] = True
            if sgl_mamba_full_memory_ratio is not None:
                assert version.parse(sgl.__version__) >= version.parse(
                    '0.5.4'
                ), f"sgl_mamba_full_memory_ratio is only supported by sglang 0.5.4 and later version, now is {sgl.__version__}"
                extra_args["mamba_full_memory_ratio"] = sgl_mamba_full_memory_ratio
            if sgl_mamba_scheduler_strategy is not None:
                assert version.parse(sgl.__version__) >= version.parse(
                    '0.5.4'
                ), f"sgl_mamba_scheduler_strategy is only supported by sglang 0.5.4 and later version, now is {sgl.__version__}"
                extra_args["mamba_scheduler_strategy"] = sgl_mamba_scheduler_strategy
            if enable_return_routed_experts:
                assert version.parse(sgl.__version__) >= version.parse(
                    '0.5.7'
                ), "enable_return_routed_experts is only supported by 0.5.7 and later version"
                extra_args["enable_return_routed_experts"] = True
                # router is not captured for trt-llm backend
                extra_args["moe_runner_backend"] = "cutlass"

            nnodes = int(math.ceil(tensor_parallel_size / num_gpus_per_node))
            node_rank = tp_rank // num_gpus_per_node
            if base_gpu_id is None:
                base_gpu_id = (
                    engine_idx * min(tensor_parallel_size, num_gpus_per_node)
                ) % num_gpus_per_node
            if version.parse(sgl.__version__) <= version.parse('0.5.9'):
                extra_args["mm_per_request_timeout"] = mm_per_request_timeout

            if enable_mtp:
                assert version.parse(sgl.__version__) >= version.parse(
                    '0.5.4'
                ), f"mlp weights update is not supported before sglang 0.5.4, now is {sgl.__version__}"
                # mtp is suitable for EAGLE
                # TODO(hessianliu): pass spec config in
                extra_args.update(
                    {
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "speculative_algorithm": "EAGLE",
                        "skip_server_warmup": True,
                    }
                )

            extra_args.update(extra_infer_engine_config)
            log(f"init infer engine with extra_args: {extra_args=}", rank=0)

            server_args = sgl.ServerArgs(
                model_path=model_path,
                tp_size=tensor_parallel_size,
                ep_size=expert_parallel_size,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                node_rank=node_rank,
                base_gpu_id=base_gpu_id,
                mem_fraction_static=gpu_memory_utilization,
                trust_remote_code=True,
                enable_memory_saver=True,
                enable_weights_cpu_backup=enable_weights_cpu_backup,
                skip_tokenizer_init=True,
                max_running_requests=max_running_requests,
                cuda_graph_max_bs=max_running_requests,
                disable_cuda_graph=enforce_eager,
                load_format=load_format,
                log_level=log_level,
                allow_auto_truncate=allow_auto_truncate,
                enable_custom_logit_processor=enable_custom_logit_processor,
                attention_backend=attention_backend,
                enable_deterministic_inference=apply_deterministic_mode,
                **extra_args,
            )
            if engine_idx == 0:
                log(
                    f"init infer engine {server_args} with {extra_args=} "
                    f"logging to level logs, infer engine log to {infer_engine_log_file}"
                )
            if infer_engine_log_file:
                engine_context = redirect_stdio_fds_to_file(infer_engine_log_file)
            else:
                engine_context = nullcontext()
            with engine_context:
                infer_engine = sgl.Engine(server_args=server_args)

            return SglangEngine(infer_engine, model_path, infer_engine_role, placement_type)
