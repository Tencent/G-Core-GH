import os

import torch

from gpatch_v4.core.device.protocol import CudaDeviceBackend

XPU_PROPAGATE_ENV_KEYS = [
    "BKCL_SOCKET_IFNAME",
    "CUDA_DEVICE_ORDER",
    "CUDART_DUMMY_REGISTER",
    "XPU_FORCE_USERMODE_LAUNCH",
    "XPU_FORCE_SHARED_DEVICE_CONTEXT",
    "XMLIR_FA_GEMM_TYPE",
    "XBLAS_FC_HBM_VERSION",
    "XMLIR_PARALLEL_SAVE_MEMORY",
    "XMLIR_DISABLE_CUDA_ALLOCATOR",
    "XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL",
    "XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL",
    "XMLIR_DUMP_FALLBACK_OP_LIST_BOOL",
    "XMLIR_DIST_ASYNC_ISEND_IRECV",
    "XMLIR_BATCH_PARALLEL",
    "XMLIR_ENABLE_NEW_PG",
    "BKCL_RDMA_PROXY_DISABLE",
    "BKCL_USE_AR",
    "BKCL_RING_OPT",
    "BKCL_FLAT_RING",
    "BKCL_CCIX_RING",
    "BKCL_TREE_THRESHOLD",
    "BKCL_CCIX_BUFFER_GM",
    "BKCL_FORCE_L3_RDMA",
    "BKCL_RING_BUFFER_GM",
    "BKCL_ENABLE_XDR",
    "BKCL_RDMA_FORCE_TREE",
    "BKCL_XLINK_D2D",
    "BKCL_XLINK_ETH",
    "BKCL_XLINK_C2C",
    "BKCL_TRANS_UNSUPPORTED_DATATYPE",
    "BKCL_KL3_TURBO_MODE",
    "BKCL_RING_BUFFER_SIZE",
    "ALLREDUCE_ASYNC",
    "ALLGATHER_ASYNC",
    "ALLREDUCE_FUSION",
    "BKCL_TIMEOUT",
    "BKCL_RDMA_VERBS",
    "BKCL_RDMA_NICS",
    "BKCL_ALL_TO_ALL_OPT",
    "CUDA_DISABLE_PRINTF",
    "TORCH_XCCL_DEFAUTL_PG_TIMEOUT_MILSEC",
    "CUDA_ERROR_LEVEL",
    "ENABLE_VLLM_XPU_CPU_BINDING",
    "TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCH_XCCL_ENABLE_TIMING",
    "TORCH_FR_BUFFER_SIZE",
    "TORCH_XCCL_DEBUG_INFO_TEMP_FILE",
    "VERL_DEBUG_AVOID_WHERE",
    "SGL_CPU_QUANTIZATION",
    "XSGL_ENABLE_MEM_SAVER",
    "XPU_SUPPORT_IPC_EVENT",
    "CUDA_ENABLE_P2P_NO_UVA",
    "CUDA_FAKE_UVA_ENABLE",
    "HYDRAX_USE_PROTEUS",
    "ENABLE_FAST_BFP16",
    "USE_FAST_BFP16_FC",
    "XSGL_USE_TORCH_CAUSAL_CONV",
    "XSGL_INT8_LM_HEAD",
    "XMLIR_ENABLE_FAST_FC",
    "XDNN_FAST_DIV_SCALAR",
    "XMLIR_ENABLE_H2D_SSE_COPY",
    "XTE_DISABLE_MOE_DW_FUSION",
    "XTE_GROUPED_GEMM_LARGE_WEIGHT",
    "TINTER_RES",
    "XME_PLUGIN_ENABLE",
    "XME_OUTPUT_HEAD_REDUCE_MEMORY",
]


class XpuDeviceBackend(CudaDeviceBackend):
    """
    mlu hijack torch.cuda by
    'import torch.mlu.utils.gpu_migration'
    """
    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def propagate_env_keys(self) -> list[str]:
        return XPU_PROPAGATE_ENV_KEYS

    @property
    def device_module(self):
        # 直接按照 cuda 的习惯来, 因为 torch.xpu 被 intel 占了
        return torch.cuda

    @property
    def name(self) -> str:
        # 直接按照 cuda 的习惯来
        return "cuda"

    @property
    def kernel_device_name(self) -> str:
        # name 仍是 cuda（torch 接口），kernel 走独立注册
        return "xpu_priv"

    @property
    def dist_backend(self) -> str:
        # xpu 实际上是 bkcl， 只不过他们应该是做了迁移适配，直接按照 cuda 的习惯来
        return "nccl"

    @property
    def visible_devices_env_var(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    @property
    def sync_param_offload(self) -> bool:
        value = os.environ.get("GPATCH_SYNC_PARAM_OFFLOAD")
        if value is None:
            return False
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(
            f"GPATCH_SYNC_PARAM_OFFLOAD must be "
            f"1/0, true/false, yes/no, or on/off; got {value!r}"
        )

    def preprocess_before_build_sgl_engine(self) -> None:
        # Workaround: sglang.klx.klx_utils.klx_register uses functools.wraps which
        # rebinds the op's __globals__ to klx_utils.py. That module does not import
        # Optional/Union/List, so torch.library.infer_schema fails to eval the
        # stringified annotation 'Optional[torch.Tensor]' on the decorated op in
        # sglang.jit_kernel.moe_topk_sigmoid (and similar). Pre-inject typing names
        # into klx_utils' globals so infer_schema can eval them.
        try:
            import typing as _typing

            import sglang.klx.klx_utils as _sgl_klx_utils  # noqa: F401
            for _name in ("Optional", "Union", "List", "Tuple", "Dict", "Any"):
                _sgl_klx_utils.__dict__.setdefault(_name, getattr(_typing, _name))
        except ImportError:
            pass  # klx not present (non-XPU build)

        # Workaround for XPU/Kunlun: the parent process loads libcuda.so.1
        # (actually a Kunlun-shimmed libxpucuda) via torch_plugin's
        # ctypes.CDLL preload, which does not propagate to spawned children.
        # sglang's TorchMemorySaverAdapter then sets
        #   LD_PRELOAD=…/torch_memory_saver_hook_mode_preload_cu12.abi3.so
        # for each spawned scheduler subprocess; that .so has libcuda.so.1
        # as a NEEDED dep, so the freshly-execed python3.10 dies with
        # "error while loading shared libraries: libcuda.so.1" (exit 127)
        # before any Python runs.  Add the xcudart shim dir to
        # LD_LIBRARY_PATH so children inherit it.
        try:
            import sysconfig as _sysconfig
            _conda_prefix = _sysconfig.get_config_var("prefix")
            if _conda_prefix:
                _xcudart_lib = os.path.join(_conda_prefix, "xcudart", "lib")
                if os.path.isfile(os.path.join(_xcudart_lib, "libcuda.so.1")):
                    _existing = os.environ.get("LD_LIBRARY_PATH", "")
                    if _xcudart_lib not in _existing.split(":"):
                        os.environ["LD_LIBRARY_PATH"] = (
                            f"{_xcudart_lib}:{_existing}" if _existing else _xcudart_lib
                        )
        except Exception:
            pass  # non-XPU build / no xcudart shim — leave env alone


def initialize_device_backend():
    _IS_XPU_AVAILABLE = False
    try:
        # mock to XPU
        from torch_xmlir.symbrewrite.plugins.torch.mock_torch import (
            mock_init_reductions,
        )
        mock_init_reductions()
        print(f"xpu mock_init_reductions success")

        _IS_XPU_AVAILABLE = torch.cuda.is_available()
    except (ImportError, ModuleNotFoundError, AttributeError):
        _IS_XPU_AVAILABLE = False
    backend = None
    if _IS_XPU_AVAILABLE:
        backend = XpuDeviceBackend()
    return backend
