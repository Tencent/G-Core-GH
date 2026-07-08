import torch

from gpatch_v4.core.device.protocol import CudaDeviceBackend, DeviceProtocol

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
    def dist_backend(self) -> str:
        # xpu 实际上是 bkcl， 只不过他们应该是做了迁移适配，直接按照 cuda 的习惯来
        return "nccl"

    @property
    def visible_devices_env_var(self) -> str:
        return "CUDA_VISIBLE_DEVICES"


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
