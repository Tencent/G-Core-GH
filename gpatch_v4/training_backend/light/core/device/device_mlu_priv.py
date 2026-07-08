import torch

from .protocol import CudaDeviceBackend, DeviceProtocol


class MluDeviceBackend(CudaDeviceBackend):
    """
    mlu hijack torch.cuda by
    'import torch.mlu.utils.gpu_migration'
    """
    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def device_module(self):
        return torch.mlu

    @property
    def name(self) -> str:
        return "mlu"

    @property
    def dist_backend(self) -> str:
        return "cncl"

    @property
    def visible_devices_env_var(self) -> str:
        return "MLU_VISIBLE_DEVICES"


def initialize_device_backend():
    try:
        # Side effect: monkey-patches torch CUDA API to forward to MLU.
        import torch.mlu.utils.gpu_migration  # noqa: F401
        if not torch.mlu.is_available():
            return None

        try:
            # patch torch._grouped_mm for mlu
            from apex.contrib.grouped_gemm import ops

            def grouped_gemm(input, weight, offs, **kwargs):
                batch_sizes = torch.cat([offs[:1], offs[1:] - offs[:-1]])
                apex_out = ops.gmm(
                    a=input, b=weight.transpose(-2, -1), batch_sizes=batch_sizes, trans_b=True
                )
                return apex_out

            # patch grouped_gemm
            torch._grouped_mm = grouped_gemm
        except Exception as e:
            print(f"patch torch._grouped_mm for mlu failed: {e}")
            pass
        return MluDeviceBackend()
    except (ImportError, ModuleNotFoundError):
        return None
