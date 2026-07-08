from abc import ABC, abstractmethod

import torch
from torch.profiler import ProfilerActivity


class DeviceProtocol(ABC):
    @property
    @abstractmethod
    def perf_activity(self) -> ProfilerActivity:
        ...

    @property
    @abstractmethod
    def profiler_module(self):
        ...

    @property
    @abstractmethod
    def is_cuda(self) -> bool:
        ...

    @property
    @abstractmethod
    def device_module(self):
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Device backend name.

        Currently this is also used as the ``torch.<name>`` device module
        identifier (e.g. ``"cuda"``, ``"mlu"``, ``"npu"``).
        """
        ...

    @property
    @abstractmethod
    def dist_backend(self) -> str:
        ...

    @property
    @abstractmethod
    def visible_devices_env_var(self) -> str:
        """Environment variable name for visible devices.

        For example ``"CUDA_VISIBLE_DEVICES"``.
        """
        ...

    @abstractmethod
    def get_flash_attn_varlen_func(self, **kwargs):
        ...


class CudaDeviceBackend(DeviceProtocol):
    @property
    def perf_activity(self) -> ProfilerActivity:
        return ProfilerActivity.CUDA

    @property
    def profiler_module(self):
        return torch.profiler

    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def device_module(self):
        return torch.cuda

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def dist_backend(self) -> str:
        return "nccl"

    @property
    def visible_devices_env_var(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    def get_flash_attn_varlen_func(self, **kwargs):
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func
