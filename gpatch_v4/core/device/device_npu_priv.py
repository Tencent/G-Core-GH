import math

import torch

from gpatch_v4.core.device.protocol import DeviceProtocol


class NpuDeviceBackend(DeviceProtocol):
    @property
    def perf_activity(self):
        import torch_npu
        return torch_npu.profiler.ProfilerActivity.NPU

    @property
    def profiler_module(self):
        import torch_npu
        return torch_npu.profiler

    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def device_module(self):
        return torch.npu

    @property
    def name(self) -> str:
        return "npu"

    @property
    def dist_backend(self) -> str:
        return "hccl"

    @property
    def visible_devices_env_var(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def get_flash_attn_varlen_func(self, max_segment_len=4096, **kwargs):
        import torch_npu
        atten_mask_npu = None

        def npu_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=True
        ):
            # lazy init
            nonlocal atten_mask_npu
            if atten_mask_npu is None:
                atten_mask_npu = torch.triu(
                    torch.ones([max_segment_len, max_segment_len]), diagonal=1
                ).bool().to(self.device_module.current_device())

            head_num = q.shape[1]
            if causal:
                output = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    head_num,
                    pse=None,
                    padding_mask=None,
                    atten_mask=atten_mask_npu,
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    keep_prob=1,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                    sparse_mode=3
                )[0]
            else:
                head_num = q.shape[1]
                output = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    head_num,
                    pse=None,
                    atten_mask=None,
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    keep_prob=1,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist())
                )[0]
            return output

        return npu_flash_attn_varlen_func


def initialize_device_backend():
    _IS_NPU_AVAILABLE = False
    try:
        import torch_npu  # noqa: F401
        _IS_NPU_AVAILABLE = torch.npu.is_available()
    except (ImportError, AttributeError):
        pass
    backend = None
    if _IS_NPU_AVAILABLE:
        backend = NpuDeviceBackend()
    return backend
