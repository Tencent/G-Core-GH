"""Mcore swap two-phase offload correctness via real EngineSwapMixin APIs."""

import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    _SKIP_REASON = f"{exc.name} is not installed"
else:
    _SKIP_REASON = None

if torch is not None and not torch.cuda.is_available():
    _SKIP_REASON = "requires CUDA"

if _SKIP_REASON is None:
    try:
        from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
        from gpatch_v4.training_backend.megatron_backend.mcore_swap_impl import (
            McoreSwapImpl,
            offload_tensor_to_cpu,
            onload_tensor_to_gpu,
            resize_offloaded_tensor_gpu,
            sync_and_resize_offloaded_tensors,
        )
    except ModuleNotFoundError as exc:
        _SKIP_REASON = f"{exc.name} is not installed"


class _TinyRefLikeModel(nn.Module):
    """Many small tensors — mimics MoE/ref named_parameters offload pressure."""

    def __init__(self, n_blocks: int = 32):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(128, 64, device="cuda", dtype=torch.bfloat16))
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64, bias=True, device="cuda", dtype=torch.bfloat16),
                nn.Linear(64, 64, bias=True, device="cuda", dtype=torch.bfloat16),
            ) for _ in range(n_blocks)
        ])
        self.register_buffer(
            "rope_cache",
            torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
        )


class _SwapEngine(EngineSwapMixin):
    """Minimal stand-in that still calls production swap_mixin + McoreSwapImpl."""

    def __init__(self, model, ref_model):
        self.swap_impl = McoreSwapImpl(early_swap_model=False)
        self.model = model
        self.ref_model = ref_model
        self.optimizer = None


def _snapshot_state_dict(modules):
    snap = {}
    for mi, module in enumerate(modules):
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            snap[(mi, name)] = tensor.detach().cpu().clone()
    return snap


def _assert_gpu_storage_empty(modules):
    for module in modules:
        for _, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            assert tensor.untyped_storage().size() == 0, (
                f"expected empty GPU storage after offload, got "
                f"size={tensor.untyped_storage().size()}"
            )


def _assert_matches_snapshot(modules, snap):
    for mi, module in enumerate(modules):
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            expected = snap[(mi, name)]
            actual = tensor.detach().cpu()
            assert torch.equal(actual, expected), f"mismatch after onload: {name}"


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class McoreSwapRoundtripTest(unittest.TestCase):
    def test_ref_model_offload_onload_roundtrip_via_swap_mixin(self):
        # 走训练侧真实入口：EngineSwapMixin.offload/onload_ref_model → McoreSwapImpl
        torch.manual_seed(0)
        ref = [_TinyRefLikeModel(n_blocks=48)]
        policy = [_TinyRefLikeModel(n_blocks=4)]
        engine = _SwapEngine(model=policy, ref_model=ref)

        before = _snapshot_state_dict(ref)
        n_tensors = len(before)
        self.assertGreater(n_tensors, 50)

        sync_count = {"n": 0}
        real_current_stream = torch.cuda.current_stream

        def wrapped_current_stream(*args, **kwargs):
            stream = real_current_stream(*args, **kwargs)
            if not getattr(stream, "_gcore_swap_test_wrapped", False):
                orig_sync = stream.synchronize

                def counting_sync(*sync_args, **sync_kwargs):
                    sync_count["n"] += 1
                    return orig_sync(*sync_args, **sync_kwargs)

                stream.synchronize = counting_sync
                stream._gcore_swap_test_wrapped = True
            return stream

        from unittest.mock import patch
        with patch.object(torch.cuda, "current_stream", wrapped_current_stream):
            engine.offload_ref_model()

        # two-phase: one stream sync for all staged D2H, not one per tensor
        self.assertEqual(sync_count["n"], 1, f"expected 1 stream sync, got {sync_count['n']}")
        _assert_gpu_storage_empty(ref)
        self.assertFalse(engine.get_swap_state().ref_model)

        engine.onload_ref_model()
        self.assertTrue(engine.get_swap_state().ref_model)
        torch.cuda.synchronize()
        _assert_matches_snapshot(ref, before)

        # 第二次：复用已有 gcore_cpu_data 分支
        before2 = _snapshot_state_dict(ref)
        engine.offload_ref_model()
        engine.onload_ref_model()
        torch.cuda.synchronize()
        _assert_matches_snapshot(ref, before2)

    def test_policy_model_offload_onload_roundtrip_via_swap_mixin(self):
        torch.manual_seed(1)
        policy = [_TinyRefLikeModel(n_blocks=16)]
        engine = _SwapEngine(model=policy, ref_model=None)

        before = _snapshot_state_dict(policy)
        engine.offload_model()
        _assert_gpu_storage_empty(policy)
        self.assertFalse(engine.get_swap_state().model)

        engine.onload_model()
        self.assertTrue(engine.get_swap_state().model)
        torch.cuda.synchronize()
        _assert_matches_snapshot(policy, before)

    def test_offload_tensor_to_cpu_single_helper_roundtrip(self):
        # 单 tensor：D2H 后需自行 sync + resize_(0)
        t = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
        expected = t.detach().cpu().clone()
        offload_tensor_to_cpu(t)
        sync_and_resize_offloaded_tensors([t])
        self.assertEqual(t.untyped_storage().size(), 0)
        onload_tensor_to_gpu(t)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(t.detach().cpu(), expected))
        # resize helper is a no-op on already-empty storage
        resize_offloaded_tensor_gpu(t)


if __name__ == "__main__":
    unittest.main()
