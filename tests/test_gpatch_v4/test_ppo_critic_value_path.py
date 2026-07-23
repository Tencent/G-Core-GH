from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    _SKIP_REASON = f"{exc.name} is not installed"
else:
    _SKIP_REASON = None

if torch is not None:
    try:
        from gpatch_v4.training_backend.megatron_backend.mcore_engine import McoreEngine
        from gpatch_v4.training_backend.megatron_backend import mcore_swap_impl
    except ModuleNotFoundError as exc:
        _SKIP_REASON = f"{exc.name} is not installed"
    except NameError as exc:
        # gpatch_v4.training_backend suppresses optional dependency ImportError,
        # then may reference symbols that were not initialized.
        _SKIP_REASON = f"training backend optional dependencies are unavailable: {exc}"


def _align_values(rollout_batches, values_list):
    engine = McoreEngine.__new__(McoreEngine)
    return McoreEngine._align_values_to_rollout_logprobs(
        engine,
        rollout_batches,
        values_list,
    )


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class PpoCriticValuePathTest(unittest.TestCase):
    def test_align_values_crops_right_padding(self):
        values = [[torch.arange(5, dtype=torch.float32)]]
        rollout_batches = [{"logprobs": [torch.zeros(3)]}]

        aligned = _align_values(rollout_batches, values)

        self.assertTrue(torch.equal(aligned[0][0], torch.tensor([0.0, 1.0, 2.0])))

    def test_align_values_left_pads_effective_prompt_prefix(self):
        values = [[torch.tensor([1.0, 2.0, 3.0])]]
        rollout_batches = [{
            "logprobs": [torch.zeros(5)],
            "prompt_lengths": [torch.tensor(3)],
            "sequence_lengths": [torch.tensor(6)],
        }]

        aligned = _align_values(rollout_batches, values)

        self.assertTrue(
            torch.equal(aligned[0][0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
        )

    def test_align_values_raises_on_unexplained_short_value(self):
        values = [[torch.tensor([1.0])]]
        rollout_batches = [{
            "logprobs": [torch.zeros(5)],
            "prompt_lengths": [torch.tensor(2)],
            "sequence_lengths": [torch.tensor(5)],
        }]

        with self.assertRaisesRegex(RuntimeError, "critic values shorter"):
            _align_values(rollout_batches, values)

    def test_align_values_raises_when_value_does_not_cover_response(self):
        values = [[torch.tensor([1.0])]]
        rollout_batches = [{
            "logprobs": [torch.zeros(5)],
            "prompt_lengths": [torch.tensor(5)],
            "sequence_lengths": [torch.tensor(7)],
        }]

        with self.assertRaisesRegex(RuntimeError, "critic values shorter"):
            _align_values(rollout_batches, values)

    def test_optimizer_swap_skips_missing_adam_moments(self):
        exp_avg = object()
        exp_avg_sq = object()
        optimizer = SimpleNamespace(
            optimizer=SimpleNamespace(
                state={
                    "initialized": {
                        "exp_avg": exp_avg,
                        "exp_avg_sq": exp_avg_sq,
                    },
                    "lazy": {},
                }
            )
        )
        offloaded = []
        onloaded = []

        with patch.object(mcore_swap_impl, "clear_memory"), \
            patch.object(mcore_swap_impl, "logging_memory_usage_details"), \
            patch.object(mcore_swap_impl, "offload_megatron_copy_params"), \
            patch.object(mcore_swap_impl, "onload_megatron_copy_params"), \
            patch.object(mcore_swap_impl, "offload_tensor_to_cpu", side_effect=offloaded.append), \
            patch.object(mcore_swap_impl, "onload_tensor_to_gpu", side_effect=onloaded.append):
            swap_impl = mcore_swap_impl.McoreSwapImpl()
            swap_impl.offload_optimizer(optimizer)
            swap_impl.onload_optimizer(optimizer)

        self.assertEqual(offloaded, [exp_avg, exp_avg_sq, None, None])
        self.assertEqual(onloaded, [exp_avg, exp_avg_sq, None, None])


if __name__ == "__main__":
    unittest.main()
