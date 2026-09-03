from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.configs.config import DpoConfig, FinetuneConfig
from gpatch_v4.training_backend.megatron_backend import mixin


@pytest.mark.parametrize(
    ("config_cls", "batch_size", "expected_micro_batch_size"),
    [
        (FinetuneConfig, 64, 1),
        (DpoConfig, 128, 2),
    ],
)
def test_default_finetune_step_uses_effective_micro_batch_size(
    monkeypatch,
    config_cls,
    batch_size,
    expected_micro_batch_size,
):
    config = config_cls()
    config.training.train_gbs = 128
    config.training.train_mbs = 1
    config.training.seq_length = 32768
    config.training.use_dynamic_mbs = False

    engine = SimpleNamespace(
        config=config,
        training_config=config.training,
        dist_config=SimpleNamespace(dynamic_context_parallel=False),
        model=object(),
        _finetune_func=lambda seq_length: object(),
        _rm_train_func=lambda seq_length: object(),
    )
    captured = {}

    def fake_forward_backward(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(mixin.mpu, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mixin, "get_max_seqlen_within_dp", lambda seq_length: seq_length)
    monkeypatch.setattr(mixin, "get_forward_backward_func", lambda: fake_forward_backward)

    batch = [{"tokens": torch.zeros(286, dtype=torch.long)} for _ in range(batch_size)]
    result = mixin.ForwardStepMixin._finetune_step_default_fwd_bwd(
        engine,
        batch,
        num_microbatches=64,
        forward_only=False,
    )
    microbatches = list(captured["data_iterator"])

    assert captured["micro_batch_size"] == expected_micro_batch_size
    assert len(microbatches) == 64
    assert all(len(microbatch) == expected_micro_batch_size for microbatch in microbatches)
    assert result[3] == 64
