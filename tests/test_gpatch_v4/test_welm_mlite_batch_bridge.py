import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MLITE_PATH = Path(__file__).resolve().parents[3] / "mlite" / "experimental" / "lite"
sys.path.insert(0, str(MLITE_PATH))

from gpatch_v4.extended_model.welm_v4 import WelmV4PrepareDataForwardLLM  # noqa: E402
from megatron.lite.runtime.backends.mlite.dynamic_cp import (  # noqa: E402
    _merge_source,
    _split_source,
)


def _make_prepare():
    config = SimpleNamespace(
        training=SimpleNamespace(training_backend="mlite"),
        policy=SimpleNamespace(model_arch="welmv4_moe"),
    )
    return WelmV4PrepareDataForwardLLM(config)


def test_welm_mlite_batches_preserve_sequence_boundaries_and_shift_masks():
    batch = [
        {
            "tokens": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([-100, -100, 3, 4, 5]),
        },
        {
            "tokens": torch.tensor([10, 11, 12]),
            "labels": torch.tensor([-100, 11, 12]),
        },
    ]

    runtime_batches, global_valid_tokens, max_seq_length = _make_prepare().sft_to_mlite_packed(
        batch,
        num_microbatches=1,
        seq_length=3,
        device=torch.device("cpu"),
        dp_size=1,
        dp_group=None,
    )

    packed_batch, loss_context = runtime_batches[0]
    assert packed_batch.seq_lens.tolist() == [4, 3]
    assert packed_batch.input_ids.tolist() == [2, 3, 4, 5, 10, 11, 12]
    assert packed_batch.labels.tolist() == packed_batch.input_ids.tolist()
    assert packed_batch.loss_mask.tolist() == [0, 1, 1, 1, 0, 1, 1]
    assert loss_context.source_batch["aligned_loss_mask"].values().tolist() == [
        1,
        1,
        1,
        0,
        1,
        1,
        0,
    ]
    assert global_valid_tokens.item() == 5
    assert max_seq_length == 3
    source_samples = _split_source(loss_context.source_batch, count=2)
    rerouted_source = _merge_source(
        [source_samples[1], source_samples[0]],
        torch.device("cpu"),
    )
    assert [
        row.tolist() for row in rerouted_source["aligned_loss_mask"].unbind()
    ] == [[1, 1, 0], [1, 1, 1, 0]]


def test_welm_mlite_batches_reject_labels_that_do_not_match_tokens():
    batch = [
        {
            "tokens": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([-100, 9, 3]),
        }
    ]

    with pytest.raises(ValueError, match="label must equal"):
        _make_prepare().sft_to_mlite_packed(
            batch,
            num_microbatches=1,
            seq_length=4,
            device=torch.device("cpu"),
            dp_size=1,
            dp_group=None,
        )
