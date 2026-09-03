import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MLITE_PATH = Path(__file__).resolve().parents[3] / "mlite" / "experimental" / "lite"
sys.path.insert(0, str(MLITE_PATH))

from gpatch_v4.extended_model.qwen3_vl import Qwen3VLPrepareDataForward  # noqa: E402
from megatron.lite.model.qwen3_5.lite.vision import Qwen35VisionInputs  # noqa: E402


def _make_prepare():
    config = SimpleNamespace(
        training=SimpleNamespace(training_backend="mlite"),
        policy=SimpleNamespace(model_arch="qwen3_5_moe"),
    )
    return Qwen3VLPrepareDataForward(config)


def test_sft_to_mlite_packed_keeps_unpadded_sequences_without_cross_sequence_roll():
    batch = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([-100, -100, 3, 4]),
        },
        {
            "tokens": torch.tensor([10, 11, 12]),
            "labels": torch.tensor([-100, 11, 12]),
        },
    ]

    runtime_batches, global_valid_tokens, max_seq_length = _make_prepare().sft_to_mlite_packed(
        batch,
        num_microbatches=1,
        seq_length=4,
        device=torch.device("cpu"),
        dp_size=1,
        dp_group=None,
    )

    packed_batch, loss_context = runtime_batches[0]
    assert packed_batch.seq_lens.tolist() == [4, 3]
    assert packed_batch.input_ids.tolist() == [1, 2, 3, 4, 10, 11, 12]
    assert packed_batch.labels.tolist() == packed_batch.input_ids.tolist()
    assert packed_batch.input_ids.numel() == sum(packed_batch.seq_lens.tolist())
    assert packed_batch.loss_mask.tolist() == [0, 0, 1, 1, 0, 1, 1]
    assert loss_context.source_batch["aligned_loss_mask"].values().tolist() == [
        0,
        1,
        1,
        0,
        1,
        1,
        0,
    ]
    assert global_valid_tokens.item() == 4
    assert max_seq_length == 3


def test_sft_to_mlite_packed_rejects_labels_that_do_not_match_tokens():
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


def test_sft_to_mlite_packed_rejects_tokens_longer_than_seq_length_plus_one():
    batch = [
        {
            "tokens": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([-100, -100, 3, 4, 5]),
        }
    ]

    with pytest.raises(ValueError, match="bound tokens to seq_length\\+1"):
        _make_prepare().sft_to_mlite_packed(
            batch,
            num_microbatches=1,
            seq_length=3,
            device=torch.device("cpu"),
            dp_size=1,
            dp_group=None,
        )


def _vision_sample(
    tokens: list[int],
    *,
    patch_count: int = 2,
) -> dict:
    token_count = len(tokens)
    return {
        "tokens": torch.tensor(tokens),
        "labels": torch.tensor([-100] + tokens[1:]),
        "position_ids": torch.arange(3 * token_count).reshape(3, 1, token_count),
        "vision_data": torch.arange(patch_count * 4).reshape(patch_count, 4).float(),
        "vision_grid_thw": torch.tensor([[1, 1, patch_count]]),
        "image_input_mask": torch.tensor(
            [[index < patch_count for index in range(token_count)]],
            dtype=torch.bool,
        ),
    }


def _text_sample_with_mrope(tokens: list[int]) -> dict:
    token_count = len(tokens)
    return {
        "tokens": torch.tensor(tokens),
        "labels": torch.tensor([-100] + tokens[1:]),
        "position_ids": torch.arange(3 * token_count).reshape(3, 1, token_count),
        "image_input_mask": torch.zeros((1, token_count), dtype=torch.bool),
        "vision_data": None,
        "vision_grid_thw": None,
    }


def test_sft_to_mlite_packed_packs_single_image_collator_fields():
    batch = [
        _vision_sample([1, 2, 3, 4]),
        _vision_sample([10, 11, 12], patch_count=1),
    ]

    runtime_batches, global_valid_tokens, max_seq_length = _make_prepare().sft_to_mlite_packed(
        batch,
        num_microbatches=1,
        seq_length=4,
        device=torch.device("cpu"),
        dp_size=1,
        dp_group=None,
    )

    packed_batch, loss_context = runtime_batches[0]
    vision = packed_batch.extras["vision"]
    assert isinstance(vision, Qwen35VisionInputs)
    assert packed_batch.seq_lens.tolist() == [4, 3]
    assert packed_batch.position_ids.shape == (3, 1, 7)
    assert torch.equal(
        packed_batch.position_ids,
        torch.cat([batch[0]["position_ids"], batch[1]["position_ids"]], dim=2),
    )
    assert vision.pixel_values.shape == (3, 4)
    assert vision.image_grid_thw.tolist() == [[1, 1, 2], [1, 1, 1]]
    assert vision.image_input_mask.tolist() == [True, True, False, False, True, False, False]
    assert vision.num_images_per_sequence.tolist() == [1, 1]
    assert vision.spatial_merge_size == 1
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


def test_sft_to_mlite_packed_mixes_text_and_image_like_sft_train():
    batch = [
        _vision_sample([1, 2, 3, 4]),
        _text_sample_with_mrope([10, 11, 12]),
    ]

    runtime_batches, _, _ = _make_prepare().sft_to_mlite_packed(
        batch,
        num_microbatches=1,
        seq_length=4,
        device=torch.device("cpu"),
        dp_size=1,
        dp_group=None,
    )

    packed_batch, _ = runtime_batches[0]
    vision = packed_batch.extras["vision"]
    assert packed_batch.seq_lens.tolist() == [4, 3]
    assert packed_batch.position_ids.shape == (3, 1, 7)
    assert torch.equal(
        packed_batch.position_ids,
        torch.cat([batch[0]["position_ids"], batch[1]["position_ids"]], dim=2),
    )
    assert vision.pixel_values.shape == (2, 4)
    assert vision.image_grid_thw.tolist() == [[1, 1, 2]]
    assert vision.num_images_per_sequence.tolist() == [1, 0]
    assert vision.spatial_merge_size == 1
    assert vision.image_input_mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_sft_to_mlite_packed_text_only_still_forwards_position_ids():
    batch = [
        _text_sample_with_mrope([1, 2, 3, 4]),
        _text_sample_with_mrope([10, 11, 12]),
    ]

    runtime_batches, _, _ = _make_prepare().sft_to_mlite_packed(
        batch,
        num_microbatches=1,
        seq_length=4,
        device=torch.device("cpu"),
        dp_size=1,
        dp_group=None,
    )

    packed_batch, _ = runtime_batches[0]
    assert "vision" not in packed_batch.extras
    assert packed_batch.position_ids.shape == (3, 1, 7)
    assert torch.equal(
        packed_batch.position_ids,
        torch.cat([batch[0]["position_ids"], batch[1]["position_ids"]], dim=2),
    )


def test_sft_to_mlite_packed_aligns_collator_prefix_when_ids_longer_than_tokens():
    # Collator may keep position_ids/mask at map length s+1 while tokens are shorter.
    sample = _vision_sample([1, 2, 3, 4])
    sample["position_ids"] = torch.arange(15).reshape(3, 1, 5)
    sample["image_input_mask"] = torch.tensor(
        [[True, True, False, False, False]],
        dtype=torch.bool,
    )

    runtime_batches, _, _ = _make_prepare().sft_to_mlite_packed(
        [sample],
        num_microbatches=1,
        seq_length=4,
        device=torch.device("cpu"),
        dp_size=1,
        dp_group=None,
    )

    packed_batch, _ = runtime_batches[0]
    vision = packed_batch.extras["vision"]
    assert packed_batch.input_ids.tolist() == [1, 2, 3, 4]
    assert torch.equal(packed_batch.position_ids, sample["position_ids"][:, :, :4])
    assert vision.image_input_mask.tolist() == [True, True, False, False]


def test_sft_to_mlite_packed_rejects_multiple_images_per_sample():
    sample = _vision_sample([1, 2, 3])
    sample["vision_grid_thw"] = torch.tensor([[1, 1, 1], [1, 1, 1]])

    with pytest.raises(ValueError, match="single-image mlite samples"):
        _make_prepare().sft_to_mlite_packed(
            [sample],
            num_microbatches=1,
            seq_length=3,
            device=torch.device("cpu"),
            dp_size=1,
            dp_group=None,
        )
