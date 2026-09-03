from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpatch_v4.extended_model.deepseek_v4 import DeepseekV4PostInitModel
from gpatch_v4.models.deepseek_v4.checkpoint import infer_dspark_num_layers
from gpatch_v4.models.deepseek_v4.dspark import (
    DSparkConfidenceHead,
    DSparkForwardOutput,
    build_dspark_sparse_topk,
    prepare_dspark_batch,
)
from gpatch_v4.models.deepseek_v4.thd import pack_sequences
from gpatch_v4.models.deepseek_v4.weight_export import (
    _classify_for_save,
    _model_to_disk_key,
)
from gpatch_v4.training_backend.fsdp2_backend.dspark_loss import (
    calculate_dspark_loss,
    dspark_loss_denominator,
)
from gpatch_v4.training_backend.loss.fsdp2_specific_loss import (
    Fsdp2FinetuneLossInput,
    fsdp2_cross_entropy_loss,
)


def _shifted_batch():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    labels = torch.tensor([[-100, -100, 13, 14, 15, -100]])
    loss_mask = (labels >= 0).float()
    return input_ids, labels, loss_mask


def _pack_config(sliding_window: int = 4) -> SimpleNamespace:
    # Tiny fixture only needs m=4; including HCA m=128 would force T>=128.
    return SimpleNamespace(
        compress_rates={"compressed_sparse_attention": 4},
        sliding_window=sliding_window,
    )


def _thd_two_seg_with_mid_pads():
    """两段真实 pack：每段 3 token → pad 到 4，轴中间有 pad。

    Layout（pad_to_multiple_of=4）::

        ids:    [10, 11, 12, PAD, 20, 21, 22, PAD]
        labels: [11, 12, 13, -100, 21, 22, 23, -100]
        #       └── seg0 ──┘ pad  └── seg1 ──┘ pad
    """
    ids_list = [
        torch.tensor([10, 11, 12], dtype=torch.long),
        torch.tensor([20, 21, 22], dtype=torch.long),
    ]
    labels_list = [
        torch.tensor([11, 12, 13], dtype=torch.long),
        torch.tensor([21, 22, 23], dtype=torch.long),
    ]
    packed_ids, packed_pos, packed_labels, psp = pack_sequences(
        ids_list,
        labels_list,
        config=_pack_config(sliding_window=4),
        pad_to_multiple_of=4,
        cp_size=1,
        pad_token_id=0,
        label_ignore_index=-100,
        device=torch.device("cpu"),
    )
    assert packed_labels is not None
    loss_mask = (packed_labels != -100).float()
    return packed_ids, packed_labels, loss_mask, packed_pos, psp


def test_dspark_batch_uses_shifted_labels():
    input_ids, labels, loss_mask = _shifted_batch()
    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=input_ids.shape[1],
        block_size=3,
    )

    assert batch.anchor_positions[0, :2].tolist() == [3, 4]
    assert batch.target_ids[0, 0].tolist() == [14, 15, 0]
    assert batch.prev_token_ids[0, 0].tolist() == [13, 14, 15]
    assert batch.eval_mask[0, 0].tolist() == [True, True, False]
    assert not batch.block_keep_mask[0, 2:].any()


def test_dspark_anchor_sampling_is_reproducible_and_without_replacement():
    input_ids = torch.arange(16).reshape(2, 8)
    labels = input_ids + 1
    labels[:, -1] = -100
    loss_mask = (labels >= 0).float()
    loss_mask[1] = torch.tensor([1, 1, 1, 0, 0, 1, 1, 0])

    torch.manual_seed(17)
    first = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=3,
        block_size=2,
    )
    torch.manual_seed(17)
    second = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=3,
        block_size=2,
    )

    torch.testing.assert_close(first.anchor_positions, second.anchor_positions)
    torch.testing.assert_close(first.block_keep_mask, second.block_keep_mask)
    valid_positions = ({1, 2, 3, 4, 5, 6}, {1, 2, 6})
    for row, valid in enumerate(valid_positions):
        anchors = first.anchor_positions[row, first.block_keep_mask[row]].tolist()
        assert anchors == sorted(anchors)
        assert len(anchors) == len(set(anchors))
        assert set(anchors).issubset(valid)


def test_dspark_batch_pads_missing_anchors_with_disabled_blocks():
    input_ids = torch.arange(4).reshape(1, 4)
    labels = torch.full_like(input_ids, -100)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)

    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=6,
        block_size=3,
    )

    assert batch.anchor_positions.tolist() == [[0, 0, 0, 0, 0, 0]]
    assert not batch.block_keep_mask.any()
    assert not batch.eval_mask.any()
    assert not batch.target_ids.any()


def test_dspark_eval_mask_stops_at_first_invalid_target():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]])
    labels = torch.tensor([[11, 12, 13, 14, -100, 16, -100]])
    loss_mask = (labels >= 0).float()

    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=input_ids.shape[1],
        block_size=4,
    )
    anchor_slot = (batch.anchor_positions[0] == 2).nonzero().item()

    assert batch.target_ids[0, anchor_slot].tolist() == [13, 14, 0, 0]
    assert batch.eval_mask[0, anchor_slot].tolist() == [True, True, False, False]
    assert batch.prev_token_ids[0, anchor_slot].tolist() == [12, 13, 14, 0]
    assert batch.target_hidden_indices[0, anchor_slot].tolist() == [2, 3, 4, 5]


def test_dspark_sparse_topk_is_block_local():
    input_ids, labels, loss_mask = _shifted_batch()
    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=3,
        block_size=3,
    )
    topk = build_dspark_sparse_topk(
        batch,
        seq_len=input_ids.shape[1],
        block_size=3,
        sliding_window=4,
    )

    assert topk[0, 0].tolist() == [-1, 0, 1, 2, 6, 7, 8]
    assert torch.equal(topk[0, 0], topk[0, 1])
    assert topk[0, 3, -3:].tolist() == [9, 10, 11]


def test_dspark_thd_prepare_uses_real_pack_sequences():
    input_ids, labels, loss_mask, _, psp = _thd_two_seg_with_mid_pads()
    assert psp.layout is not None
    assert psp.layout.pad_token_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert psp.layout.seg_id_per_token.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=3,
        block_size=3,
        packed_seq_params=psp,
    )

    assert batch.anchor_positions.tolist() == [[1, 2, 0, 5, 6, 0]]
    assert batch.block_keep_mask.tolist() == [[True, True, False, True, True, False]]
    anchors = batch.anchor_positions[0, batch.block_keep_mask[0]].tolist()
    # pos0 永远无效；pos3/7 是 pad；pos4 是 seg1 起点（跨 seg 邻接 mask 失败）。
    assert anchors == [1, 2, 5, 6]

    slot2 = (batch.anchor_positions[0] == 2).nonzero().item()
    assert batch.eval_mask[0, slot2].tolist() == [True, False, False]
    assert batch.target_ids[0, slot2].tolist() == [13, 0, 0]
    assert batch.prev_token_ids[0, slot2].tolist() == [12, 13, 0]

    slot6 = (batch.anchor_positions[0] == 6).nonzero().item()
    assert batch.eval_mask[0, slot6].tolist() == [True, False, False]
    assert batch.target_ids[0, slot6].tolist() == [23, 0, 0]
    assert batch.prev_token_ids[0, slot6].tolist() == [22, 23, 0]


def test_dspark_thd_topk_masks_pad_inside_context_window():
    input_ids, labels, loss_mask, _, psp = _thd_two_seg_with_mid_pads()
    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=3,
        block_size=3,
        packed_seq_params=psp,
    )
    topk = build_dspark_sparse_topk(
        batch,
        seq_len=input_ids.shape[1],
        block_size=3,
        sliding_window=4,
        packed_seq_params=psp,
    )

    # anchor=5（seg1）：context 候选 [1,2,3,4]；1/2 跨 seg，3 是段间 pad，仅 4 可见。
    slot5 = (batch.anchor_positions[0] == 5).nonzero().item()
    draft5 = input_ids.shape[1] + slot5 * 3
    assert topk[0, slot5 * 3].tolist() == [-1, -1, -1, 4, draft5, draft5 + 1, draft5 + 2]

    # anchor=2（seg0）：context 候选 [-2,-1,0,1] → 负下标变 -1，0/1 同 seg 保留。
    slot2 = (batch.anchor_positions[0] == 2).nonzero().item()
    draft2 = input_ids.shape[1] + slot2 * 3
    assert topk[0, slot2 * 3].tolist() == [-1, -1, 0, 1, draft2, draft2 + 1, draft2 + 2]


def test_dspark_thd_loss_respects_truncated_eval_mask():
    input_ids, labels, loss_mask, _, psp = _thd_two_seg_with_mid_pads()
    batch = prepare_dspark_batch(
        input_ids,
        labels,
        loss_mask,
        num_anchors=2,
        block_size=3,
        packed_seq_params=psp,
    )
    assert batch.block_keep_mask[0].tolist() == [True, True, True, True]
    # anchors [1,2,5,6] → eval_mask 行长 [2,1,2,1]
    assert batch.eval_mask[0].tolist() == [
        [True, True, False],
        [True, False, False],
        [True, True, False],
        [True, False, False],
    ]
    assert batch.eval_mask.sum().item() == 6

    decay_gamma = 4.0
    denominator = dspark_loss_denominator(
        batch,
        block_size=3,
        loss_decay_gamma=decay_gamma,
    )
    expected_den = 4.0 + 2.0 * float(torch.exp(torch.tensor(-1.0 / decay_gamma)))
    torch.testing.assert_close(denominator, torch.tensor(expected_den))

    torch.manual_seed(3)
    num_anchors = batch.anchor_positions.shape[1]
    vocab_size = int(batch.target_ids.max().item()) + 1
    draft_logits = torch.randn(1, num_anchors, 3, vocab_size, requires_grad=True)
    output = DSparkForwardOutput(
        draft_logits=draft_logits,
        target_logits=torch.randn(1, num_anchors, 3, vocab_size),
        target_ids=batch.target_ids,
        eval_mask=batch.eval_mask,
        block_keep_mask=batch.block_keep_mask,
        confidence_logits=torch.randn(1, num_anchors, 3, requires_grad=True),
    )
    result = calculate_dspark_loss(
        outputs=output,
        global_denominator=denominator,
        dp_size=1,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_loss_alpha=1.0,
        loss_decay_gamma=decay_gamma,
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert draft_logits.grad is not None
    assert result.ce_loss.item() > 0


def test_dspark_three_term_loss():
    torch.manual_seed(7)
    draft_logits = torch.randn(1, 1, 3, 5, requires_grad=True)
    target_logits = torch.randn(1, 1, 3, 5)
    confidence_logits = torch.randn(1, 1, 3, requires_grad=True)
    eval_mask = torch.tensor([[[True, True, False]]])
    output = DSparkForwardOutput(
        draft_logits=draft_logits,
        target_logits=target_logits,
        target_ids=torch.tensor([[[1, 2, 0]]]),
        eval_mask=eval_mask,
        block_keep_mask=torch.tensor([[True]]),
        confidence_logits=confidence_logits,
    )
    input_ids = torch.tensor([[3, 4, 5]])
    batch = prepare_dspark_batch(
        input_ids,
        torch.tensor([[-100, 1, 2]]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        num_anchors=1,
        block_size=3,
    )
    batch.eval_mask = eval_mask
    denominator = dspark_loss_denominator(
        batch,
        block_size=3,
        loss_decay_gamma=4.0,
    )
    result = calculate_dspark_loss(
        outputs=output,
        global_denominator=denominator,
        dp_size=1,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_loss_alpha=1.0,
        loss_decay_gamma=4.0,
    )

    weights = eval_mask.float() * torch.exp(
        -torch.arange(3).float() / 4.0
    ).view(1, 1, -1)
    ce = F.cross_entropy(
        draft_logits.reshape(-1, 5),
        output.target_ids.reshape(-1),
        reduction="none",
    ).reshape(1, 1, 3)
    draft_probs = draft_logits.softmax(dim=-1)
    target_probs = target_logits.softmax(dim=-1)
    l1 = (draft_probs - target_probs).abs().sum(dim=-1)
    accept_rate = 1.0 - 0.5 * l1
    confidence = F.binary_cross_entropy_with_logits(
        confidence_logits,
        accept_rate.detach(),
        reduction="none",
    )
    expected = (
        0.1 * (ce * weights).sum()
        + 0.9 * (l1 * weights).sum()
        + (confidence * weights).sum()
    ) / weights.sum()
    torch.testing.assert_close(result.ce_loss, (ce * weights).sum() / weights.sum())
    torch.testing.assert_close(result.l1_loss, (l1 * weights).sum() / weights.sum())
    torch.testing.assert_close(
        result.confidence_loss,
        (confidence * weights).sum() / weights.sum(),
    )
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert draft_logits.grad is not None
    assert confidence_logits.grad is not None


def test_dspark_loss_matches_deepspec_reference(monkeypatch):
    reference_root = Path(__file__).resolve().parents[3] / "DeepSpec"
    if not (reference_root / "deepspec").is_dir():
        pytest.skip("DeepSpec checkout is required for the differential test")
    monkeypatch.syspath_prepend(str(reference_root))

    from deepspec.modeling.dspark.common import (
        DSparkForwardOutput as DeepSpecForwardOutput,
    )
    from deepspec.modeling.dspark.loss import compute_dspark_loss
    from deepspec.utils.metrics import reset as reset_deepspec_metrics

    torch.manual_seed(19)
    draft_logits = torch.randn(1, 2, 3, 5)
    target_logits = torch.randn(1, 2, 3, 5)
    confidence_logits = torch.randn(1, 2, 3)
    target_ids = torch.tensor([[[1, 2, 3], [4, 0, 0]]])
    eval_mask = torch.tensor([[[True, True, True], [True, False, False]]])
    block_keep_mask = torch.ones(1, 2, dtype=torch.bool)

    gcore_draft = draft_logits.clone().requires_grad_()
    gcore_target = target_logits.clone().requires_grad_()
    gcore_confidence = confidence_logits.clone().requires_grad_()
    gcore_output = DSparkForwardOutput(
        draft_logits=gcore_draft,
        target_logits=gcore_target,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep_mask,
        confidence_logits=gcore_confidence,
    )
    decay = torch.exp(-torch.arange(3).float() / 4.0).view(1, 1, -1)
    denominator = (eval_mask.float() * decay).sum()
    gcore_result = calculate_dspark_loss(
        outputs=gcore_output,
        global_denominator=denominator,
        dp_size=1,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_loss_alpha=1.0,
        loss_decay_gamma=4.0,
    )

    deepspec_draft = draft_logits.clone().requires_grad_()
    deepspec_target = target_logits.clone().requires_grad_()
    deepspec_confidence = confidence_logits.clone().requires_grad_()
    deepspec_output = DeepSpecForwardOutput(
        draft_logits=deepspec_draft,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep_mask,
        confidence_pred=deepspec_confidence,
        aligned_target_logits=deepspec_target,
    )
    monkeypatch.setattr(
        "deepspec.modeling.dspark.loss.dist.get_world_size",
        lambda: 1,
    )
    reset_deepspec_metrics()
    deepspec_loss = compute_dspark_loss(
        outputs=deepspec_output,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )

    torch.testing.assert_close(
        gcore_result.loss,
        deepspec_loss,
        rtol=1e-5,
        atol=1e-6,
    )
    gcore_result.loss.backward()
    deepspec_loss.backward()
    torch.testing.assert_close(
        gcore_draft.grad,
        deepspec_draft.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        gcore_target.grad,
        deepspec_target.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        gcore_confidence.grad,
        deepspec_confidence.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    reset_deepspec_metrics()


def test_dspark_confidence_target_does_not_backpropagate_to_distributions():
    torch.manual_seed(19)
    draft_logits = torch.randn(1, 1, 2, 4, requires_grad=True)
    target_logits = torch.randn(1, 1, 2, 4, requires_grad=True)
    confidence_logits = torch.randn(1, 1, 2, requires_grad=True)
    eval_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    output = DSparkForwardOutput(
        draft_logits=draft_logits,
        target_logits=target_logits,
        target_ids=torch.tensor([[[1, 2]]]),
        eval_mask=eval_mask,
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        confidence_logits=confidence_logits,
    )

    result = calculate_dspark_loss(
        outputs=output,
        global_denominator=1.0 + torch.exp(torch.tensor(-0.25)),
        dp_size=1,
        ce_loss_alpha=0.0,
        l1_loss_alpha=0.0,
        confidence_loss_alpha=1.0,
        loss_decay_gamma=4.0,
    )
    result.loss.backward()

    assert not draft_logits.grad.any()
    assert not target_logits.grad.any()
    assert confidence_logits.grad.any()


def test_dspark_dp_scaling_recovers_global_token_weighted_mean():
    torch.manual_seed(23)
    outputs = []
    masks = (
        torch.tensor([[[True, True, True]]]),
        torch.tensor([[[True, False, False]]]),
    )
    for mask in masks:
        outputs.append(
            DSparkForwardOutput(
                draft_logits=torch.randn(1, 1, 3, 5),
                target_logits=torch.randn(1, 1, 3, 5),
                target_ids=torch.tensor([[[1, 2, 3]]]),
                eval_mask=mask,
                block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
                confidence_logits=torch.randn(1, 1, 3),
            )
        )

    decay = torch.exp(-torch.arange(3).float() / 4.0).view(1, 1, -1)
    global_denominator = sum(
        (output.eval_mask.float() * decay).sum() for output in outputs
    )
    results = [
        calculate_dspark_loss(
            outputs=output,
            global_denominator=global_denominator,
            dp_size=2,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_loss_alpha=1.0,
            loss_decay_gamma=4.0,
        )
        for output in outputs
    ]

    ce_num = torch.zeros(())
    l1_num = torch.zeros(())
    confidence_num = torch.zeros(())
    for output in outputs:
        weights = output.eval_mask.float() * decay
        ce = F.cross_entropy(
            output.draft_logits.reshape(-1, 5),
            output.target_ids.reshape(-1),
            reduction="none",
        ).reshape(1, 1, 3)
        draft_probs = output.draft_logits.softmax(dim=-1)
        target_probs = output.target_logits.softmax(dim=-1)
        l1 = (draft_probs - target_probs).abs().sum(dim=-1)
        accept_rate = (1.0 - 0.5 * l1).clamp(0.0, 1.0)
        confidence = F.binary_cross_entropy_with_logits(
            output.confidence_logits,
            accept_rate,
            reduction="none",
        )
        ce_num += (ce * weights).sum()
        l1_num += (l1 * weights).sum()
        confidence_num += (confidence * weights).sum()
    expected = (
        0.1 * ce_num + 0.9 * l1_num + confidence_num
    ) / global_denominator

    torch.testing.assert_close((results[0].loss + results[1].loss) / 2, expected)


def test_dspark_online_distillation_keeps_main_loss():
    torch.manual_seed(11)
    main_logits = torch.randn(1, 2, 5, requires_grad=True)
    draft_logits = torch.randn(1, 1, 2, 5, requires_grad=True)
    confidence_logits = torch.randn(1, 1, 2, requires_grad=True)
    dspark_output = DSparkForwardOutput(
        draft_logits=draft_logits,
        target_logits=torch.randn(1, 1, 2, 5),
        target_ids=torch.tensor([[[1, 2]]]),
        eval_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        confidence_logits=confidence_logits,
    )
    training = SimpleNamespace(
        dspark_ce_loss_alpha=0.1,
        dspark_l1_loss_alpha=0.9,
        dspark_confidence_loss_alpha=1.0,
        dspark_loss_decay_gamma=4.0,
        dspark_loss_scaling_factor=0.5,
    )
    result = fsdp2_cross_entropy_loss(
        SimpleNamespace(training=training),
        Fsdp2FinetuneLossInput(
            labels_2d=torch.tensor([[1, 2]]),
            loss_mask_2d=torch.ones(1, 2),
            batch={},
            dp_size=1,
            logits=main_logits,
            global_n=torch.tensor(2.0),
            vocab_size=5,
            loss_fct=nn.CrossEntropyLoss(reduction="none"),
            online_train_dspark=True,
            dspark_output=dspark_output,
            global_n_for_dspark=1.0 + torch.exp(torch.tensor(-0.25)),
        ),
    )

    assert result.dspark_result is not None
    expected_main = F.cross_entropy(
        main_logits.reshape(-1, 5),
        torch.tensor([1, 2]),
    )
    torch.testing.assert_close(result.main_loss, expected_main)
    torch.testing.assert_close(
        result.loss,
        result.main_loss + 0.5 * result.dspark_result.loss,
    )
    result.loss.backward()
    assert main_logits.grad is not None
    assert draft_logits.grad is not None
    assert confidence_logits.grad is not None


def test_dspark_checkpoint_depth_inference(tmp_path):
    weight_map = {
        "mtp.0.main_proj.weight": "model-00001-of-00001.safetensors",
        "mtp.1.attn.wkv.weight": "model-00001-of-00001.safetensors",
        "mtp.2.confidence_head.proj.weight": "model-00001-of-00001.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    assert infer_dspark_num_layers(str(tmp_path)) == 3


def test_dspark_main_projection_export_contract():
    disk_key = _model_to_disk_key("mtp.layers.0.main_proj.weight")
    assert disk_key == "mtp.0.main_proj.weight"
    assert _classify_for_save(disk_key, torch.empty(4, 4)) == "fp8_e4m3"


def test_dspark_markov_confidence_export_contract():
    markov_w1 = _model_to_disk_key("mtp.layers.2.markov_head.markov_w1.weight")
    markov_w2 = _model_to_disk_key("mtp.layers.2.markov_head.markov_w2.weight")
    confidence = _model_to_disk_key("mtp.layers.2.confidence_head.proj.weight")
    main_norm = _model_to_disk_key("mtp.layers.0.main_norm.weight")
    assert markov_w1 == "mtp.2.markov_head.markov_w1.weight"
    assert markov_w2 == "mtp.2.markov_head.markov_w2.weight"
    assert confidence == "mtp.2.confidence_head.proj.weight"
    assert main_norm == "mtp.0.main_norm.weight"
    assert _classify_for_save(markov_w1, torch.empty(4, 4)) == "bf16_passthrough"
    assert _classify_for_save(markov_w2, torch.empty(4, 4)) == "bf16_passthrough"
    assert _classify_for_save(confidence, torch.empty(4, 4)) == "bf16_passthrough"
    assert _classify_for_save(main_norm, torch.empty(4)) == "bf16_passthrough"


def test_dspark_confidence_head_has_no_bias():
    assert DSparkConfidenceHead(8).proj.bias is None


def test_dspark_post_init_keeps_joint_model_trainable():
    model = nn.Module()
    model.model = nn.Linear(2, 2)
    model.lm_head = nn.Linear(2, 2)
    model.mtp = nn.Linear(2, 2)
    config = SimpleNamespace(
        training=SimpleNamespace(
            enable_dspark=True,
            online_train_dspark=True,
            freeze_router_weight=False,
            freeze_csa_indexer=False,
            freeze_router_correction_bias=True,
        )
    )
    DeepseekV4PostInitModel(config)(model)

    assert all(p.requires_grad for p in model.model.parameters())
    assert all(p.requires_grad for p in model.lm_head.parameters())
    assert all(p.requires_grad for p in model.mtp.parameters())


def test_dspark_post_init_freezes_mtp_when_load_only():
    model = nn.Module()
    model.model = nn.Linear(2, 2)
    model.lm_head = nn.Linear(2, 2)
    model.mtp = nn.Linear(2, 2)
    config = SimpleNamespace(
        training=SimpleNamespace(
            enable_dspark=True,
            online_train_dspark=False,
            freeze_router_weight=False,
            freeze_csa_indexer=False,
            freeze_router_correction_bias=True,
        )
    )
    DeepseekV4PostInitModel(config)(model)

    assert all(p.requires_grad for p in model.model.parameters())
    assert all(p.requires_grad for p in model.lm_head.parameters())
    assert all(not p.requires_grad for p in model.mtp.parameters())
