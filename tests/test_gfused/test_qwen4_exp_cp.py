"""Qwen3.8-Flash-Next contiguous CP slicing, PLE halo, and QSA parity tests."""
import datetime
import os
import pathlib
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.models.qwen4_exp import Qwen4ExpTextConfig
from gpatch_v4.models.qwen4_exp.cp import (
    Qwen4ExpCPContext,
    qwen4_exp_cp_chunk_data,
    qwen4_exp_cp_left_halo,
    qwen4_exp_pack_sequences,
)
from gpatch_v4.models.qwen4_exp.engram import Qwen4ExpEngramEmbedding
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextPLELayer,
)
from gpatch_v4.models.qwen4_exp.qsa import Qwen4ExpQSAAttention

WORLD_SIZE = 2
SEQUENCE_LENGTH = 16

TINY_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=32,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=256,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4,
    eos_token_id=1,
    tie_word_embeddings=False,
    use_cache=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def _config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(**TINY_KWARGS)


def _gdn_config() -> Qwen4ExpTextConfig:
    kwargs = dict(TINY_KWARGS)
    kwargs["linear_key_head_dim"] = 16
    kwargs["linear_value_head_dim"] = 16
    return Qwen4ExpTextConfig(**kwargs)


def _init_pg(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )


def _run_workers(worker) -> None:
    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = str(pathlib.Path(tmpdir) / "pg_init")
            mp.start_processes(
                worker,
                args=(WORLD_SIZE, init_file),
                nprocs=WORLD_SIZE,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)


def _context(
    rank: int,
    group: dist.ProcessGroup,
    *,
    logical_length: int = SEQUENCE_LENGTH,
    boundaries: torch.Tensor | None = None,
) -> Qwen4ExpCPContext:
    local_length = SEQUENCE_LENGTH // WORLD_SIZE
    positions = torch.arange(SEQUENCE_LENGTH)
    return Qwen4ExpCPContext(
        group=group,
        rank=rank,
        size=WORLD_SIZE,
        global_input_ids=(positions + 2).unsqueeze(0),
        global_padding_mask=(positions >= logical_length).unsqueeze(0),
        local_sequence_start=rank * local_length,
        local_sequence_length=local_length,
        global_cu_seqlens=boundaries,
    )


def test_packed_layout_keeps_documents_contiguous_and_only_pads_the_tail() -> None:
    ids = [torch.arange(5), torch.arange(10, 17)]
    labels = [document + 1 for document in ids]
    packed_ids, positions, packed_labels, params = qwen4_exp_pack_sequences(
        ids,
        labels,
        cp_size=2,
        pad_multiple=4,
        pad_token_id=99,
    )

    assert packed_ids.shape == (1, 16)
    assert params.cu_seqlens_q.tolist() == [0, 5, 12]
    assert params.cu_seqlens_q_padded.tolist() == [0, 5, 16]
    assert positions[0, :12].tolist() == list(range(5)) + list(range(7))
    assert packed_ids[0, :12].tolist() == ids[0].tolist() + ids[1].tolist()
    assert packed_ids[0, 12:].tolist() == [99] * 4
    assert packed_labels[0, 12:].tolist() == [-100] * 4


def _slicing_ple_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        group = dist.group.WORLD
        tokens = (torch.arange(SEQUENCE_LENGTH) + 2).unsqueeze(0)
        labels = tokens + 1
        loss_mask = torch.ones_like(tokens, dtype=torch.float32)
        position_ids = torch.arange(SEQUENCE_LENGTH).unsqueeze(0)
        padding_mask = (torch.arange(SEQUENCE_LENGTH) >= 13).unsqueeze(0)
        local = qwen4_exp_cp_chunk_data(
            rank,
            world_size,
            group,
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            global_padding_mask=padding_mask,
        )
        local_length = SEQUENCE_LENGTH // world_size
        sequence_slice = slice(rank * local_length, (rank + 1) * local_length)
        torch.testing.assert_close(local[0], tokens[:, sequence_slice])
        torch.testing.assert_close(local[1], labels[:, sequence_slice])
        torch.testing.assert_close(local[2], loss_mask[:, sequence_slice])
        torch.testing.assert_close(local[3], position_ids[:, sequence_slice])
        context = local[4]
        torch.testing.assert_close(
            context.local_attention_mask,
            padding_mask[:, sequence_slice].logical_not(),
        )

        local_values = torch.arange(SEQUENCE_LENGTH, dtype=torch.float32)[
            sequence_slice
        ].view(1, local_length, 1).requires_grad_(True)
        halo = qwen4_exp_cp_left_halo(local_values, context, history=3)
        expected_halo = (
            torch.zeros(1, 3, 1)
            if rank == 0
            else torch.arange(local_length - 3, local_length, dtype=torch.float32).view(
                1, 3, 1
            )
        )
        torch.testing.assert_close(halo, expected_halo)
        halo.sum().backward()
        expected_grad = torch.zeros_like(local_values)
        if rank == 0:
            expected_grad[:, -3:] = 1
        torch.testing.assert_close(local_values.grad, expected_grad)

        config = _config()
        torch.manual_seed(5)
        reference = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0).eval()
        sharded = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0).eval()
        sharded.load_state_dict(reference.state_dict())
        dense_weight = reference.ple_embedding.ngram_embedding.weight.detach().clone()
        sharded_embedding = Qwen4ExpEngramEmbedding(
            config,
            config.ple_embed_dim,
            layer_idx=1,
            process_group=group,
        )
        table = sharded_embedding.ngram_embedding
        table.weight.data.copy_(
            dense_weight[table.global_row_start : table.global_row_start + table.local_rows]
        )
        sharded.ple_embedding = sharded_embedding
        torch.manual_seed(8)
        hidden_states = torch.randn(
            1, SEQUENCE_LENGTH, config.hidden_size * config.hc_count
        )
        upstream = torch.randn_like(hidden_states)
        valid_mask = padding_mask.logical_not()
        full_hidden = hidden_states.clone().requires_grad_(True)
        expected = reference(
            full_hidden,
            tokens,
            None,
            conv_mask=valid_mask,
        )
        expected.backward(upstream)

        local_hidden = hidden_states[:, sequence_slice].clone().requires_grad_(True)
        got = sharded(
            local_hidden,
            tokens[:, sequence_slice],
            None,
            conv_mask=valid_mask[:, sequence_slice],
            cp_context=context,
        )
        got.backward(upstream[:, sequence_slice])
        torch.testing.assert_close(got, expected[:, sequence_slice], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            local_hidden.grad,
            full_hidden.grad[:, sequence_slice],
            rtol=1e-5,
            atol=1e-6,
        )

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            assert parameter.grad is not None
            reference_gradient = reference_parameters[name].grad
            assert reference_gradient is not None
            if name.endswith("ple_embedding.ngram_embedding.weight"):
                expected_gradient = reference_gradient[
                    table.global_row_start : table.global_row_start + table.local_rows
                ]
            else:
                dist.all_reduce(parameter.grad, group=group)
                expected_gradient = reference_gradient
            torch.testing.assert_close(
                parameter.grad, expected_gradient, rtol=1e-5, atol=1e-6
            )
    finally:
        dist.destroy_process_group()


def test_contiguous_slicing_halo_and_ple_match_full_sequence() -> None:
    _run_workers(_slicing_ple_worker)


def _qsa_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        group = dist.group.WORLD
        config = _config()
        torch.manual_seed(11)
        reference = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
        torch.manual_seed(11)
        sharded = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
        sharded.load_state_dict(reference.state_dict())

        torch.manual_seed(12)
        hidden = torch.randn(1, SEQUENCE_LENGTH, config.hidden_size)
        full_hidden = hidden.clone().requires_grad_(True)
        local_length = SEQUENCE_LENGTH // world_size
        start = rank * local_length
        sequence_slice = slice(start, start + local_length)
        local_hidden = hidden[:, sequence_slice].clone().requires_grad_(True)

        rotary_dim = config.head_dim // 4
        full_cos = torch.ones(1, SEQUENCE_LENGTH, rotary_dim)
        full_sin = torch.zeros_like(full_cos)
        query_positions = torch.arange(SEQUENCE_LENGTH)
        key_positions = torch.arange(SEQUENCE_LENGTH)
        visible = key_positions.view(1, -1) <= query_positions.view(-1, 1)
        full_mask = visible.view(1, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH)

        expected, _ = reference(
            full_hidden,
            (full_cos, full_sin),
            full_mask,
        )
        expected.sum().backward()

        context = _context(rank, group)
        local_mask = full_mask[
            :, :, sequence_slice, sequence_slice
        ].contiguous()
        got, _ = sharded(
            local_hidden,
            (
                full_cos[:, sequence_slice].contiguous(),
                full_sin[:, sequence_slice].contiguous(),
            ),
            local_mask,
            cp_context=context,
        )
        got.sum().backward()

        torch.testing.assert_close(got, expected[:, sequence_slice], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            local_hidden.grad,
            full_hidden.grad[:, sequence_slice],
            rtol=1e-5,
            atol=1e-6,
        )
        reference_params = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, group=group)
            torch.testing.assert_close(
                parameter.grad,
                reference_params[name].grad,
                rtol=1e-5,
                atol=1e-6,
            )
    finally:
        dist.destroy_process_group()


def test_qsa_cp_forward_backward_matches_full_sequence() -> None:
    _run_workers(_qsa_worker)


def test_packed_qsa_and_ple_match_independent_documents() -> None:
    boundaries = torch.tensor([0, 5, 13, 16])
    config = _config()
    torch.manual_seed(31)
    packed_attention = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
    independent_attention = Qwen4ExpQSAAttention(
        config, layer_idx=3, attn_backend="dense"
    )
    independent_attention.load_state_dict(packed_attention.state_dict())

    torch.manual_seed(32)
    hidden_values = torch.randn(1, 16, config.hidden_size)
    upstream = torch.randn_like(hidden_values)
    rotary_dim = config.head_dim // 4
    cos = torch.ones(1, 16, rotary_dim)
    sin = torch.zeros_like(cos)
    packed_hidden = hidden_values.clone().requires_grad_(True)
    packed_context = Qwen4ExpCPContext(
        group=None,
        rank=0,
        size=1,
        global_input_ids=(torch.arange(16) + 2).unsqueeze(0),
        global_padding_mask=torch.zeros(1, 16, dtype=torch.bool),
        local_sequence_start=0,
        local_sequence_length=16,
        global_cu_seqlens=boundaries,
    )
    packed_output, _ = packed_attention(
        packed_hidden,
        (cos, sin),
        None,
        cp_context=packed_context,
    )
    packed_output.backward(upstream)

    independent_hidden = hidden_values.clone().requires_grad_(True)
    document_outputs = []
    for start, end in zip(boundaries.tolist(), boundaries[1:].tolist()):
        length = end - start
        causal = torch.ones(length, length, dtype=torch.bool).tril().view(
            1, 1, length, length
        )
        output, _ = independent_attention(
            independent_hidden[:, start:end],
            (cos[:, start:end], sin[:, start:end]),
            causal,
        )
        document_outputs.append(output)
    independent_output = torch.cat(document_outputs, dim=1)
    independent_output.backward(upstream)

    torch.testing.assert_close(packed_output, independent_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        packed_hidden.grad, independent_hidden.grad, rtol=1e-5, atol=1e-6
    )
    independent_parameters = dict(independent_attention.named_parameters())
    for name, parameter in packed_attention.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad,
                independent_parameters[name].grad,
                rtol=1e-5,
                atol=1e-6,
            )

    torch.manual_seed(33)
    packed_ple = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0).eval()
    dense_weight = packed_ple.ple_embedding.ngram_embedding.weight.detach().clone()
    packed_ple.ple_embedding = Qwen4ExpEngramEmbedding(
        config,
        config.ple_embed_dim,
        layer_idx=1,
        process_group=None,
    )
    packed_ple.ple_embedding.ngram_embedding.weight.data.copy_(dense_weight)
    ple_hidden_values = torch.randn(1, 16, config.hidden_size * config.hc_count)
    token_ids = torch.arange(20, 36).unsqueeze(0)
    ple_context = Qwen4ExpCPContext(
        group=None,
        rank=0,
        size=1,
        global_input_ids=token_ids,
        global_padding_mask=torch.zeros_like(token_ids, dtype=torch.bool),
        local_sequence_start=0,
        local_sequence_length=16,
        global_cu_seqlens=boundaries,
    )
    ple_upstream = torch.randn_like(ple_hidden_values)
    packed_ple_hidden = ple_hidden_values.clone().requires_grad_(True)
    got_ple = packed_ple(
        packed_ple_hidden,
        token_ids,
        None,
        conv_mask=torch.ones_like(token_ids, dtype=torch.bool),
        cp_context=ple_context,
    )
    got_ple.backward(ple_upstream)
    packed_ple_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in packed_ple.named_parameters()
        if parameter.grad is not None
    }
    packed_ple.zero_grad(set_to_none=True)

    independent_ple_hidden = ple_hidden_values.clone().requires_grad_(True)
    expected_ple = torch.cat(
        [
            packed_ple(
                independent_ple_hidden[:, start:end],
                token_ids[:, start:end],
                None,
                conv_mask=torch.ones(1, end - start, dtype=torch.bool),
            )
            for start, end in zip(boundaries.tolist(), boundaries[1:].tolist())
        ],
        dim=1,
    )
    expected_ple.backward(ple_upstream)
    torch.testing.assert_close(got_ple, expected_ple, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        packed_ple_hidden.grad, independent_ple_hidden.grad, rtol=1e-5, atol=1e-6
    )
    for name, parameter in packed_ple.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(
                packed_ple_gradients[name], parameter.grad, rtol=1e-5, atol=1e-6
            )


def _packed_qsa_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        config = _config()
        boundaries = torch.tensor([0, 5, 13])
        torch.manual_seed(41)
        reference = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
        sharded = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
        sharded.load_state_dict(reference.state_dict())

        torch.manual_seed(42)
        hidden_values = torch.randn(1, SEQUENCE_LENGTH, config.hidden_size)
        upstream = torch.randn_like(hidden_values)
        rotary_dim = config.head_dim // 4
        cos = torch.ones(1, SEQUENCE_LENGTH, rotary_dim)
        sin = torch.zeros_like(cos)
        padding_mask = (torch.arange(SEQUENCE_LENGTH) >= int(boundaries[-1])).unsqueeze(0)
        full_context = Qwen4ExpCPContext(
            group=None,
            rank=0,
            size=1,
            global_input_ids=(torch.arange(SEQUENCE_LENGTH) + 2).unsqueeze(0),
            global_padding_mask=padding_mask,
            local_sequence_start=0,
            local_sequence_length=SEQUENCE_LENGTH,
            global_cu_seqlens=boundaries,
        )
        full_hidden = hidden_values.clone().requires_grad_(True)
        expected, _ = reference(
            full_hidden,
            (cos, sin),
            None,
            cp_context=full_context,
        )
        expected.backward(upstream)

        local_length = SEQUENCE_LENGTH // world_size
        start = rank * local_length
        sequence_slice = slice(start, start + local_length)
        context = Qwen4ExpCPContext(
            group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            global_input_ids=full_context.global_input_ids,
            global_padding_mask=padding_mask,
            local_sequence_start=start,
            local_sequence_length=local_length,
            global_cu_seqlens=boundaries,
        )
        local_hidden = hidden_values[:, sequence_slice].clone().requires_grad_(True)
        got, _ = sharded(
            local_hidden,
            (cos[:, sequence_slice], sin[:, sequence_slice]),
            None,
            cp_context=context,
        )
        got.backward(upstream[:, sequence_slice])

        torch.testing.assert_close(got, expected[:, sequence_slice], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            local_hidden.grad,
            full_hidden.grad[:, sequence_slice],
            rtol=1e-5,
            atol=1e-6,
        )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(
                parameter.grad,
                reference_parameters[name].grad,
                rtol=1e-5,
                atol=1e-6,
            )
    finally:
        dist.destroy_process_group()


def test_packed_qsa_cp2_matches_packed_cp1() -> None:
    _run_workers(_packed_qsa_worker)


def _gdn_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        device = torch.device("cuda", rank)
        config = _gdn_config()
        torch.manual_seed(21)
        reference = Qwen4ExpTextGatedDeltaNet(config, layer_idx=1).to(
            device=device, dtype=torch.bfloat16
        )
        sharded = Qwen4ExpTextGatedDeltaNet(config, layer_idx=1).to(
            device=device, dtype=torch.bfloat16
        )
        sharded.load_state_dict(reference.state_dict())
        for reference_parameter, sharded_parameter in zip(
            reference.parameters(), sharded.parameters()
        ):
            dist.broadcast(reference_parameter.data, src=0)
            dist.broadcast(sharded_parameter.data, src=0)

        sequence_length = 32
        local_length = sequence_length // world_size
        sequence_slice = slice(rank * local_length, (rank + 1) * local_length)
        torch.manual_seed(22)
        hidden = torch.randn(
            1,
            sequence_length,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        upstream_grad = torch.randn_like(hidden) / sequence_length

        full_hidden = hidden.clone().requires_grad_(True)
        expected = reference(
            full_hidden,
            attention_mask=torch.ones(
                1, sequence_length, dtype=torch.bool, device=device
            ),
        )
        (expected * upstream_grad).sum().backward()

        local_hidden = hidden[:, sequence_slice].clone().requires_grad_(True)
        positions = torch.arange(sequence_length, device=device)
        context = Qwen4ExpCPContext(
            group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            global_input_ids=(positions + 2).unsqueeze(0),
            global_padding_mask=torch.zeros(
                1, sequence_length, dtype=torch.bool, device=device
            ),
            local_sequence_start=rank * local_length,
            local_sequence_length=local_length,
        )
        got = sharded(
            local_hidden,
            attention_mask=torch.ones(
                1, local_length, dtype=torch.bool, device=device
            ),
            cp_context=context,
        )
        (got * upstream_grad[:, sequence_slice]).sum().backward()

        torch.testing.assert_close(
            got,
            expected[:, sequence_slice],
            rtol=5e-2,
            atol=5e-2,
        )
        torch.testing.assert_close(
            local_hidden.grad,
            full_hidden.grad[:, sequence_slice],
            rtol=8e-2,
            atol=5e-3,
        )
        reference_parameters = dict(reference.named_parameters())
        worst_relative_error = (0.0, "")
        worst_absolute_error = (0.0, "")
        for name, parameter in sharded.named_parameters():
            if parameter.grad is None:
                assert reference_parameters[name].grad is None
                continue
            parameter_grad = parameter.grad.detach().float().clone()
            dist.all_reduce(parameter_grad)
            reference_grad = reference_parameters[name].grad.float()
            difference = parameter_grad - reference_grad
            relative_error = (
                difference.norm() / reference_grad.norm().clamp_min(1e-8)
            ).item()
            absolute_error = difference.abs().max().item()
            worst_relative_error = max(worst_relative_error, (relative_error, name))
            worst_absolute_error = max(worst_absolute_error, (absolute_error, name))
        assert worst_relative_error[0] < 2e-2, (
            "GDN CP parameter-gradient relative L2 error is too large: "
            f"{worst_relative_error}"
        )
        assert worst_absolute_error[0] < 2e-2, (
            "GDN CP parameter-gradient maximum error is too large: "
            f"{worst_absolute_error}"
        )

        reference.zero_grad(set_to_none=True)
        sharded.zero_grad(set_to_none=True)
        packed_boundaries = torch.tensor([0, 7, 19, 29], device=device)
        torch.manual_seed(23)
        packed_hidden_values = torch.randn(
            1,
            sequence_length,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        packed_upstream = torch.randn_like(packed_hidden_values) / sequence_length
        packed_reference_hidden = packed_hidden_values.clone().requires_grad_(True)
        segment_boundaries = packed_boundaries.tolist() + [sequence_length]
        reference_outputs = []
        for segment_index, (segment_start, segment_end) in enumerate(
            zip(segment_boundaries, segment_boundaries[1:])
        ):
            valid = segment_index < packed_boundaries.numel() - 1
            segment_mask = torch.full(
                (1, segment_end - segment_start),
                valid,
                dtype=torch.bool,
                device=device,
            )
            reference_outputs.append(
                reference(
                    packed_reference_hidden[:, segment_start:segment_end],
                    attention_mask=segment_mask,
                )
            )
        packed_expected = torch.cat(reference_outputs, dim=1)
        (packed_expected * packed_upstream).sum().backward()

        packed_local_hidden = packed_hidden_values[:, sequence_slice].clone().requires_grad_(True)
        packed_padding_mask = (
            torch.arange(sequence_length, device=device) >= int(packed_boundaries[-1])
        ).unsqueeze(0)
        packed_context = Qwen4ExpCPContext(
            group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            global_input_ids=(torch.arange(sequence_length, device=device) + 2).unsqueeze(0),
            global_padding_mask=packed_padding_mask,
            local_sequence_start=rank * local_length,
            local_sequence_length=local_length,
            global_cu_seqlens=packed_boundaries,
        )
        packed_got = sharded(
            packed_local_hidden,
            attention_mask=packed_context.local_attention_mask,
            cp_context=packed_context,
        )
        (packed_got * packed_upstream[:, sequence_slice]).sum().backward()

        torch.testing.assert_close(
            packed_got,
            packed_expected[:, sequence_slice],
            rtol=5e-2,
            atol=5e-2,
        )
        torch.testing.assert_close(
            packed_local_hidden.grad,
            packed_reference_hidden.grad[:, sequence_slice],
            rtol=8e-2,
            atol=5e-3,
        )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            if parameter.grad is None:
                assert reference_parameters[name].grad is None
                continue
            parameter_grad = parameter.grad.detach().float().clone()
            dist.all_reduce(parameter_grad)
            torch.testing.assert_close(
                parameter_grad,
                reference_parameters[name].grad.float(),
                rtol=2e-2,
                atol=2e-2,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE,
    reason="requires two CUDA devices for FLA GatedDeltaNet CP",
)
def test_gated_delta_net_cp_forward_backward_matches_full_sequence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = str(pathlib.Path(tmpdir) / "nccl_init")
        mp.start_processes(
            _gdn_worker,
            args=(WORLD_SIZE, init_file),
            nprocs=WORLD_SIZE,
            join=True,
            start_method="spawn",
        )
