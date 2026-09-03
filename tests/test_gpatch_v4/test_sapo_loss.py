"""Tests for SAPO's THD / Dynamic-CP-compatible legacy loss path."""
import types
from unittest.mock import patch

import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.mappings import reduce_from_context_parallel_region
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    is_seq_mean_rl_loss_fn,
    sapo_loss_func,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    _pack_sequences_by_keys,
    build_packed_microbatches_by_keys,
)


_REDUCE_METRICS_PATH = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)
_MPU_GET_DYN_CP_GROUP = (
    "gpatch_v4.training_backend.loss_factory.mpu.get_dynamic_data_context_parallel_groups"
)
_REDUCE_FROM_CP = (
    "gpatch_v4.training_backend.loss_factory.reduce_from_context_parallel_region"
)
_PACKED_KEYS = ["curr_log_probs", "prev_log_probs", "advantages", "response_mask"]


def _simulate_te_thd_local_indices(cu_seqlens_padded, cp_size, cp_rank):
    """Python mirror of TE ``thd_partition_indices_kernel`` local layout."""
    cu = [int(x) for x in cu_seqlens_padded.tolist()]
    local_cu = [c // cp_size for c in cu]
    local_len = cu[-1] // cp_size
    indices = []
    for token_id in range(local_len):
        seq_id = 0
        for i in range(len(local_cu) - 1):
            if local_cu[i] <= token_id < local_cu[i + 1]:
                seq_id = i
                break
        seq_len = local_cu[seq_id + 1] - local_cu[seq_id]
        index = token_id - local_cu[seq_id]
        offset = cp_rank if index < seq_len // 2 else (cp_size - 1) * 2 - cp_rank
        index = index + local_cu[seq_id] * cp_size + (seq_len // 2) * offset
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long)


def _shard_thd_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).index_select(0, indices).unsqueeze(0)


def _make_rl_samples(orig_and_pad_lens, seed: int = 11):
    """Build per-sample dicts in the shape gcore Dyn-CP packing expects.

    ``orig_and_pad_lens`` is a list of ``(original_seq_len, padded_seq_len)``.
    Pad tokens keep values but ``response_mask=0``, matching real reroute/pack.
    """
    torch.manual_seed(seed)
    samples = []
    for orig, pad in orig_and_pad_lens:
        assert pad >= orig
        curr = torch.randn(pad)
        prev = curr.detach() + 0.05 * torch.randn(pad)
        advantages = torch.randn(pad)
        mask = torch.zeros(pad)
        mask[:orig] = 1.0
        samples.append(
            {
                "curr_log_probs": curr,
                "prev_log_probs": prev,
                "advantages": advantages,
                "response_mask": mask,
                "original_seq_len": torch.tensor([orig], dtype=torch.int32),
                "padded_seq_len": torch.tensor([pad], dtype=torch.int32),
            }
        )
    return samples


def _pack_samples_with_gcore(samples, local_cp_size: int):
    """Use gcore's ``_pack_sequences_by_keys`` (same helper Dyn-CP schedule calls)."""
    return _pack_sequences_by_keys(
        samples,
        padded_lengths=torch.cat([s["padded_seq_len"].reshape(-1) for s in samples]),
        original_lengths=torch.cat([s["original_seq_len"].reshape(-1) for s in samples]),
        local_cp_size=torch.tensor(local_cp_size, dtype=torch.int32),
        dev=torch.device("cpu"),
        packed_keys=_PACKED_KEYS,
        cat_keys=[],
    )


def _pack_samples_via_microbatch_builder(samples, cp_size: int):
    """Use ``build_packed_microbatches_by_keys`` with CP-sibling sample-id groups.

    Mirrors Dyn-CP: every CP rank in the subgroup receives the same sample ids,
    so ``local_cp_size`` becomes ``cp_size``.
    """
    samples_with_id = {i: sample for i, sample in enumerate(samples)}
    ids = list(range(len(samples)))
    sample_id_groups = [[list(ids) for _ in range(cp_size)]]
    packed_list = build_packed_microbatches_by_keys(
        samples_with_id,
        sample_id_groups,
        dcp_rank=0,
        dev=torch.device("cpu"),
        is_dynamic_cp=True,
        packed_keys=_PACKED_KEYS,
        cat_keys=[],
    )
    return packed_list[0]


def _loss_tensors_from_packed(packed):
    curr = packed["curr_log_probs"].reshape(1, -1).detach().requires_grad_()
    prev = packed["prev_log_probs"].reshape(1, -1)
    advantages = packed["advantages"].reshape(1, -1)
    mask = packed["response_mask"].reshape(1, -1)
    cu = packed["cu_seqlens_padded"].to(dtype=torch.long)
    local_cp_size = int(packed["local_cp_size"].item())
    return curr, prev, advantages, mask, cu, local_cp_size


def _make_config(
    *,
    entropy_bonus: float = 0.0,
    kl_beta: float = 0.0,
):
    return types.SimpleNamespace(
        ppo=PpoConfig(
            loss_func="sapo",
            ppo_entropy_bonus=entropy_bonus,
            grpo_kl_loss_beta=kl_beta,
            sapo_tau_pos=1.0,
            sapo_tau_neg=1.25,
        ),
        debug=DebugConfig(),
    )


def _make_input(
    curr: torch.Tensor,
    prev: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    cu_seqlens_padded: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
    local_cp_size: int = 1,
) -> PolicyLossInput:
    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=None,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(curr) if entropy is None else entropy,
        cu_seqlens_padded=cu_seqlens_padded,
        calculate_per_token_loss=calculate_per_token_loss,
        local_cp_size=local_cp_size,
    )


def _collect_dyn_cp_reduce_payloads(
    *,
    config,
    curr: torch.Tensor,
    prev: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    cu: torch.Tensor,
    cp_size: int,
    calculate_per_token_loss: bool,
):
    """Capture each rank's actor ``stack([sums, counts])`` before CP SUM.

    One SAPO forward also reduces entropy (and maybe KL). Those later reduces
    are ignored: on/off checks inject only the actor CP SUM. Summing every
    reduce would double-count and halve seq-mean loss.
    """
    payloads = []

    for rank in range(cp_size):
        captured = []

        def capture_reduce(inp, group=None, _bucket=captured):
            _bucket.append(inp.detach().clone())
            return inp

        idx = _simulate_te_thd_local_indices(cu, cp_size, rank)
        with patch(_REDUCE_METRICS_PATH), patch(_MPU_GET_DYN_CP_GROUP, return_value=object()), \
                patch(_REDUCE_FROM_CP, side_effect=capture_reduce):
            sapo_loss_func(
                config,
                _make_input(
                    _shard_thd_tokens(curr.detach(), idx).requires_grad_(),
                    _shard_thd_tokens(prev, idx),
                    _shard_thd_tokens(advantages, idx),
                    _shard_thd_tokens(mask, idx),
                    cu_seqlens_padded=cu,
                    calculate_per_token_loss=calculate_per_token_loss,
                    local_cp_size=cp_size,
                ),
            )
        assert captured, f"rank {rank} never hit CP reduce"
        # First reduce is actor; subsequent are entropy (/KL).
        payloads.append(captured[0])
    return payloads


class _InjectedCpReduce(torch.autograd.Function):
    """Forward injects a precomputed CP SUM; backward stays local (MCore protocol)."""

    @staticmethod
    def forward(ctx, inp, reduced):
        return reduced.to(device=inp.device, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class TestSapoSeqMean:
    def test_dynamic_cp_reduce_keeps_local_backward(self):
        """DCP forward SUM must not add another CP all-reduce in backward."""
        values = torch.tensor([1.0, 2.0], requires_grad=True)
        dynamic_cp_group = object()

        def fake_all_reduce(tensor, group):
            assert group is dynamic_cp_group
            tensor.add_(3.0)

        with patch("gpatch_v4.core.mappings.torch.distributed.all_reduce", fake_all_reduce):
            reduced = reduce_from_context_parallel_region(values, group=dynamic_cp_group)
            reduced.sum().backward()

        # The forward saw the reduced values, while the custom mapping's
        # backward preserves this rank's local token-gradient contribution.
        assert torch.equal(reduced, torch.tensor([4.0, 5.0]))
        assert torch.equal(values.grad, torch.ones_like(values))

    def test_registered_for_global_seq_mean_normalization(self):
        assert is_seq_mean_rl_loss_fn(sapo_loss_func)

    @patch(_REDUCE_METRICS_PATH)
    def test_thd_backward_is_pack_partition_invariant(self, mock_reduce):
        """A Dynamic-CP THD pack has the same SAPO numerator when split."""
        torch.manual_seed(7)
        lengths = [3, 5, 2]
        total = sum(lengths)
        curr = torch.randn(1, total, requires_grad=True)
        prev = curr.detach() + 0.1 * torch.randn(1, total)
        advantages = torch.randn(1, total)
        mask = torch.ones(1, total)

        def run(start, end, boundaries):
            return sapo_loss_func(
                _make_config(),
                _make_input(
                    curr[:, start:end],
                    prev[:, start:end],
                    advantages[:, start:end],
                    mask[:, start:end],
                    cu_seqlens_padded=torch.tensor(boundaries),
                ),
            )[0]

        full = run(0, total, [0, 3, 8, 10])
        first = run(0, 3, [0, 3])
        second = run(3, total, [0, 5, 7])
        assert torch.allclose(full, first + second, atol=1e-6)

    @patch(_REDUCE_METRICS_PATH)
    def test_bshd_and_thd_have_equal_loss_and_valid_token_gradients(self, mock_reduce):
        """THD boundaries preserve SAPO's per-response aggregation."""
        config = _make_config()
        bshd_curr = torch.tensor(
            [[-0.2, -0.4, 0.0], [-0.1, -0.3, -0.5]],
            requires_grad=True,
        )
        bshd_prev = torch.tensor([[-0.3, -0.35, 0.0], [-0.2, -0.25, -0.6]])
        bshd_advantages = torch.tensor([[1.0, -0.5, 0.0], [0.0, 0.25, 0.75]])
        bshd_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        bshd_input = _make_input(
            bshd_curr,
            bshd_prev,
            bshd_advantages,
            bshd_mask,
        )

        thd_curr = torch.zeros(1, 8)
        thd_prev = torch.zeros(1, 8)
        thd_advantages = torch.zeros(1, 8)
        thd_mask = torch.zeros(1, 8)
        thd_curr[0, 0:2] = bshd_curr.detach()[0, 0:2]
        thd_curr[0, 4:7] = bshd_curr.detach()[1, 0:3]
        thd_curr.requires_grad_()
        thd_prev[0, 0:2] = bshd_prev[0, 0:2]
        thd_prev[0, 4:7] = bshd_prev[1, 0:3]
        thd_advantages[0, 0:2] = bshd_advantages[0, 0:2]
        thd_advantages[0, 4:7] = bshd_advantages[1, 0:3]
        thd_mask[0, 0:2] = bshd_mask[0, 0:2]
        thd_mask[0, 4:7] = bshd_mask[1, 0:3]
        thd_input = _make_input(
            thd_curr,
            thd_prev,
            thd_advantages,
            thd_mask,
            cu_seqlens_padded=torch.tensor([0, 4, 8]),
        )

        bshd_loss, _ = sapo_loss_func(config, bshd_input)
        thd_loss, _ = sapo_loss_func(config, thd_input)
        assert torch.allclose(thd_loss, bshd_loss, atol=1e-6)

        bshd_loss.backward()
        thd_loss.backward()
        assert torch.allclose(thd_curr.grad[0, 0:2], bshd_curr.grad[0, 0:2], atol=1e-6)
        assert torch.allclose(thd_curr.grad[0, 4:7], bshd_curr.grad[1, 0:3], atol=1e-6)
        assert torch.count_nonzero(thd_curr.grad[0, [2, 3, 7]]) == 0

    @patch(_REDUCE_METRICS_PATH)
    def test_seq_mean_gives_each_response_equal_weight(self, mock_reduce):
        """A long and short response contribute one sequence mean each."""
        curr = torch.zeros(1, 10, requires_grad=True)
        prev = torch.zeros_like(curr)
        advantages = torch.zeros_like(curr)
        advantages[0, :8] = 1.0
        advantages[0, 8:] = 5.0
        mask = torch.ones_like(curr)
        loss, metrics = sapo_loss_func(
            _make_config(),
            _make_input(
                curr,
                prev,
                advantages,
                mask,
                cu_seqlens_padded=torch.tensor([0, 8, 10]),
            ),
        )

        # At ratio=1, SAPO's gate is 2 / tau = 2.  The returned value is the
        # unnormalized numerator: -(2 * 1 + 2 * 5) = -12.
        assert torch.allclose(loss, torch.tensor(-12.0))
        assert torch.allclose(metrics["policy_loss"][1], torch.tensor(2.0))

    @patch(_REDUCE_METRICS_PATH)
    def test_per_token_mode_returns_token_sum(self, mock_reduce):
        """Per-token mode sums the soft-gated surrogate over valid tokens."""
        curr = torch.zeros(1, 3, requires_grad=True)
        prev = torch.zeros_like(curr)
        advantages = torch.ones_like(curr)
        mask = torch.ones_like(curr)
        loss, metrics = sapo_loss_func(
            _make_config(),
            _make_input(
                curr,
                prev,
                advantages,
                mask,
                calculate_per_token_loss=True,
            ),
        )

        # At ratio=1 and tau_pos=1, gate=2, so token sum is -2 * 3 = -6.
        assert torch.allclose(loss, torch.tensor(-6.0))
        assert torch.allclose(metrics["policy_loss"][1], torch.tensor(3.0))

    @patch(_REDUCE_METRICS_PATH)
    def test_per_token_mode_weights_tokens_not_sequences(self, mock_reduce):
        """Long responses contribute more under token-mean than seq-mean."""
        curr = torch.zeros(1, 10, requires_grad=True)
        prev = torch.zeros_like(curr)
        advantages = torch.zeros_like(curr)
        advantages[0, :8] = 1.0
        advantages[0, 8:] = 5.0
        mask = torch.ones_like(curr)
        loss, metrics = sapo_loss_func(
            _make_config(),
            _make_input(
                curr,
                prev,
                advantages,
                mask,
                cu_seqlens_padded=torch.tensor([0, 8, 10]),
                calculate_per_token_loss=True,
            ),
        )

        # gate=2: -(2*1*8 + 2*5*2) = -36 over 10 tokens, not -12 over 2 seqs.
        assert torch.allclose(loss, torch.tensor(-36.0))
        assert torch.allclose(metrics["policy_loss"][1], torch.tensor(10.0))


class TestSapoDynCpOnOff:
    """Dyn-CP off vs on using gcore's real pack helpers + TE CP shard layout.

    Pack path matches ``dynamic_cp_utils``:
      per-sample dicts → ``_pack_sequences_by_keys`` /
      ``build_packed_microbatches_by_keys`` → THD + ``cu_seqlens(_padded)``.
    """

    def _gcore_packed_batch(self, cp_size: int = 2, seed: int = 11, with_padding: bool = True):
        # padded_seq_len must be divisible by cp_size * 2 (TE requirement).
        if with_padding:
            # Real Dyn-CP pads; original < padded, mask covers only original.
            orig_and_pad = [(6, 8), (14, 16), (5, 8)]
        else:
            orig_and_pad = [(8, 8), (16, 16), (8, 8)]
        assert all(pad % (cp_size * 2) == 0 for _, pad in orig_and_pad)
        samples = _make_rl_samples(orig_and_pad, seed=seed)
        packed = _pack_samples_via_microbatch_builder(samples, cp_size)
        # Sanity: builder must recover cp_size from sibling sample-id groups.
        assert int(packed["local_cp_size"].item()) == cp_size
        assert packed["cu_seqlens"].tolist() != packed["cu_seqlens_padded"].tolist() or not with_padding
        # Direct pack helper must match the microbatch builder's THD layout.
        packed_direct = _pack_samples_with_gcore(samples, local_cp_size=cp_size)
        assert torch.equal(packed["cu_seqlens_padded"], packed_direct["cu_seqlens_padded"])
        assert torch.equal(packed["cu_seqlens"], packed_direct["cu_seqlens"])
        for key in _PACKED_KEYS:
            assert torch.equal(packed[key], packed_direct[key])
        return _loss_tensors_from_packed(packed)

    @patch(_REDUCE_METRICS_PATH)
    def test_seq_mean_dyn_cp_on_matches_off_loss_and_local_grads(self, mock_reduce):
        config = _make_config()
        curr, prev, advantages, mask, cu, cp_size = self._gcore_packed_batch()

        # Dyn-CP off: same gcore-packed THD, local_cp_size=1 (no CP shard).
        ref_loss, ref_metrics = sapo_loss_func(
            config,
            _make_input(
                curr,
                prev,
                advantages,
                mask,
                cu_seqlens_padded=cu,
                calculate_per_token_loss=False,
                local_cp_size=1,
            ),
        )
        ref_loss.backward()
        ref_grad = curr.grad.detach().clone()
        curr.grad = None

        payloads = _collect_dyn_cp_reduce_payloads(
            config=config,
            curr=curr,
            prev=prev,
            advantages=advantages,
            mask=mask,
            cu=cu,
            cp_size=cp_size,
            calculate_per_token_loss=False,
        )
        assert len(payloads) == cp_size
        reduced = sum(payloads)

        for rank in range(cp_size):
            idx = _simulate_te_thd_local_indices(cu, cp_size, rank)
            curr_r = _shard_thd_tokens(curr.detach(), idx).requires_grad_()
            call_n = {"n": 0}

            def inject_reduce(inp, group=None, _state=call_n):
                # First reduce is actor (inject CP SUM); later entropy/KL stay local.
                _state["n"] += 1
                if _state["n"] == 1:
                    return _InjectedCpReduce.apply(inp, reduced)
                return inp

            with patch(_MPU_GET_DYN_CP_GROUP, return_value=object()), \
                    patch(_REDUCE_FROM_CP, side_effect=inject_reduce):
                loss_r, metrics_r = sapo_loss_func(
                    config,
                    _make_input(
                        curr_r,
                        _shard_thd_tokens(prev, idx),
                        _shard_thd_tokens(advantages, idx),
                        _shard_thd_tokens(mask, idx),
                        cu_seqlens_padded=cu,
                        calculate_per_token_loss=False,
                        local_cp_size=cp_size,
                    ),
                )
            # After CP SUM, every rank holds the full seq-mean numerator.
            assert torch.allclose(loss_r, ref_loss, atol=1e-5)
            assert torch.allclose(metrics_r["policy_loss"][1], ref_metrics["policy_loss"][1])
            loss_r.backward()
            assert torch.allclose(
                curr_r.grad.reshape(-1),
                ref_grad.reshape(-1).index_select(0, idx),
                atol=1e-5,
            )

    @patch(_REDUCE_METRICS_PATH)
    def test_per_token_dyn_cp_shards_sum_to_off_loss_and_match_grads(self, mock_reduce):
        config = _make_config()
        curr, prev, advantages, mask, cu, cp_size = self._gcore_packed_batch(seed=17)

        ref_loss, ref_metrics = sapo_loss_func(
            config,
            _make_input(
                curr,
                prev,
                advantages,
                mask,
                cu_seqlens_padded=cu,
                calculate_per_token_loss=True,
                local_cp_size=1,
            ),
        )
        ref_loss.backward()
        ref_grad = curr.grad.detach().clone()
        curr.grad = None

        shard_losses = []
        shard_token_counts = []
        for rank in range(cp_size):
            idx = _simulate_te_thd_local_indices(cu, cp_size, rank)
            curr_r = _shard_thd_tokens(curr.detach(), idx).requires_grad_()
            mask_r = _shard_thd_tokens(mask, idx)
            # Per-token bwd uses local token-sum (no CP reduce). Sample metrics
            # still touch the dyn-CP group when local_cp_size>1, so stub it.
            with patch(_MPU_GET_DYN_CP_GROUP, return_value=object()), \
                    patch(_REDUCE_FROM_CP, side_effect=lambda inp, group=None: inp):
                loss_r, metrics_r = sapo_loss_func(
                    config,
                    _make_input(
                        curr_r,
                        _shard_thd_tokens(prev, idx),
                        _shard_thd_tokens(advantages, idx),
                        mask_r,
                        cu_seqlens_padded=cu,
                        calculate_per_token_loss=True,
                        local_cp_size=cp_size,
                    ),
                )
            shard_losses.append(loss_r)
            shard_token_counts.append(metrics_r["policy_loss"][1])
            loss_r.backward()
            assert torch.allclose(
                curr_r.grad.reshape(-1),
                ref_grad.reshape(-1).index_select(0, idx),
                atol=1e-5,
            )
            # Pad tokens contribute 0 to the local token count.
            assert torch.allclose(metrics_r["policy_loss"][1], mask_r.sum())

        assert torch.allclose(sum(shard_losses), ref_loss, atol=1e-5)
        assert torch.allclose(sum(shard_token_counts), ref_metrics["policy_loss"][1], atol=1e-5)
        assert torch.allclose(ref_metrics["policy_loss"][1], mask.sum())
