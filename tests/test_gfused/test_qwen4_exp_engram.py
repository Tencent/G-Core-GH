"""Engram（n-gram embedding）owner-sharded 表的测试。

分两块：
1. 单进程 CPU：hash 沿用上游（子类只换查表），验证 world_size=1 退化为普通查表、
   真实配置下的表形状、以及 cache 被拒绝。
2. 多进程 gloo：owner-sharded 查表必须等于整表查表；**梯度只能落在 owner 上**，
   且要等于把所有 rank 的请求合起来对整表求梯度后的对应切片。gloo 走 CPU，
   所以这部分本地就能验，不需要 GPU。
"""
import datetime
import os
import pathlib
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from gpatch_v4.models.qwen4_exp import Qwen4ExpTextConfig
from gpatch_v4.models.qwen4_exp.engram import (
    OwnerShardedNGramEmbedding,
    Qwen4ExpEngramEmbedding,
)
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

# 真实 checkpoint 的 Engram 表形状（NVIDIA 文档 + 实际 index.json 双重确认）
RELEASED_NGRAM_ROWS = 320_001_536
RELEASED_NGRAM_WIDTH = 160

TINY_KWARGS = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=16,
    indexer_compress_ratio=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=512,
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


def _config(**overrides) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(**{**TINY_KWARGS, **overrides})


# ---------------------------------------------------------------------------
# 单进程
# ---------------------------------------------------------------------------


def test_single_rank_lookup_equals_plain_embedding() -> None:
    torch.manual_seed(0)
    rows, dim = 256, 8
    module = OwnerShardedNGramEmbedding(rows, dim)
    module.weight.data.normal_()
    assert module.world_size == 1 and module.local_rows == rows

    ids = torch.randint(0, rows, (2, 5, 3))
    got = module(ids)
    expected = F.embedding(ids, module.weight)
    assert got.shape == (2, 5, 3, dim)
    torch.testing.assert_close(got, expected)


def test_subclass_reproduces_upstream_hashing_and_lookup() -> None:
    """子类只替换查表，hash 结果必须与上游逐字节一致。"""
    torch.manual_seed(0)
    config = _config()
    upstream = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=1).eval()
    ours = Qwen4ExpEngramEmbedding(config, config.ple_embed_dim, layer_idx=1).eval()

    upstream.ngram_embedding.weight.data.normal_()
    ours.ngram_embedding.weight.data.copy_(upstream.ngram_embedding.weight.data)

    input_ids = torch.randint(0, config.vocab_size, (2, 17))
    with torch.no_grad():
        expected = upstream(input_ids, None)
        got = ours(input_ids, None)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_global_hash_local_slice_matches_full_engram() -> None:
    torch.manual_seed(4)
    config = _config()
    module = Qwen4ExpEngramEmbedding(config, config.ple_embed_dim, layer_idx=1).eval()
    module.ngram_embedding.weight.data.normal_()
    input_ids = torch.randint(2, config.vocab_size, (2, 20))
    input_ids[:, 7] = config.eos_token_id

    with torch.no_grad():
        expected = module(input_ids, None)[:, 8:16]
        got = module.forward_global_slice(
            input_ids,
            sequence_start=8,
            sequence_end=16,
        )

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_hash_head_layout_matches_the_released_table() -> None:
    """真实配置：16 个 hash 头、宽 160、padded 到 320_001_536 行。"""
    config = _config(
        vocab_size=248320,
        ple_embed_dim=2560,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
    )
    # 51.2B 参数：只能在 meta 上建
    with torch.device("meta"):
        module = Qwen4ExpEngramEmbedding(config, config.ple_embed_dim, layer_idx=1)

    assert module.ngram_heads == 16  # (3-1) * 8：8 个 bigram + 8 个 trigram
    assert module.ngram_embedding.weight.shape == (RELEASED_NGRAM_ROWS, RELEASED_NGRAM_WIDTH)
    assert RELEASED_NGRAM_WIDTH * module.ngram_heads == config.ple_embed_dim
    # 每个头用一个不同的、> base-1 的素数
    assert len(set(module.head_vocab_sizes)) == 16
    assert all(size >= config.ngram_vocab_size_base for size in module.head_vocab_sizes)
    assert module.head_offsets[0] == 0
    assert module.total_vocab_size == sum(module.head_vocab_sizes)
    assert module.ngram_embedding.weight.shape[0] % 128 == 0


def test_output_dtype_casts_values_but_keeps_fp32_master() -> None:
    """表被 FSDP 的 ignored_params 排除，拿不到 mp_policy 的 bf16 cast。

    所以它必须自己把**查出来的值**转成下游的计算 dtype，否则第一个 projection 就会报
    `expected mat1 and mat2 to have the same dtype: float != c10::BFloat16`
    （这是 GPU 上真实踩到的错）。同时 master 必须留在 fp32 给优化器用，
    梯度也要以 fp32 回到 master。
    """
    torch.manual_seed(0)
    module = OwnerShardedNGramEmbedding(64, 8)
    module.weight.data.normal_()
    assert module.weight.dtype == torch.float32
    ids = torch.randint(0, 64, (2, 3))

    assert module(ids).dtype == torch.float32  # 默认不转换

    module.output_dtype = torch.bfloat16
    values = module(ids)
    assert values.dtype == torch.bfloat16
    assert module.weight.dtype == torch.float32, "master 被就地转换了"

    values.float().sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.dtype == torch.float32, "梯度不是 fp32，会和 master 不匹配"


def test_cache_is_rejected() -> None:
    config = _config()
    module = Qwen4ExpEngramEmbedding(config, config.ple_embed_dim, layer_idx=1)
    with pytest.raises(NotImplementedError, match="KV cache"):
        module(torch.zeros(1, 4, dtype=torch.long), past_key_values=object())


def test_non_divisible_row_count_is_rejected(monkeypatch) -> None:
    # world_size=1 永远能整除，所以伪造一个 size=3 的 group 来触发校验
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 3)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    with pytest.raises(ValueError, match="divisible"):
        OwnerShardedNGramEmbedding(10, 4)


# ---------------------------------------------------------------------------
# 多进程（gloo / CPU）
# ---------------------------------------------------------------------------

WORLD_SIZE = 2
ROWS, DIM = 64, 8


def _make_ids(world_size: int) -> list:
    """每个 rank 一份固定的 id，故意覆盖：跨 shard 边界、重复 id、只属于对端的 id。"""
    torch.manual_seed(1234)
    per_rank = []
    local_rows = ROWS // world_size
    for rank in range(world_size):
        own = torch.randint(rank * local_rows, (rank + 1) * local_rows, (6, ))
        other = torch.randint(0, ROWS, (10, ))
        boundary = torch.tensor([0, local_rows - 1, local_rows, ROWS - 1])
        duplicated = other[:3].repeat(2)
        per_rank.append(torch.cat([own, other, boundary, duplicated]).reshape(1, -1))
    return per_rank


def _owner_sharded_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        full_table = torch.randn(ROWS, DIM)  # 所有 rank 一致
        local_rows = ROWS // world_size
        row_slice = slice(rank * local_rows, (rank + 1) * local_rows)

        module = OwnerShardedNGramEmbedding(ROWS, DIM)
        assert module.world_size == world_size
        assert module.local_rows == local_rows
        assert module.global_row_start == rank * local_rows
        module.weight.data.copy_(full_table[row_slice])

        per_rank_ids = _make_ids(world_size)
        ids = per_rank_ids[rank]

        got = module(ids)
        expected = F.embedding(ids, full_table)
        torch.testing.assert_close(got, expected)

        # 用带位置权重的 loss，permutation 搞错就会被抓到
        weight = torch.arange(got.numel(), dtype=got.dtype).reshape(got.shape) + 1.0
        (got * weight).sum().backward()

        # 参考：把所有 rank 的请求合起来，对整表求梯度
        reference = full_table.clone().requires_grad_(True)
        total = 0.0
        for other_rank, other_ids in enumerate(per_rank_ids):
            values = F.embedding(other_ids, reference)
            other_weight = torch.arange(values.numel(), dtype=values.dtype).reshape(values.shape) + 1.0
            total = total + (values * other_weight).sum()
        total.backward()

        assert module.weight.grad is not None
        torch.testing.assert_close(module.weight.grad, reference.grad[row_slice])
        # owner 之外的行不该被本 rank 碰到：本地梯度只覆盖自己那一段，
        # 而参考梯度在别处非零，说明确实发生了跨 rank 路由
        assert reference.grad.abs().sum() > module.weight.grad.abs().sum()
    finally:
        dist.destroy_process_group()


def _init_pg(rank: int, world_size: int, init_file: str) -> None:
    """带超时地建 gloo group。

    超时很重要：某个 rank 抛异常时，其余 rank 还卡在集合通信上，默认 30 分钟超时会让
    测试看起来是"挂住"而不是"失败"。
    """
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )


def _run_workers(fn, world_size: int) -> None:
    # 线程数降到 1：多个 worker 在同一台机器上跑，避免 CPU 线程超订。
    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        # fork 不能安全复用父进程已经启动过的 autograd 线程池。
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = str(pathlib.Path(tmpdir) / "pg_init")
            mp.start_processes(
                fn,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)


def test_owner_sharded_lookup_and_gradient_ownership() -> None:
    _run_workers(_owner_sharded_worker, 4)


def _engram_module_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        config = _config()
        torch.manual_seed(0)
        reference = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=1).eval()
        reference.ngram_embedding.weight.data.normal_()

        sharded = Qwen4ExpEngramEmbedding(config, config.ple_embed_dim, layer_idx=1).eval()
        rows = reference.ngram_embedding.weight.shape[0]
        local_rows = rows // world_size
        sharded.ngram_embedding.weight.data.copy_(
            reference.ngram_embedding.weight.data[rank * local_rows:(rank + 1) * local_rows]
        )

        # 每个 rank 喂不同的数据（模拟 DP），并且带 EOS 以覆盖 n-gram 上下文重置
        torch.manual_seed(100 + rank)
        input_ids = torch.randint(2, config.vocab_size, (2, 13))
        input_ids[0, 5] = config.eos_token_id
        input_ids[1, 9] = config.eos_token_id

        with torch.no_grad():
            expected = reference(input_ids, None)
            got = sharded(input_ids, None)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        assert got.shape == (2, 13, config.ple_embed_dim)
    finally:
        dist.destroy_process_group()


def test_sharded_engram_matches_full_table_across_ranks() -> None:
    _run_workers(_engram_module_worker, WORLD_SIZE)


def _dtensor_optimizer_worker(rank: int, world_size: int, init_file: str) -> None:
    """DTensor 表要兼容 AdamW，并撤销 FSDP loss 的 DP 乘数。

    这条复现的是真机上的报错：Engram 表被 FSDP 的 ignored_params 排除后仍是 plain
    tensor，而其他参数都是 DTensor；AdamW 默认 foreach=True 会把同 device/dtype 的参数
    打包调 `torch._foreach_mul_`，一组里混两种类型就会抛

        RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor
    """
    _init_pg(rank, world_size, init_file)
    try:
        from torch.distributed.tensor import DTensor

        from gpatch_v4.models.qwen4_exp.hp import _wrap_engram_as_dtensor

        module = OwnerShardedNGramEmbedding(ROWS, DIM)
        module.weight.data.normal_()
        container = torch.nn.Module()
        container.engram = module

        # 这里只断言"包成 DTensor 之后能跑"，不去断言"不包会炸"：那个报错依赖 torch
        # 版本（pod 上 2.11+cu129 会抛，本地 2.12 CPU 不抛），拿它当断言会很脆。
        # loss 为抵消 FSDP 平均会乘 DP size；owner 路由自己完成求和，却没有 FSDP
        # reduce-scatter，所以 apply_hp 注册的 hook 必须把这个乘数除回去。
        _wrap_engram_as_dtensor(container, world_size, world_size)
        assert module.local_weight.shape == (ROWS // world_size, DIM)

        per_rank_ids = _make_ids(world_size)
        ids = per_rank_ids[rank].to(module.local_weight.device)
        values = module(ids)
        (values.float().sum() * world_size).backward()
        assert module.weight.grad is not None

        expected = torch.zeros(ROWS, DIM)
        for requester_ids in per_rank_ids:
            expected.index_add_(
                0,
                requester_ids.reshape(-1),
                torch.ones(requester_ids.numel(), DIM),
            )
        row_start = rank * module.local_rows
        torch.testing.assert_close(
            module.weight.grad.to_local().cpu(),
            expected[row_start:row_start + module.local_rows],
        )

        mesh = module.weight.device_mesh
        other = torch.nn.Parameter(
            DTensor.from_local(
                torch.randn(4, DIM),
                device_mesh=mesh,
                placements=module.weight.placements,
            )
        )
        optimizer = torch.optim.AdamW([module.weight, other], lr=1e-3, weight_decay=0.1)
        other.grad = torch.zeros_like(other)
        optimizer.step()  # 不再抛异常
    finally:
        dist.destroy_process_group()


def test_dtensor_wrapped_table_works_with_foreach_adamw() -> None:
    _run_workers(_dtensor_optimizer_worker, WORLD_SIZE)
