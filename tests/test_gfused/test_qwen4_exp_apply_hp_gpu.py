"""`apply_hp` 的 GPU 分布式测试：EP + FSDP2 包裹后能前向、反向、裁剪梯度。

需要 >=2 张卡（NCCL）。CPU 侧的模块替换/切片/激活重算策略已在
test_qwen4_exp_hp.py 覆盖，这里专门验只能上卡才能验的部分：

- `fully_shard` 能同时处理三种参数：全局 mesh 上的稠密参数、`ep_fsdp` 子 mesh 上的
  expert 参数、以及**被 ignored_params 排除**的 Engram 表；
- 前向/反向能跑通且数值有限；
- EP + FSDP2 下的 loss 与单卡稠密参考一致（EP 是精确的）；
- `clip_grad_norm_` 能跨 mesh 工作，并把 Engram 那份局部梯度也计入总范数。
"""
import datetime
import os
import pathlib
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from gpatch_v4.models.qwen4_exp import Qwen4ExpTextConfig
from gpatch_v4.models.qwen4_exp.hp import Qwen4ExpHpForCausalLM, apply_hp

NUM_EXPERTS = 8

TINY_KWARGS = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=8,
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
    num_experts=NUM_EXPERTS,
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

requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="需要至少 2 张 GPU",
)


def _config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(**TINY_KWARGS)


def _build_reference_model(seed: int = 0) -> Qwen4ExpHpForCausalLM:
    """在 CPU 上用 fp32 建一份带真实权重的模型（所有 rank 结果一致）。"""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(seed)
        model = Qwen4ExpHpForCausalLM(_config())
    finally:
        torch.set_default_dtype(prev_dtype)
    return model


def _apply_hp_worker(rank: int, world_size: int, init_file: str, ep_size: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=180),
    )
    try:
        config = _config()
        model = _build_reference_model().to("cuda")
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // ep_size, ep_size),
            mesh_dim_names=("ep_fsdp", "ep"),
        )

        model = apply_hp(model, mesh, attn_backend="flex", ep_backend="eager")

        assert model._ep_size == ep_size
        assert hasattr(model, "clip_grad_norm_")

        engram_module = model.model.layers[1].ple.ple_embedding.ngram_embedding
        engram_weight = engram_module.weight
        # 表被 FSDP 的 ignored_params 排除（按行 ownership 已经分好了），但仍要包成
        # DTensor：否则优化器一组里混 DTensor 和 plain tensor，AdamW 的 foreach 路径会报
        # "got mixed torch.Tensor and DTensor"（GPU 上真实踩过）。
        assert isinstance(engram_weight, DTensor), "Engram 没有被包成 DTensor"
        assert engram_weight.placements[0].is_shard(0)
        # 全局形状是整表；本地只有自己那一段
        assert engram_weight.shape[0] == (
            engram_module.local_rows * world_size
        ), engram_weight.shape
        assert engram_module.local_weight.shape[0] == engram_module.local_rows
        # 因为被 FSDP 排除，表拿不到 mp_policy 的 cast：master 留 fp32，输出自己转 bf16。
        # 少了这一步，第一个 projection 会报 float != c10::BFloat16（也是真实踩过的）。
        assert engram_module.local_weight.dtype == torch.float32, "Engram master 应保持 fp32"
        assert engram_module.output_dtype == torch.bfloat16, "Engram 没有设置输出 cast"

        model.train()
        torch.manual_seed(100 + rank)
        input_ids = torch.randint(0, config.vocab_size, (1, 24), device="cuda")

        output = model(input_ids=input_ids)
        assert torch.isfinite(output.logits).all(), "logits 出现 NaN/Inf"

        loss = output.logits.float().pow(2).mean()
        loss.backward()

        missing = [
            name for name, param in model.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        # indexer 是冻结的（requires_grad=False），不该出现在这里
        assert not missing, f"这些可训练参数没有梯度: {missing[:8]}"
        assert engram_module.weight.grad is not None, "Engram 表没有梯度"

        total_norm = model.clip_grad_norm_(max_norm=1.0)
        assert total_norm > 0 and torch.isfinite(torch.tensor(total_norm)), total_norm
    finally:
        dist.destroy_process_group()


@requires_two_gpus
@pytest.mark.parametrize("ep_size", [1, 2])
def test_apply_hp_forward_backward_and_clip(ep_size: int) -> None:
    world_size = 2
    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = str(pathlib.Path(tmpdir) / "pg_init")
            mp.start_processes(
                _apply_hp_worker,
                args=(world_size, init_file, ep_size),
                nprocs=world_size,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)


def _ep_equivalence_worker(rank: int, world_size: int, init_file: str) -> None:
    """EP 是**精确**的：EP2 下的 logits 必须等于单卡稠密结果。

    这里刻意用 fp32 的 mp_policy 和 dense 注意力后端，把 bf16 和 flex 的数值噪声排除掉，
    只留下"EP 切分 + all-to-all 是否等价"这一个变量，容差才有意义。
    bf16 + flex 的路径由 test_apply_hp_forward_backward_and_clip 覆盖。
    """
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=180),
    )
    try:
        config = _config()
        reference = _build_reference_model().to("cuda").eval()
        torch.manual_seed(7)
        input_ids = torch.randint(0, config.vocab_size, (1, 24), device="cuda")
        with torch.no_grad():
            expected = reference(input_ids=input_ids).logits.float()

        model = _build_reference_model().to("cuda")
        mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size // 2, 2), mesh_dim_names=("ep_fsdp", "ep")
        )
        model = apply_hp(
            model,
            mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32, reduce_dtype=torch.float32
            ),
            attn_backend="dense",
            ep_backend="eager",
        )
        model.eval()
        with torch.no_grad():
            got = model(input_ids=input_ids).logits.float()

        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    finally:
        dist.destroy_process_group()


@requires_two_gpus
def test_ep_wrapped_forward_matches_dense_reference() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = str(pathlib.Path(tmpdir) / "pg_init")
        mp.start_processes(
            _ep_equivalence_worker,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
