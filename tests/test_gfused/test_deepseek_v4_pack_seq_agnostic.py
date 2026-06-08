# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""逐一过 transformers 原版 DeepSeek-V4 的 module / function，判定其对 pack-seq 是否「无感知」。

判据（pack-seq agnostic）：序列 A,B,C 分别过 module 得 X,Y,Z；拼成 [A,B,C] 过同一
module 应得 [X,Y,Z]（逐 token 独立 → 相等/高度相近）。跨 token 交互的 module 不满足
此不变量（负向对照：packed 与单算显著不同）。

单进程单卡，load 真权重（bf16 + fp8 dequantize），抠出单个 module/function 直接调用。
MoE 用 `experts_implementation="eager"` 回到逐 expert 的 loop forward（默认 grouped_mm）。

modeling_deepseek_v4.py 全量覆盖表
----------------------------------
AGNOSTIC（逐 token，positive 测试，应满足不变量）:
  DeepseekV4RMSNorm                 -> test_rmsnorm_input_layernorm / test_rmsnorm_final_norm
  DeepseekV4UnweightedRMSNorm       -> test_unweighted_rmsnorm（q_b_norm）
  DeepseekV4GroupedLinear           -> test_grouped_linear（o_a_proj）
  DeepseekV4MLP                     -> test_mlp_shared_experts
  DeepseekV4Experts                 -> 经 SparseMoeBlock 覆盖（test_moe_block_*）
  DeepseekV4TopKRouter              -> test_topk_router
  DeepseekV4HashRouter              -> test_hash_router
  DeepseekV4SparseMoeBlock          -> test_moe_block_hash / test_moe_block_topk
  DeepseekV4HyperConnection         -> test_hyperconnection_attn_hc
  DeepseekV4HyperHead               -> test_hyperhead
  nn.Embedding (model.embed_tokens) -> test_embedding
  DeepseekV4RotaryEmbedding         -> 经 test_rope_sample_local_positions 覆盖
  apply_rotary_pos_emb              -> test_rope_sample_local_positions
  rotate_half                       -> test_rotate_half
  repeat_kv                         -> test_repeat_kv

AWARE（跨 token，negative 对照，应 NOT 满足不变量）:
  DeepseekV4HCACompressor           -> test_hca_compressor_is_pack_aware
  DeepseekV4CSACompressor           -> test_csa_compressor_is_pack_aware
  DeepseekV4Indexer                 -> test_indexer_is_pack_aware
  eager_attention_forward           -> 经 DeepseekV4Attention 覆盖
  DeepseekV4Attention               -> test_attention_is_pack_aware
  DeepseekV4DecoderLayer            -> test_decoder_layer_is_pack_aware

不适用 A,B,C 逐 token 判据（不测，说明原因）:
  DeepseekV4HCACache / DeepseekV4CSACache   -> 生成期有状态 cache（rolling window），非逐 token 变换
  DeepseekV4Model / DeepseekV4ForCausalLM   -> 整模型前向（含 attention），属 pack-seq 功能本身而非单元
  load_balancing_loss_func                  -> 跨 token 聚合标量 loss（reduction），非 seq→seq 变换
  compute_default_rope_parameters           -> 静态 inv_freq 计算，无 forward
  DeepseekV4PreTrainedModel._init_weights   -> 权重初始化

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/transformers/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gfused/test_deepseek_v4_pack_seq_agnostic.py
"""

import os
import unittest

import torch
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    FineGrainedFP8Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)

HF_MODEL_PATH = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
NUM_LAYERS = 4  # 最小前缀，覆盖 sliding / CSA / HCA 三种 attn + hash_moe / moe 两种 mlp
DEVICE = "cuda:0"
SEG_LENS = (16, 24, 40)  # 三段不同长度；均为 4 的倍数（CSA m=4 窗口对齐）；总长 80 < 128（HCA m'=128）
HCA_SEG_LENS = (128, 128, 128)  # HCA compressor 需每段 >= m'=128 才有完整窗口


def _truncate_config(config):
    """截断到 NUM_LAYERS 层以加速加载。"""
    config.num_hidden_layers = NUM_LAYERS
    config.layer_types = config.layer_types[:NUM_LAYERS]
    config.mlp_layer_types = config.mlp_layer_types[:NUM_LAYERS]
    return config


class TestPackSeqAgnostic(unittest.TestCase):
    """逐 module/function 的 pack-seq 不变量测试（前向）。"""

    model = None

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda is required")
        if not os.path.isdir(HF_MODEL_PATH):
            raise unittest.SkipTest(
                f"model dir not found: {HF_MODEL_PATH}; "
                f"download DeepSeek-V4-Flash into hf-hub/ first"
            )
        torch.cuda.set_device(0)
        cls.device = torch.device(DEVICE)

        config = DeepseekV4Config.from_pretrained(HF_MODEL_PATH)
        _truncate_config(config)
        print(
            f"loading {HF_MODEL_PATH} with {NUM_LAYERS} layers (bf16, eager experts) ..."
        )
        # experts_implementation="eager"：回到逐 expert 的 loop forward（默认 grouped_mm）。
        # 不做 model.to(...)：保留 fp8 dequant 后各 module 的原生 dtype。
        cls.model = DeepseekV4ForCausalLM.from_pretrained(
            HF_MODEL_PATH,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": DEVICE},
            trust_remote_code=True,
            quantization_config=FineGrainedFP8Config(dequantize=True),
            experts_implementation="eager",
        )
        cls.model.eval()
        cls.config = cls.model.config

        lt = list(cls.config.layer_types)
        mt = list(cls.config.mlp_layer_types)
        cls.hash_idx = mt.index("hash_moe") if "hash_moe" in mt else None
        cls.moe_idx = mt.index("moe") if "moe" in mt else None
        cls.hca_idx = (
            lt.index("heavily_compressed_attention")
            if "heavily_compressed_attention" in lt else None
        )
        cls.csa_idx = (
            lt.index("compressed_sparse_attention")
            if "compressed_sparse_attention" in lt else None
        )
        cls.sliding_idx = lt.index(
            "sliding_attention"
        ) if "sliding_attention" in lt else None
        print(f"layer_types={lt}\nmlp_layer_types={mt}")

    @classmethod
    def tearDownClass(cls):
        cls.model = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _gen(self, shape, dtype, seed):
        """固定种子的随机张量（CPU 生成 → 搬到 device/dtype，保证可复现）。"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(
            device=self.device, dtype=dtype
        )

    def _gen_ids(self, length, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.randint(0, self.config.vocab_size, (1, length), generator=g)
        return ids.to(self.device)

    def _q_residual(self, layer_idx, hidden):
        """复用真实 attention 的 q_a 路径造 q_residual（CSA/Indexer 需要）。"""
        attn = self.model.model.layers[layer_idx].self_attn
        return attn.q_a_norm(attn.q_a_proj(hidden))

    def _worst_diff(self, packed_out, seg_outs, seq_dim):
        """packed_out 按各段长度切回，逐段与单算输出比，返回最坏 (abs, rel)。"""
        offset = 0
        worst_abs = 0.0
        worst_rel = 0.0
        for seg in seg_outs:
            length = seg.shape[seq_dim]
            sliced = packed_out.narrow(seq_dim, offset, length)
            offset += length
            abs_d = (sliced.float() - seg.float()).abs().max().item()
            rel_d = abs_d / max(seg.float().abs().max().item(), 1e-12)
            worst_abs = max(worst_abs, abs_d)
            worst_rel = max(worst_rel, rel_d)
        return worst_abs, worst_rel

    def _assert_agnostic(self, name, packed_out, seg_outs, seq_dim, rtol, atol):
        worst_abs, worst_rel = self._worst_diff(packed_out, seg_outs, seq_dim)
        print(
            f"[agnostic:{name}] max_abs_diff={worst_abs:.3e} rel_diff={worst_rel:.3e} "
            f"(rtol={rtol}, atol={atol})"
        )
        self.assertTrue(
            worst_abs <= atol or worst_rel <= rtol,
            f"{name} is NOT pack-agnostic: max_abs_diff={worst_abs:.3e} "
            f"rel_diff={worst_rel:.3e} exceeds (atol={atol}, rtol={rtol})",
        )

    def _assert_aware(self, name, packed_out, seg_outs, seq_dim, threshold=0.1):
        worst_abs, worst_rel = self._worst_diff(packed_out, seg_outs, seq_dim)
        print(
            f"[aware:{name}] max_abs_diff={worst_abs:.3e} rel_diff={worst_rel:.3e} "
            f"(expect rel_diff > {threshold})"
        )
        self.assertGreater(
            worst_rel,
            threshold,
            f"{name} was expected to be pack-AWARE (cross-token) but packed≈split "
            f"(rel_diff={worst_rel:.3e} <= {threshold}); methodology would miss an aware module",
        )

    def _assert_aware_indices(self, name, packed_idx, seg_idxs, threshold=0.1):
        """整型 top-k 索引专用：按段切回，逐段比错配比例（k 不同则截到 min）。"""
        offset = 0
        worst_frac = 0.0
        for seg in seg_idxs:
            length = seg.shape[1]
            sliced = packed_idx[:, offset:offset + length, :]
            offset += length
            kmin = min(seg.shape[2], sliced.shape[2])
            mism = (sliced[:, :, :kmin]
                    != seg[:, :, :kmin]).float().mean().item()
            worst_frac = max(worst_frac, mism)
        print(
            f"[aware:{name}] worst index-mismatch fraction={worst_frac:.3f} "
            f"(expect > {threshold})"
        )
        self.assertGreater(
            worst_frac,
            threshold,
            f"{name} was expected to be pack-AWARE but packed top-k indices ≈ split "
            f"(mismatch={worst_frac:.3f} <= {threshold})",
        )

    # ==================================================================
    # AGNOSTIC modules（应满足不变量）
    # ==================================================================

    def test_rmsnorm_input_layernorm(self):
        norm = self.model.model.layers[0].input_layernorm
        h = self.config.hidden_size
        segs = [
            self._gen((1, l, h), torch.float32, 100 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [norm(x) for x in segs]
            packed = norm(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "RMSNorm.input_layernorm", packed, seg_out, 1, rtol=1e-3, atol=1e-3
        )

    def test_rmsnorm_final_norm(self):
        norm = self.model.model.norm
        h = self.config.hidden_size
        segs = [
            self._gen((1, l, h), torch.float32, 110 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [norm(x) for x in segs]
            packed = norm(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "RMSNorm.norm", packed, seg_out, 1, rtol=1e-3, atol=1e-3
        )

    def test_unweighted_rmsnorm(self):
        norm = self.model.model.layers[
            0].self_attn.q_b_norm  # DeepseekV4UnweightedRMSNorm
        d = self.config.head_dim
        segs = [
            self._gen((1, l, d), torch.float32, 120 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [norm(x) for x in segs]
            packed = norm(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "UnweightedRMSNorm.q_b_norm",
            packed,
            seg_out,
            1,
            rtol=1e-3,
            atol=1e-3
        )

    def test_grouped_linear(self):
        proj = self.model.model.layers[
            0].self_attn.o_a_proj  # DeepseekV4GroupedLinear
        n_groups = self.config.o_groups
        in_per_group = self.config.num_attention_heads * self.config.head_dim // n_groups
        segs = [
            self._gen((1, l, n_groups, in_per_group), torch.bfloat16, 130 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [proj(x) for x in segs]
            packed = proj(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "GroupedLinear.o_a_proj", packed, seg_out, 1, rtol=2e-2, atol=2e-2
        )

    def test_mlp_shared_experts(self):
        mlp = self.model.model.layers[0].mlp.shared_experts  # DeepseekV4MLP
        h = self.config.hidden_size
        segs = [
            self._gen((1, l, h), torch.bfloat16, 140 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [mlp(x) for x in segs]
            packed = mlp(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "MLP.shared_experts", packed, seg_out, 1, rtol=2e-2, atol=2e-2
        )

    def test_hyperconnection_attn_hc(self):
        hc = self.model.model.layers[0].attn_hc
        h = self.config.hidden_size
        n_hc = self.config.hc_mult
        segs = [
            self._gen((1, l, n_hc, h), torch.float32, 200 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [hc(x) for x in segs]  # 各段 (post, comb, collapsed)
            packed = hc(torch.cat(segs, dim=1))
        for k, label in enumerate(("post", "comb", "collapsed")):
            self._assert_agnostic(
                f"HyperConnection.{label}",
                packed[k],
                [o[k] for o in seg_out],
                1,
                rtol=5e-3,
                atol=5e-3,
            )

    def test_hyperhead(self):
        head = self.model.model.hc_head
        h = self.config.hidden_size
        n_hc = self.config.hc_mult
        segs = [
            self._gen((1, l, n_hc, h), torch.float32, 300 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [head(x) for x in segs]
            packed = head(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "HyperHead", packed, seg_out, 1, rtol=5e-3, atol=5e-3
        )

    def test_embedding(self):
        emb = self.model.model.embed_tokens
        seg_ids = [self._gen_ids(l, 400 + i) for i, l in enumerate(SEG_LENS)]
        with torch.no_grad():
            seg_out = [emb(x) for x in seg_ids]
            packed = emb(torch.cat(seg_ids, dim=1))
        self._assert_agnostic(
            "Embedding", packed, seg_out, 1, rtol=1e-4, atol=1e-4
        )

    def test_topk_router(self):
        if self.moe_idx is None:
            self.skipTest("no moe (topk) layer in truncated config")
        gate = self.model.model.layers[self.moe_idx
                                      ].mlp.gate  # DeepseekV4TopKRouter
        h = self.config.hidden_size
        segs = [
            self._gen((1, l, h), torch.bfloat16, 450 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_logits = [
                gate(x)[0] for x in segs
            ]  # logits [N, num_experts]，seq 在第 0 维
            packed_logits = gate(torch.cat(segs, dim=1))[0]
        self._assert_agnostic(
            "TopKRouter.logits",
            packed_logits,
            seg_logits,
            0,
            rtol=2e-2,
            atol=2e-2
        )

    def test_hash_router(self):
        if self.hash_idx is None:
            self.skipTest("no hash_moe layer in truncated config")
        gate = self.model.model.layers[self.hash_idx
                                      ].mlp.gate  # DeepseekV4HashRouter
        h = self.config.hidden_size
        segs = [
            self._gen((1, l, h), torch.bfloat16, 460 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        seg_ids = [self._gen_ids(l, 465 + i) for i, l in enumerate(SEG_LENS)]
        with torch.no_grad():
            seg_logits = [gate(x, ids)[0] for x, ids in zip(segs, seg_ids)]
            packed_logits = gate(
                torch.cat(segs, dim=1), torch.cat(seg_ids, dim=1)
            )[0]
        self._assert_agnostic(
            "HashRouter.logits",
            packed_logits,
            seg_logits,
            0,
            rtol=2e-2,
            atol=2e-2
        )

    def test_rope_sample_local_positions(self):
        """RoPE 在「逐样本本地 position_ids（每段从 0 重新计数）」下对 pack 无感知。

        这是 pack-seq 设计最核心的假设。覆盖 RotaryEmbedding + apply_rotary_pos_emb。
        """
        rotary = self.model.model.rotary_emb
        n_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim

        def rope(q, position_ids):
            cos, sin = rotary(q, position_ids=position_ids, layer_type="main")
            return apply_rotary_pos_emb(q, cos, sin)

        segs = [
            self._gen((1, n_heads, l, head_dim), torch.float32, 500 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [
                rope(q,
                     torch.arange(l, device=self.device).unsqueeze(0))
                for q, l in zip(segs, SEG_LENS)
            ]
            q_packed = torch.cat(segs, dim=2)  # seq 在第 2 维
            pos_packed = torch.cat(
                [torch.arange(l, device=self.device) for l in SEG_LENS]
            ).unsqueeze(0)
            packed = rope(q_packed, pos_packed)
        self._assert_agnostic(
            "RoPE.main", packed, seg_out, 2, rtol=1e-3, atol=1e-3
        )

    def test_rotate_half(self):
        d = self.config.head_dim
        segs = [
            self._gen((1, l, d), torch.float32, 520 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [rotate_half(x) for x in segs]
            packed = rotate_half(torch.cat(segs, dim=1))
        self._assert_agnostic(
            "rotate_half", packed, seg_out, 1, rtol=1e-5, atol=1e-5
        )

    def test_repeat_kv(self):
        n_rep = self.config.num_attention_heads  # V4 单 KV head broadcast 到全部 head
        head_dim = self.config.head_dim
        # repeat_kv 输入 [B, n_kv, S, head_dim]，seq 在第 2 维
        segs = [
            self._gen((1, 1, l, head_dim), torch.float32, 540 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [repeat_kv(x, n_rep) for x in segs]
            packed = repeat_kv(torch.cat(segs, dim=2), n_rep)
        self._assert_agnostic(
            "repeat_kv", packed, seg_out, 2, rtol=1e-5, atol=1e-5
        )

    def test_moe_block_hash(self):
        if self.hash_idx is None:
            self.skipTest("no hash_moe layer in truncated config")
        mlp = self.model.model.layers[self.hash_idx].mlp
        assert mlp.is_hash, "expected hash router on a hash_moe layer"
        h = self.config.hidden_size
        seg_h = [
            self._gen((1, l, h), torch.bfloat16, 600 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        seg_ids = [self._gen_ids(l, 650 + i) for i, l in enumerate(SEG_LENS)]
        with torch.no_grad():
            seg_out = [mlp(x, input_ids=ids) for x, ids in zip(seg_h, seg_ids)]
            packed = mlp(
                torch.cat(seg_h, dim=1), input_ids=torch.cat(seg_ids, dim=1)
            )
        self._assert_agnostic(
            "SparseMoeBlock.hash", packed, seg_out, 1, rtol=3e-2, atol=3e-2
        )

    def test_moe_block_topk(self):
        if self.moe_idx is None:
            self.skipTest("no moe (topk) layer in truncated config")
        mlp = self.model.model.layers[self.moe_idx].mlp
        assert not mlp.is_hash, "expected topk router on a moe layer"
        h = self.config.hidden_size
        seg_h = [
            self._gen((1, l, h), torch.bfloat16, 700 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [mlp(x) for x in seg_h]  # topk 路由不用 input_ids
            packed = mlp(torch.cat(seg_h, dim=1))
        self._assert_agnostic(
            "SparseMoeBlock.topk", packed, seg_out, 1, rtol=3e-2, atol=3e-2
        )

    # ==================================================================
    # AWARE 负向对照（应 NOT 满足不变量）
    # ==================================================================

    def test_attention_is_pack_aware(self):
        """attention 跨 token 混合 → packed 与单算显著不同。"""
        idx = self.hca_idx if self.hca_idx is not None else self.sliding_idx
        if idx is None:
            self.skipTest(
                "no HCA / sliding layer to use as attention negative control"
            )
        if self.hca_idx is not None:
            assert sum(
                SEG_LENS
            ) < 128, "negative control needs total length < 128 for an HCA layer"
        attn = self.model.model.layers[idx].self_attn
        rotary = self.model.model.rotary_emb
        h = self.config.hidden_size

        def run(hidden, position_ids):
            pe = {
                "main":
                    rotary(
                        hidden, position_ids=position_ids, layer_type="main"
                    ),
                "compress":
                    rotary(
                        hidden,
                        position_ids=position_ids,
                        layer_type="compress"
                    ),
            }
            out, _ = attn(hidden, pe, position_ids, None, past_key_values=None)
            return out

        seg_h = [
            self._gen((1, l, h), torch.bfloat16, 800 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [
                run(x,
                    torch.arange(l, device=self.device).unsqueeze(0))
                for x, l in zip(seg_h, SEG_LENS)
            ]
            # packed 也用逐样本本地 position_ids（与各段一致），隔离纯跨 token 混合：
            # 无感知 module 在相同 position 下会完全一致，attention 仍因跨段注意力发散。
            pos_packed = torch.cat(
                [torch.arange(l, device=self.device) for l in SEG_LENS]
            ).unsqueeze(0)
            packed = run(torch.cat(seg_h, dim=1), pos_packed)
        self._assert_aware("Attention", packed, seg_out, 1, threshold=0.1)

    def test_hca_compressor_is_pack_aware(self):
        """HCA compressor 把 token 按窗口压缩 + 绝对位置 RoPE（按全局窗口序），跨段不同。"""
        if self.hca_idx is None:
            self.skipTest("no HCA layer for compressor negative control")
        comp = self.model.model.layers[
            self.hca_idx].self_attn.compressor  # DeepseekV4HCACompressor
        h = self.config.hidden_size

        def run(hidden, position_ids):
            qr = self._q_residual(
                self.hca_idx, hidden
            )  # HCA 不用 q_residual，传真实值无害
            compressed_kv, _ = comp(
                hidden, qr, position_ids, None, self.hca_idx
            )
            return compressed_kv  # [B, 1, T_windows, head_dim]

        seg_h = [
            self._gen((1, l, h), torch.bfloat16, 900 + i)
            for i, l in enumerate(HCA_SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [
                run(x,
                    torch.arange(l, device=self.device).unsqueeze(0))
                for x, l in zip(seg_h, HCA_SEG_LENS)
            ]
            total = sum(HCA_SEG_LENS)
            packed = run(
                torch.cat(seg_h, dim=1),
                torch.arange(total, device=self.device).unsqueeze(0)
            )
        # compressed_kv 的窗口维是第 2 维
        self._assert_aware(
            "HCACompressor.compressed_kv", packed, seg_out, 2, threshold=0.1
        )

    def test_csa_compressor_is_pack_aware(self):
        """CSA compressor 有 Ca/Cb 跨窗口 overlap + 绝对位置 RoPE，跨段不同。"""
        if self.csa_idx is None:
            self.skipTest("no CSA layer for compressor negative control")
        comp = self.model.model.layers[
            self.csa_idx].self_attn.compressor  # DeepseekV4CSACompressor
        h = self.config.hidden_size

        def run(hidden, position_ids):
            qr = self._q_residual(self.csa_idx, hidden)
            compressed_kv, _ = comp(
                hidden, qr, position_ids, None, self.csa_idx
            )
            return compressed_kv  # [B, 1, T_windows, head_dim]

        seg_h = [
            self._gen((1, l, h), torch.bfloat16, 1000 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [
                run(x,
                    torch.arange(l, device=self.device).unsqueeze(0))
                for x, l in zip(seg_h, SEG_LENS)
            ]
            total = sum(SEG_LENS)
            packed = run(
                torch.cat(seg_h, dim=1),
                torch.arange(total, device=self.device).unsqueeze(0)
            )
        self._assert_aware(
            "CSACompressor.compressed_kv", packed, seg_out, 2, threshold=0.1
        )

    def test_indexer_is_pack_aware(self):
        """Lightning Indexer 跨 token 打分选 top-k 压缩块，packed 选出的索引与单算不同。"""
        if self.csa_idx is None:
            self.skipTest("no CSA layer for indexer negative control")
        indexer = self.model.model.layers[self.csa_idx
                                         ].self_attn.compressor.indexer
        h = self.config.hidden_size

        def run(hidden, position_ids):
            qr = self._q_residual(self.csa_idx, hidden)
            return indexer(
                hidden, qr, position_ids, None, self.csa_idx
            )  # top_k_indices [B, S, k]

        seg_h = [
            self._gen((1, l, h), torch.bfloat16, 1100 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        with torch.no_grad():
            seg_out = [
                run(x,
                    torch.arange(l, device=self.device).unsqueeze(0))
                for x, l in zip(seg_h, SEG_LENS)
            ]
            total = sum(SEG_LENS)
            # packed 用全局 position_ids（pack 无感知调用方的天真行为）→ 跨段可见性变化
            packed = run(
                torch.cat(seg_h, dim=1),
                torch.arange(total, device=self.device).unsqueeze(0)
            )
        self._assert_aware_indices(
            "Indexer.top_k_indices", packed, seg_out, threshold=0.1
        )

    def test_decoder_layer_is_pack_aware(self):
        """整个 decoder layer 含 attention → 跨 token，packed 与单算显著不同。"""
        idx = self.hca_idx if self.hca_idx is not None else self.sliding_idx
        if idx is None:
            self.skipTest(
                "no HCA / sliding layer for decoder-layer negative control"
            )
        if self.hca_idx is not None:
            assert sum(
                SEG_LENS
            ) < 128, "negative control needs total length < 128 for an HCA layer"
        layer = self.model.model.layers[idx]
        rotary = self.model.model.rotary_emb
        h = self.config.hidden_size
        n_hc = self.config.hc_mult

        def run(hidden_streams, input_ids, position_ids):
            pe = {
                "main":
                    rotary(
                        hidden_streams,
                        position_ids=position_ids,
                        layer_type="main"
                    ),
                "compress":
                    rotary(
                        hidden_streams,
                        position_ids=position_ids,
                        layer_type="compress"
                    ),
            }
            return layer(
                hidden_streams,
                input_ids=input_ids,
                position_embeddings=pe,
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=None,
            )

        seg_h = [
            self._gen((1, l, n_hc, h), torch.bfloat16, 1200 + i)
            for i, l in enumerate(SEG_LENS)
        ]
        seg_ids = [self._gen_ids(l, 1250 + i) for i, l in enumerate(SEG_LENS)]
        with torch.no_grad():
            seg_out = [
                run(x, ids,
                    torch.arange(l, device=self.device).unsqueeze(0))
                for x, ids, l in zip(seg_h, seg_ids, SEG_LENS)
            ]
            pos_packed = torch.cat(
                [torch.arange(l, device=self.device) for l in SEG_LENS]
            ).unsqueeze(0)
            packed = run(
                torch.cat(seg_h, dim=1), torch.cat(seg_ids, dim=1), pos_packed
            )
        self._assert_aware("DecoderLayer", packed, seg_out, 1, threshold=0.1)


if __name__ == "__main__":
    unittest.main()
