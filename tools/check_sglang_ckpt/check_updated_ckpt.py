'''
python3 -u tools/check_sglang_ckpt/check_updated_ckpt.py \
  --tp_size 1 \
  --check_dp_ranges 0:32 \
  --real_base examples/xiaotaoliu/vllm_weight_transfer/debug_weights/real_weights_{} \
  --src_base  examples/xiaotaoliu/vllm_weight_transfer/debug_weights/src_weights_{} \
  --zero_base examples/xiaotaoliu/vllm_weight_transfer/debug_weights/zero_weights_{}
'''

import argparse
import os
from typing import Dict, List, Tuple

import safetensors.torch as st
import torch

# ===================== 固定配置（你的ckpt路径） =====================
REAL_CKPT_BASE = "examples/xiaotaoliu/vllm_weight_transfer/debug_weights_qwen3_30b_moe/real_weights_{}"
SRC_CKPT_BASE = "examples/xiaotaoliu/vllm_weight_transfer/debug_weights_qwen3_30b_moe/src_weights_{}"
ZERO_CKPT_BASE = "examples/xiaotaoliu/vllm_weight_transfer/debug_weights_qwen3_30b_moe/zero_weights_{}"
# 默认文件名前缀（可根据实际情况调整）
FILE_PREFIX = "model-rank-{}-part-0.safetensors"

# zero-weights 检查时忽略的 key 后缀。
# 这些 key 是 vLLM 内部的 runtime buffer（如 attention 的 fp8 量化 scale），
# 默认值为 1.0 且 trainer 端并不发送，因此在 "全零覆盖" 那一轮不会被写成 0，
# 属于预期行为，不应算作失败。
ZERO_CHECK_IGNORE_SUFFIXES: Tuple[str, ...] = (
    "_k_scale",
    "_v_scale",
    "_q_scale",
    "_prob_scale",
)


def _should_ignore_zero_check(key: str) -> bool:
    """判断某个 key 在零值检查时是否应该被忽略。"""
    return any(key.endswith(suf) for suf in ZERO_CHECK_IGNORE_SUFFIXES)


def get_tp_files(tp_size: int, ckpt_dir: str, file_prefix: str = FILE_PREFIX) -> List[str]:
    """
    根据TP_SIZE生成指定数量的safetensors文件路径（按rank从0到tp_size-1）
    :param tp_size: 张量并行尺寸，决定文件数量（rank 0 ~ tp_size-1）
    :param ckpt_dir: ckpt根目录
    :param file_prefix: 文件命名模板（包含rank占位符）
    :return: 存在的文件路径列表
    """
    file_paths = []
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"CKPT目录不存在: {ckpt_dir}")

    for rank in range(tp_size):
        file_name = file_prefix.format(rank)
        file_path = os.path.abspath(os.path.join(ckpt_dir, file_name))
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            print(f"警告：TP rank {rank} 对应的文件不存在 - {file_path}")

    if len(file_paths) == 0:
        raise ValueError(f"TP_SIZE={tp_size}时，{ckpt_dir}下无可用的safetensors文件")
    return file_paths


def load_safetensors_file(file_path: str) -> Dict[str, torch.Tensor]:
    """
    加载safetensors文件（替代torch.load，适配safetensors格式）
    """
    try:
        # 加载到CPU，避免设备不兼容
        weights = st.load_file(file_path, device="cpu")
        if not isinstance(weights, dict):
            raise ValueError(f"文件 {file_path} 不是字典格式！")
        return weights
    except Exception as e:
        raise RuntimeError(f"加载 {file_path} 失败: {str(e)}") from e


def compare_weights_dict(
    src_weights: Dict[str, torch.Tensor], real_weights: Dict[str, torch.Tensor],
    file_pair: Tuple[str, str]
) -> Tuple[bool, List[str]]:
    """
    对比两个权重字典的所有key对应的张量是否相等
    返回：(是否全相等, 不相等的key列表)
    """
    unequal_keys = []
    # 1. 检查key集合是否一致
    src_keys = set(src_weights.keys())
    real_keys = set(real_weights.keys())

    if src_keys != real_keys:
        missing_in_src = real_keys - src_keys
        missing_in_real = src_keys - real_keys
        if missing_in_src:
            unequal_keys.append(f"src缺失key: {sorted(missing_in_src)}")
        if missing_in_real:
            unequal_keys.append(f"real缺失key: {sorted(missing_in_real)}")
        print(
            f"文件对 {os.path.basename(file_pair[0])} <-> {os.path.basename(file_pair[1])} - Key集合不一致"
        )
        return False, unequal_keys

    # 2. 逐key对比张量
    for key in sorted(src_keys):
        src_tensor = src_weights[key]
        real_tensor = real_weights[key]

        # 检查形状
        if src_tensor.shape != real_tensor.shape:
            unequal_keys.append(
                f"key={key} 形状不一致: src={src_tensor.shape}, real={real_tensor.shape}"
            )
            continue

        # 检查值（浮点型近似相等，整型严格相等）
        try:
            if torch.is_floating_point(src_tensor):
                is_equal = torch.allclose(src_tensor, real_tensor, rtol=1e-5, atol=1e-8)
            else:
                is_equal = torch.equal(src_tensor, real_tensor)

            if not is_equal:
                # 输出统计信息便于定位
                src_mean = src_tensor.mean().item()
                real_mean = real_tensor.mean().item()
                src_max = src_tensor.max().item()
                real_max = real_tensor.max().item()
                unequal_keys.append(
                    f"key={key} 值不相等 - src(均值={src_mean:.6f}, 最大值={src_max:.6f}), "
                    f"real(均值={real_mean:.6f}, 最大值={real_max:.6f})"
                )
        except Exception as e:
            unequal_keys.append(f"key={key} 对比失败: {str(e)}")

    if unequal_keys:
        print(
            f"文件对 {os.path.basename(file_pair[0])} <-> {os.path.basename(file_pair[1])} - 不相等key数: {len(unequal_keys)}"
        )
        return False, unequal_keys
    return True, []


def check_all_zeros(tensor: torch.Tensor) -> bool:
    """判断张量所有元素是否为0（支持浮点型极小值容错）"""
    try:
        # 浮点型允许极小值（避免数值精度问题）
        if torch.is_floating_point(tensor):
            return torch.all(torch.abs(tensor) < 1e-10).item()
        else:
            return torch.all(tensor == 0).item()
    except Exception as e:
        raise RuntimeError(f"张量零值检查失败: {str(e)}") from e


def check_zero_weights(zero_weights: Dict[str, torch.Tensor],
                       file_path: str) -> Tuple[bool, List[str]]:
    """检查零值权重字典中所有张量是否全为0

    带 ``ZERO_CHECK_IGNORE_SUFFIXES`` 后缀的 key（如 vLLM 内部的 fp8 attention scale）
    会在检查时被跳过：这些是 runtime buffer，trainer 并不发送，因此不会被
    "全零覆盖" 那一轮写成 0，不算作失败。
    """
    non_zero_keys = []
    ignored_non_zero_keys: List[str] = []
    for key in sorted(zero_weights.keys()):
        tensor = zero_weights[key]
        if check_all_zeros(tensor):
            continue

        tensor_mean = tensor.mean().item()
        tensor_max = tensor.max().item()
        tensor_min = tensor.min().item()
        info = (
            f"key={key} 非零 - 均值={tensor_mean:.6f}, "
            f"最大值={tensor_max:.6f}, 最小值={tensor_min:.6f}"
        )
        if _should_ignore_zero_check(key):
            ignored_non_zero_keys.append(info)
        else:
            non_zero_keys.append(info)

    if ignored_non_zero_keys:
        print(
            f"文件 {os.path.basename(file_path)} - 已忽略非零key数: {len(ignored_non_zero_keys)}"
            f"（匹配 {ZERO_CHECK_IGNORE_SUFFIXES}）"
        )

    if non_zero_keys:
        print(f"文件 {os.path.basename(file_path)} - 非零key数: {len(non_zero_keys)}")
        return False, non_zero_keys
    return True, []


def main(args):
    print("=" * 60)
    print(f"开始执行权重检查 | TP_SIZE={args.tp_size}")
    print("=" * 60)

    # CLI 提供的路径模板优先；未提供则回退到模块级默认值（便于通过 `import` 调用）。
    real_base = getattr(args, "real_base", None) or REAL_CKPT_BASE
    src_base = getattr(args, "src_base", None) or SRC_CKPT_BASE
    zero_base = getattr(args, "zero_base", None) or ZERO_CKPT_BASE
    file_prefix = getattr(args, "file_prefix", None) or FILE_PREFIX

    # ===================== 1. 生成TP对应的文件列表 =====================
    try:
        print(f"\n【步骤1】根据TP_SIZE={args.tp_size}加载文件...")
        real_files = get_tp_files(
            args.tp_size, real_base.format(args.check_dp_rank), file_prefix=file_prefix
        )
        src_files = get_tp_files(
            args.tp_size, src_base.format(args.check_dp_rank), file_prefix=file_prefix
        )
        zero_files = get_tp_files(
            args.tp_size, zero_base.format(args.check_dp_rank), file_prefix=file_prefix
        )

        print(
            f"  real_weights文件: {[os.path.basename(f) for f in real_files]} real_base={real_base!r}"
        )
        print(f"  src_weights文件: {[os.path.basename(f) for f in src_files]} src_base={src_base!r}")
        print(
            f"  zero_weights文件: {[os.path.basename(f) for f in zero_files]} zero_base={zero_base!r}"
        )
    except Exception as e:
        print(f"文件加载失败: {e}")
        return

    # ===================== 2. 对比src vs real =====================
    print(f"\n【步骤2】对比src_weights vs real_weights...")
    compare_summary = {
        "total": 0,
        "equal": 0,
        "unequal": [],
    }

    # 按TP rank一一对应对比（取最短列表避免越界）
    max_rank = min(len(src_files), len(real_files))
    compare_summary["total"] = max_rank

    for rank in range(max_rank):
        src_file = src_files[rank]
        real_file = real_files[rank]
        print(f"\n--- TP rank {rank} 对比 ---")
        print(f"  src: {os.path.basename(src_file)}")
        print(f"  real: {os.path.basename(real_file)}")

        # 加载权重
        try:
            src_weights = load_safetensors_file(src_file)
            real_weights = load_safetensors_file(real_file)
        except Exception as e:
            compare_summary["unequal"].append((rank, src_file, real_file, f"加载失败: {e}"))
            print(f"  ❌ 加载失败: {e}")
            continue

        # 对比权重
        is_equal, unequal_keys = compare_weights_dict(
            src_weights, real_weights, (src_file, real_file)
        )
        if is_equal:
            compare_summary["equal"] += 1
            print(f"  ✅ 所有权重相等")
        else:
            compare_summary["unequal"].append((rank, src_file, real_file, unequal_keys))
            print(f"  ❌ 存在不相等权重")

    # ===================== 3. 检查zero_weights =====================
    print(f"\n【步骤3】检查zero_weights是否全为0...")
    zero_summary = {
        "total": len(zero_files),
        "all_zero": 0,
        "non_zero": [],
    }

    for rank, zero_file in enumerate(zero_files):
        print(f"\n--- TP rank {rank} 检查 ---")
        print(f"  zero: {os.path.basename(zero_file)}")

        # 加载权重
        try:
            zero_weights = load_safetensors_file(zero_file)
        except Exception as e:
            zero_summary["non_zero"].append((rank, zero_file, f"加载失败: {e}"))
            print(f"  ❌ 加载失败: {e}")
            continue

        # 检查零值
        is_all_zero, non_zero_keys = check_zero_weights(zero_weights, zero_file)
        if is_all_zero:
            zero_summary["all_zero"] += 1
            print(f"  ✅ 所有张量全为0")
        else:
            zero_summary["non_zero"].append((rank, zero_file, non_zero_keys))
            print(f"  ❌ 存在非零张量")

    # ===================== 4. 输出最终报告 =====================
    print("\n" + "=" * 60)
    print("最终检查报告")
    print("=" * 60)

    # 对比结果汇总
    print(f"\n【src_weights vs real_weights 对比结果】")
    print(f"  总对比TP rank数: {compare_summary['total']}")
    print(f"  权重全相等的rank数: {compare_summary['equal']}")
    print(f"  权重不相等的rank数: {len(compare_summary['unequal'])}")
    if compare_summary["unequal"]:
        print(f"  不相等详情:")
        for rank, src, real, reason in compare_summary["unequal"]:
            print(f"    Rank {rank}:")
            print(f"      src: {os.path.basename(src)}")
            print(f"      real: {os.path.basename(real)}")
            print(f"      原因: {reason}")

    # 零值检查汇总
    print(f"\n【zero_weights 零值检查结果】")
    print(f"  总检查TP rank数: {zero_summary['total']}")
    print(f"  全为0的rank数: {zero_summary['all_zero']}")
    print(f"  存在非零的rank数: {len(zero_summary['non_zero'])}")
    if zero_summary["non_zero"]:
        print(f"  非零详情:")
        for rank, file, reason in zero_summary["non_zero"]:
            print(f"    Rank {rank}:")
            print(f"      文件: {os.path.basename(file)}")
            print(f"      原因: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按TP_SIZE对比权重&检查零值（适配safetensors）")
    parser.add_argument(
        "--check_dp_rank",
        type=int,
        default=0,
        help="要检查的 DP rank，会被 format 到 *_base 模板的 `{}` 占位符里"
    )
    parser.add_argument(
        "--check_dp_ranges",
        type=str,
        default=None,
        help="批量检查多个 DP rank，格式 'start:end'（左闭右开）或 '0,1,3'；设置后忽略 --check_dp_rank"
    )
    parser.add_argument(
        "--tp_size", type=int, required=True, help="张量并行尺寸（TP_SIZE），决定文件数量（rank 0~tp_size-1）"
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default=FILE_PREFIX,
        help="文件命名模板，默认: model-rank-{}-part-0.safetensors"
    )
    parser.add_argument(
        "--real_base",
        type=str,
        default=REAL_CKPT_BASE,
        help=f"real_weights 目录模板，默认: {REAL_CKPT_BASE}"
    )
    parser.add_argument(
        "--src_base", type=str, default=SRC_CKPT_BASE, help=f"src_weights 目录模板，默认: {SRC_CKPT_BASE}"
    )
    parser.add_argument(
        "--zero_base",
        type=str,
        default=ZERO_CKPT_BASE,
        help=f"zero_weights 目录模板，默认: {ZERO_CKPT_BASE}"
    )

    args = parser.parse_args()

    if args.check_dp_ranges:
        spec = args.check_dp_ranges.strip()
        if ":" in spec:
            start_str, end_str = spec.split(":", 1)
            dp_ranks = list(range(int(start_str), int(end_str)))
        else:
            dp_ranks = [int(x) for x in spec.split(",") if x.strip()]
    else:
        dp_ranks = [args.check_dp_rank]

    for dp in dp_ranks:
        if len(dp_ranks) > 1:
            print(f"\n############### DP rank {dp} ###############")
        args.check_dp_rank = dp
        main(args)
