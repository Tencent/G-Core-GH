# adapted from: https://github.com/fla-org/flash-linear-attention/blob/main/scripts/extract_triton_autotune_cache.py
"""
预生成 FLA (flash-linear-attention) 的 Triton autotune 缓存。

功能：
  1. 自动从模型 config.json 中检测 GDN/Attention 的 head_dim
  2. 运行 FLA kernel 的完整 forward + backward，触发 Triton autotune benchmark
  3. 从 benchmark 结果中提取最优配置，写入 FLA 配置目录（fla/configs/{GPU}/）
  4. 支持 --list-only 列出已有缓存文件

用法：
  # 自动检测 head_dim（推荐）
  python -m tools.fla_autotune.generate_fla_autotune_cache --model-path hf-hub/Qwen/Qwen3.5-4B --op gdn --versioned

  # 手动指定 head_dim
  python -m tools.fla_autotune.generate_fla_autotune_cache -d 128 --op gdn --versioned

  # 多个 head_dim
  python -m tools.fla_autotune.generate_fla_autotune_cache -d 64 128 256 --op both --versioned

  # 仅从已有缓存提取（不生成）
  python -m tools.fla_autotune.generate_fla_autotune_cache --extract-only

依赖：
  - flash-linear-attention（运行时需能 import fla.*）
  - triton
  - torch

引用链路（方便追溯）：
  本工具最初改编自 FLA 仓库的两个脚本，目前已本地化独立维护：

  本地模块布局（tools/fla_autotune/）：
    generate_fla_autotune_cache.py        ← 主入口：解析参数、检测 head_dim、调度生成/提取
    utils/
      __init__.py                         ← 导出 extract_configs, generate_fla_cache, get_triton_cache_dir
      autotune_export.py                  ← extract_configs() 从 .autotune.json 提取最优配置
      autotune_generate.py                ← generate_fla_cache() 运行 kernel 触发 autotune，封装 FLACacheGenerator

  上游 FLA 仓库原始文件（仅作参考，不直接使用）：
    flash-linear-attention/
      scripts/
        extract_triton_autotune_cache.py  ← 缓存生成/提取入口
        utils/autotune_export.py          ← 从 .autotune.json 提取最佳配置
        utils/autotune_generate.py        ← FLACacheGenerator 运行 kernel 触发 autotune
      fla/ops/utils/cache.py              ← FLA 配置缓存系统（FLA_CACHE_MODE, get_fla_config_dir）
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# 确保 tools 目录在 sys.path 中，使直接 python3 执行时也能找到 fla_autotune.utils.*
_SCRIPT_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _SCRIPT_DIR.parent  # tools/
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from fla_autotune.utils.autotune_export import extract_configs
from fla_autotune.utils.autotune_generate import (
    generate_fla_cache,
    get_triton_cache_dir,
)

# autotune 生成过程中禁用 FLA 缓存（避免干扰首次 benchmark）
os.environ.setdefault("FLA_CACHE_MODE", "disabled")

# ============================================================================
# 1. 从模型 config.json 自动检测 head_dim
# ============================================================================


def detect_head_dims(model_path: str) -> list[int]:
    """从 HuggingFace 模型 config.json 中提取 head_dim。

    对于 Qwen3.5 系列（GDN + Attention 混合架构），优先读取 GDN 层的 head_dim；
    如果是纯 Attention 模型，读标准 attention 的 head_dim。

    返回排序去重后的 head_dim 列表。
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print(f"[ERROR] 找不到 {config_path}", file=sys.stderr)
        sys.exit(1)

    configs = []
    with open(config_path) as f:
        json_config = json.load(f)
        configs.append(json_config)
        if "text_config" in json_config:
            configs.append(json_config["text_config"])

    head_dims: set[int] = set()
    for config in configs:
        # GDN 层 key/value head_dim HF 扩展字段
        for key in ("linear_key_head_dim", "linear_value_head_dim"):
            val = config.get(key)
            if val is not None:
                head_dims.add(int(val))
                print(f"[INFO] cfg.{key} = {val}")

        # 标准 attention head_dim
        if not head_dims:
            hd = config.get("head_dim")
            if hd is None:
                hidden = config.get("hidden_size")
                n_heads = config.get("num_attention_heads")
                if hidden and n_heads:
                    hd = hidden // n_heads
            if hd:
                head_dims.add(int(hd))
                print(f"[INFO] attention head_dim = {hd}")

    if not head_dims:
        print("[ERROR] 无法从 config.json 判断 head_dim，请手动指定 -d", file=sys.stderr)
        sys.exit(1)

    return sorted(head_dims)


# ============================================================================
# 2. FLA 缓存生成与提取（来自 extract_triton_autotune_cache.py）
# ============================================================================


def _resolve_output_dir(output_dir: Optional[str] = None, *, versioned: bool = False) -> Path:
    """确定 autotune 配置的输出目录。"""
    if output_dir is not None:
        return Path(output_dir)
    import triton
    from fla.ops.utils.cache import get_fla_config_dir
    resolved = get_fla_config_dir()
    if versioned and "FLA_CONFIG_DIR" not in os.environ:
        return resolved / triton.__version__
    return resolved


def _list_cache_files(triton_cache_dir: Path) -> None:
    """列出已有的 autotune 缓存文件。"""
    if not triton_cache_dir.exists():
        print(f"Triton cache directory not found: {triton_cache_dir}")
        return
    files = list(triton_cache_dir.rglob("*.autotune.json"))
    print(f"Found {len(files)} .autotune.json files in {triton_cache_dir}:\n")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")


# ============================================================================
# 3. 主入口
# ============================================================================


def main() -> None:
    import triton

    parser = argparse.ArgumentParser(
        description="预生成 FLA Triton autotune 缓存并提取最优配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测 Qwen3.5-4B 的 head_dim 并生成 GDN 缓存
  python tools/generate_fla_autotune_cache.py --model-path hf-hub/Qwen/Qwen3.5-4B --op gdn --versioned

  # 手动指定 head_dim
  python tools/generate_fla_autotune_cache.py -d 128 --op gdn --versioned

  # 仅从已有缓存提取（不重新 benchmark）
  python tools/generate_fla_autotune_cache.py --extract-only
        """,
    )

    # ---- head_dim 来源（互斥） ----
    dim_group = parser.add_mutually_exclusive_group()
    dim_group.add_argument(
        "--model-path",
        type=str,
        help="HF 模型路径，自动从 config.json 检测 head_dim",
    )
    dim_group.add_argument(
        "--head-dim",
        "-d",
        type=int,
        nargs="+",
        default=[128],
        metavar="D",
        help="手动指定 head_dim（默认 128），可多个如 -d 64 128 256",
    )

    # ---- 操作模式 ----
    parser.add_argument(
        "--op",
        choices=("kda", "gdn", "both"),
        default="gdn",
        help="FLA kernel 类型 (default: gdn)",
    )
    parser.add_argument(
        "--versioned",
        action="store_true",
        help=f"在输出路径中附带 Triton 版本号（{triton.__version__}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="手动指定输出目录（默认 FLA configs/{GPU}/）",
    )
    parser.add_argument(
        "--triton-cache-dir",
        type=str,
        help="Triton 缓存目录（默认 ~/.triton/cache）",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="仅从已有缓存提取，不运行 kernel benchmark",
    )
    parser.add_argument(
        "--list-only",
        "-l",
        action="store_true",
        help="仅列出已有缓存文件",
    )

    args = parser.parse_args()

    # 确定 head_dim
    if args.model_path:
        head_dims = detect_head_dims(args.model_path)
    else:
        head_dims = list(dict.fromkeys(args.head_dim))  # preserve order, dedupe

    # 确定输出目录
    output_dir = _resolve_output_dir(args.output_dir, versioned=args.versioned)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定 Triton 缓存目录
    if not args.extract_only:
        triton_cache_dir = Path(generate_fla_cache(args.op, head_dims, args.triton_cache_dir))
    else:
        triton_cache_dir = get_triton_cache_dir(args.triton_cache_dir)

    if args.list_only:
        _list_cache_files(triton_cache_dir)
        return

    extract_configs(triton_cache_dir, output_dir)


if __name__ == "__main__":
    main()
