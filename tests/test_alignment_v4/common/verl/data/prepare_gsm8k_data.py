"""Convert gcore's gsm8k jsonl into verl's RL parquet schema (gsm8k dense alignment).

Mirrors the gcore dense alignment config (model/dense + base) so verl and gcore
train/eval on byte-identical prompts:

  - reads the SAME jsonl gcore consumes via tasks/math_rl_v4/simple_dataset.py
    (hf-hub/openai/gsm8k-jsonl/{train,eval}) with ``question`` / ``answer`` fields.
  - ground truth = the number after ``####`` (gcore ``extract_gt_answer`` regex).
  - prompt = [system_prompt, user(question)]; gcore only prepends the system prompt
    and applies the chat template, it does NOT rewrite the user content. The chat
    template / enable_thinking=False is applied by verl at train time
    (data.apply_chat_template_kwargs.enable_thinking=False in the verl base config).

verl row schema: prompt / data_source / ability / reward_model{ground_truth,style} / extra_info.

Usage
-----
python3 tests/test_alignment_v4/common/verl/data/prepare_gsm8k_data.py  # in-repo defaults
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

# gcore rl_config.yaml data.system_prompt (verbatim).
SYSTEM_PROMPT = (
    "Please reason step by step, keep your reasoning as short as possible, "
    "and put your final answer within \\boxed{}."
)

# gcore tasks/math_rl_v4/simple_dataset.py extract_gt_answer
_GT_RE = re.compile(r"####\s*(-?\d+(\.\d+)?)")


def _load_jsonl(src_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(src_dir, "*.jsonl")))
    assert files, f"no *.jsonl under {src_dir}"
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _extract_gt(answer: str) -> str:
    m = _GT_RE.search(answer)
    assert m is not None, f"no #### answer in: {answer[-80:]!r}"
    return m.group(1)


def convert(src_dir: str, data_source: str) -> pd.DataFrame:
    records = _load_jsonl(src_dir)
    rows = []
    for i, rec in enumerate(records):
        assert "question" in rec and "answer" in rec, f"row {i} malformed: {rec}"
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(rec["question"])},
                ],
                "data_source": data_source,
                "ability": "MATH",
                "reward_model": {"ground_truth": _extract_gt(rec["answer"]), "style": "rule"},
                "extra_info": {"index": str(i)},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    # data -> verl -> common -> test_alignment_v4 -> tests -> repo root
    gcore_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train_src", default=os.path.join(gcore_root, "hf-hub/openai/gsm8k-jsonl/train"))
    parser.add_argument("--val_src", default=os.path.join(gcore_root, "hf-hub/openai/gsm8k-jsonl/eval"))
    parser.add_argument("--out_dir", default=os.path.join(here, "gsm8k"))
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, "train.parquet")
    val_out = os.path.join(out_dir, "val.parquet")

    train_df = convert(args.train_src, "openai/gsm8k")
    train_df.to_parquet(train_out)
    val_df = convert(args.val_src, "openai/gsm8k")
    val_df.to_parquet(val_out)

    print(f"[train] {len(train_df)} rows -> {train_out}")
    print(f"[val]   {len(val_df)} rows -> {val_out}")
    print("sample prompt:", json.dumps(train_df.iloc[0]["prompt"], ensure_ascii=False))
    print("sample gt:", train_df.iloc[0]["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()
