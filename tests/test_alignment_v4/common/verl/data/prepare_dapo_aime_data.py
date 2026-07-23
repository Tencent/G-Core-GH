"""Convert a zhuzilin dapo-math-17k / aime-2024 jsonl into verl's RL parquet schema.

Mirrors ``prepare_gsm8k_data.py``, but processes ONE jsonl into ONE parquet (run it
twice: once for dapo-math-17k, once for aime-2024). Unlike the gsm8k jsonl
(``question`` / ``answer`` with a ``####`` ground truth), these datasets are already
in a verl-friendly shape:

  - ``prompt``: a list of chat messages ({role, content}).
  - ``label``: the ground-truth answer string (no ``####`` extraction needed).

To keep verl and gcore byte-identical, we mirror gcore's DAPO dataset
(``tasks/math_rl_v4/dapo_dataset.py``): ``DapoMathDataset._apply_chat_template``
prepends ``config.data.system_prompt`` and keeps the record's user message(s)
verbatim before applying the chat template (``enable_thinking=False``). We do the
same here -- prepend ``SYSTEM_PROMPT`` (identical to the gcore yaml
``data.system_prompt``) -- and verl applies the chat template at train time
(``data.apply_chat_template_kwargs.enable_thinking=False`` in the verl base config).

verl row schema: prompt / data_source / ability / reward_model{ground_truth,style} / extra_info.

Usage
-----
python3 tests/test_alignment_v4/common/verl/data/prepare_dapo_aime_data.py \
    --src hf-hub/zhuzilin/dapo-math-17k/dapo-math-17k.jsonl
python3 tests/test_alignment_v4/common/verl/data/prepare_dapo_aime_data.py \
    --src hf-hub/zhuzilin/aime-2024/aime-2024.jsonl
"""

import argparse
import glob
import json
import os

import pandas as pd

# Must match gcore config/base.yaml ``data.system_prompt`` verbatim so verl and
# gcore feed byte-identical prompts.
SYSTEM_PROMPT = (
    "Please reason step by step, keep your reasoning as short as possible, "
    "and put your final answer within \\boxed{}."
)


def _load_jsonl(src: str) -> list[dict]:
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    else:
        files = [src]
    assert files, f"no jsonl found at {src}"
    rows = []
    for fp in files:
        assert os.path.exists(fp), f"missing jsonl: {fp}"
        with open(fp, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def convert(src: str, data_source: str) -> pd.DataFrame:
    records = _load_jsonl(src)
    rows = []
    for i, rec in enumerate(records):
        assert "prompt" in rec and "label" in rec, f"row {i} malformed: {rec}"
        prompt = rec["prompt"]
        assert isinstance(prompt, list) and prompt, f"row {i} prompt not a message list: {prompt}"
        for msg in prompt:
            assert "role" in msg and "content" in msg, f"row {i} bad message: {msg}"
        messages = [
            {"role": str(msg["role"]), "content": str(msg["content"])} for msg in prompt
        ]
        rows.append(
            {
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                "data_source": data_source,
                "ability": "MATH",
                "reward_model": {"ground_truth": str(rec["label"]), "style": "rule"},
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
    parser.add_argument(
        "--src",
        default=os.path.join(gcore_root, "hf-hub/zhuzilin/dapo-math-17k/dapo-math-17k.jsonl"),
        help="input jsonl file (or a dir of *.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output parquet path (default: <verl data dir>/<stem>.parquet)",
    )
    parser.add_argument(
        "--data_source",
        default=None,
        help="verl data_source tag (default: derived from the src filename)",
    )
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    stem = os.path.splitext(os.path.basename(src.rstrip("/")))[0]
    data_source = args.data_source or f"zhuzilin/{stem}"
    # default: write next to the gsm8k parquets (verl data dir), NOT into the gcore
    # jsonl dir -- gcore reads the raw jsonl, verl reads the parquet.
    out = os.path.abspath(args.out) if args.out else os.path.join(here, f"{stem}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = convert(src, data_source)
    df.to_parquet(out)

    print(f"[{data_source}] {len(df)} rows -> {out}")
    print("sample prompt:", json.dumps(df.iloc[0]["prompt"], ensure_ascii=False))
    print("sample gt:", df.iloc[0]["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()
