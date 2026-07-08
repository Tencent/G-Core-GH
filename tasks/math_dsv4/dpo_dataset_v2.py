"""
dpo dataset for DeepSeek-V4.
"""

import copy
import glob
import os
import re
import sys
import random
import json

from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler

from tasks.math_dsv4.encoding.encoding_dsv4 import (
    bos_token,
    encode_messages,
    merge_tool_messages,
    sort_tool_results_by_call_order,
)

IGNORE_INDEX = -100
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_think(content: str) -> Tuple[str, str]:
    """Pull a single ``<think>...</think>`` block out of ``content``.

    The DeepSeek-V4 contract guarantees at most **one** think block per
    assistant turn, so we use a single ``re.search`` and slice the string
    instead of running ``findall`` + ``sub``.

    Returns ``(reasoning_text, content_without_think)``.  When ``content`` has
    no think block, ``reasoning_text`` is ``""`` and ``content`` is returned
    unchanged.
    """
    if not isinstance(content, str):
        return "", content
    m = THINK_PATTERN.search(content)
    if m is None:
        return "", content
    reasoning_text = m.group(1).strip()
    new_content = (content[: m.start()] + content[m.end() :]).strip()
    return reasoning_text, new_content


def _normalize_messages(
    example: Dict[str, Any],
    default_system_prompt: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Coerce a raw dataset row into ``(messages, tools)`` ready for the
    DSV4 encoder.

    The expected schema is the standard OpenAI chat format::

        {
            "messages": [
                {"role": "system",    "content": "..."},
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "...",
                 "reasoning_content": "...", "tool_calls": [...]},
                {"role": "tool",      "tool_call_id": "...", "content": "..."},
                ...
            ],
            "tools": [ ... ]   # optional, OpenAI tool schema
        }

    Single-turn data is just a degenerate two-message conversation
    (``user`` + ``assistant``) and requires no special handling.

    Assistant turns get their ``<think>...</think>`` lifted into
    ``reasoning_content`` automatically.
    """
    # if "messages" not in example or example["messages"] is None:
    #     raise ValueError(
    #         "Each example must contain a non-empty 'messages' field in OpenAI "
    #         f"chat format; got keys: {list(example.keys())}"
    #     )

    if "conversations" in example:
        messages = copy.deepcopy(example["conversations"])
    else:
        messages = copy.deepcopy(example["messages"])
    
    tools = example.get("tools", None)
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tools JSON string: {tools[:200]!r}") from exc
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"tools must be a list, got {type(tools)}")

    has_system = bool(messages) and messages[0].get("role") == "system"
    if default_system_prompt and not has_system:
        messages.insert(0, {"role": "system", "content": default_system_prompt})

    if tools:
        # The encoder reads ``tools`` from the system / developer message.
        sys_idx = 0 if messages and messages[0].get("role") == "system" else None
        if sys_idx is None:
            messages.insert(0, {"role": "system", "content": ""})
            sys_idx = 0
        messages[sys_idx] = dict(messages[sys_idx])
        messages[sys_idx]["tools"] = tools

    rejected_messages = copy.deepcopy(messages)
    assert rejected_messages[-1].get("role") == "assistant", f"rejected_messages[-1].get('role') != 'assistant': {rejected_messages[-1]}"
    rejected_messages[-1]["content"] = example["rejected_response"]

    def post_process_messages(messages):
        # Move embedded <think> blocks into reasoning_content.
        for m in messages:
            if m.get("role") != "assistant":
                continue
            if m.get("reasoning_content"):
                continue
            rc, new_content = _extract_think(m.get("content") or "")
            if rc:
                m["reasoning_content"] = rc
                m["content"] = new_content
        return messages
    messages = post_process_messages(messages)
    rejected_messages = post_process_messages(rejected_messages)

    return messages, rejected_messages, tools


def _encode_with_offsets(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    thinking_mode: str,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Encode ``messages`` and return ``(input_ids, assistant_token_spans)``.

    ``assistant_token_spans`` is a list of ``(start, end)`` token indices --
    each pair delimits exactly the tokens emitted by one assistant turn that
    is *not* masked out.  ``end`` is exclusive.  These spans are the regions
    that participate in the SFT loss.

    The implementation re-encodes successive prefixes of the conversation and
    uses ``startswith`` on the resulting token id sequences to recover the
    boundary tokens placed by the encoder (e.g. ``<｜Assistant｜>``,
    ``<think>``, ``</think>``, EOS, tool-call wrappers, ...).  This avoids
    duplicating the encoder's bookkeeping here.
    """
    full_prompt = encode_messages(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=False,
        add_default_bos_token=True,
    )
    input_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids

    # Pre-process the way ``encode_messages`` does so that indices line up
    # with ``messages`` after the merge step.
    norm_messages = sort_tool_results_by_call_order(merge_tool_messages(messages))

    spans: List[Tuple[int, int]] = []
    for idx, msg in enumerate(norm_messages):
        if msg.get("role") != "assistant":
            continue
        if msg.get("mask") is True:
            continue

        before_prompt = _encode_prefix(norm_messages, idx, thinking_mode)
        after_prompt = _encode_prefix(norm_messages, idx + 1, thinking_mode)

        before_ids = tokenizer(before_prompt, add_special_tokens=False).input_ids
        after_ids = tokenizer(after_prompt, add_special_tokens=False).input_ids

        # Sanity: ``after_ids`` must extend ``before_ids``.  If the tokenizer
        # is sensitive to the way special tokens merge, fall back to a
        # longest-common-prefix instead of failing hard.
        common = _common_prefix_len(before_ids, after_ids)
        start = common
        end = len(after_ids)

        # Guard against pathological zero-length spans (an assistant with
        # neither reasoning nor content nor tool_calls).
        if end <= start:
            continue

        # Guard: the global tokenization should also start with this prefix.
        # We trust ``after_ids`` because it was produced from exactly the
        # same prefix the global encoder walks through.
        assert end <= len(input_ids), (
            f"assistant span [{start}, {end}) exceeds full sequence "
            f"length {len(input_ids)} -- is the tokenizer non-deterministic?"
        )
        spans.append((start, end))

    return input_ids, spans


def _encode_prefix(
    norm_messages: List[Dict[str, Any]],
    upto: int,
    thinking_mode: str,
) -> str:
    """Re-encode ``norm_messages[:upto]`` with the same flags the dataset
    uses for the full sequence.  ``norm_messages`` is assumed to be already
    normalized (tool messages merged), so we replay the encode without
    invoking ``merge_tool_messages`` a second time -- otherwise the
    tool-result containing user message would be re-merged differently when
    we slice in the middle of a tool block.
    """
    if upto <= 0:
        return bos_token  # ``add_default_bos_token=True`` always emits BOS.
    prefix = bos_token
    for i in range(upto):
        from tasks.math_dsv4.encoding.encoding_dsv4 import render_message  # local import: cheap
        prefix += render_message(
            i,
            norm_messages,
            thinking_mode=thinking_mode,
            drop_thinking=False,
        )
    return prefix


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dsv4DpoDataset(Dataset):
    """SFT dataset that tokenizes with ``encoding_dsv4.encode_messages`` and
    builds segment-aware labels for multi-turn conversations."""

    def __init__(
        self,
        config: FinetuneConfig,
        tokenizer: AutoTokenizer,
        json_pattern: str = "*.jsonl",
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = config.data.data_pathes[0]
        self.train = True
        self.system_prompt = config.data.system_prompt
        self.seq_len = config.training.seq_length
        self.thinking_mode = (
            "thinking" if config.training.enable_thinking else "chat"
        )
        self.rng = random.Random(0)


        self.data_dir  = os.path.join(self.data_dir, split) 
        json_files = sorted(glob.glob(os.path.join(self.data_dir, json_pattern)))
        if not json_files:
            raise FileNotFoundError(
                f"No files matching {json_pattern!r} under {self.data_dir!r}"
            )
        # self.dataset = load_dataset("json", data_files=json_files, split=split)
        self.dataset = []
        for fp in json_files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.dataset.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.dataset)

    def encode_and_return_dict(self, messages):
        input_ids, assistant_spans = _encode_with_offsets(
            self.tokenizer, messages, self.thinking_mode
        )
        if self.config.data.mask_history:
             assistant_spans = [assistant_spans[-1]]
        seq_length = len(input_ids)
        labels = [IGNORE_INDEX] * seq_length
        for start, end in assistant_spans:
            for j in range(start, min(end, seq_length)):
                labels[j] = input_ids[j]

        # ``prompt_len`` is kept for downstream compatibility with the
        # original ``SimpleDataset``: it is the index of the first token that
        # participates in the loss (i.e. start of the first kept assistant
        # span).  Falls back to ``seq_length`` when every turn is masked.
        prompt_len = assistant_spans[0][0] if assistant_spans else seq_length

        def _debug_show(input_ids, labels, seq_length, prompt_len, tokenizer):
            RESET   = "\033[0m"
            GREEN   = "\033[92m"   # label 段（参与 loss）
            GREY    = "\033[90m"   # input 段（被 mask 掉）
            YELLOW  = "\033[93m"   # 头部 prompt 提示
            debug_text = ""
            cur_buf: List[int] = []
            cur_is_label = labels[0] != IGNORE_INDEX
            def _flush(buf, is_label):
                if not buf:
                    return
                text = tokenizer.decode(buf, skip_special_tokens=False)
                color = GREEN if is_label else GREY
                return f"{color}{text}{RESET}"

            for i in range(seq_length):
                is_label = labels[i] != IGNORE_INDEX
                if is_label:
                    assert labels[i] == input_ids[i], f"labels err:\n {labels} \n vs \n {input_ids}"
                if cur_buf and is_label != cur_is_label:
                    debug_text += _flush(cur_buf, cur_is_label)
                    cur_buf = []
                cur_buf.append(input_ids[i])
                cur_is_label = is_label
            debug_text += _flush(cur_buf, cur_is_label)
            print(debug_text)
        
        if self.rng.random() < 0.01:
            _debug_show(input_ids, labels, seq_length, prompt_len, self.tokenizer)

        return {
            "tokens": input_ids,
            "labels": labels,
            "seq_length": seq_length,
            "prompt_len": prompt_len,
        }


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        chosen_messages, rejected_messages, _tools = _normalize_messages(example, self.system_prompt)
        chosen_dict = self.encode_and_return_dict(chosen_messages)
        rejected_dict = self.encode_and_return_dict(rejected_messages)
        return (chosen_dict, rejected_dict)
    

# ---------------------------------------------------------------------------
# Collate / loader plumbing (kept identical to the math_dsv4 baseline)
# ---------------------------------------------------------------------------

def dpo_collate_fn(examples):
    chosen_list = []
    rejected_list = []
    for chosen, rejected in examples:
        chosen_list.append(chosen)
        rejected_list.append(rejected)
    all_samples = chosen_list + rejected_list

    tokens_list = [torch.tensor(s["tokens"], dtype=torch.long) for s in all_samples]
    labels_list = [torch.tensor(s["labels"], dtype=torch.long) for s in all_samples]

    return {
        "tokens": tokens_list,
        "labels": labels_list,
    }


def get_dataset_and_dataloader(
    config: FinetuneConfig = None,
    tokenizer: AutoTokenizer = None,
    dp_rank: int = 0,
    dp_size: int = 1,
    meta_info: Optional[Dict[str, Any]] = None,
):
    dataset = Dsv4DpoDataset(config=config, tokenizer=tokenizer)
    sampler = ResumableDistributedSampler(
        dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
        drop_last=True,
    )
    if meta_info is not None and "resume_step" in meta_info:
        resume_step = meta_info["resume_step"]
        gas = config.training.train_gbs // (dp_size * config.training.train_mbs)
        consumed_batches = resume_step * gas
        sampler.set_start_index(consumed_batches, config.training.train_mbs)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=dpo_collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        multiprocessing_context=config.data.multiprocessing_method,
    )
    return {
        "train_dataset": dataset,
        "train_sampler": sampler,
        "train_dataloader": dataloader,
    }


def get_batched_data(batched_data=None):
    assert batched_data is not None
    return batched_data
