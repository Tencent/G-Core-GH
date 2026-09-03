import json
import os

import torch


def save_data(data, save_dir, file_name, only_save_rank0=False):
    os.makedirs(save_dir, exist_ok=True)
    if only_save_rank0:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(data, os.path.join(save_dir, file_name))
    else:
        torch.save(data, os.path.join(save_dir, file_name))


def _to_scalar_or_list(v):
    """Best-effort convert a tensor/np value into JSON-serializable python."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def save_rollout_jsonl(rollout_batches, tokenizer, save_dir, file_name, metric_keys=None):
    """Dump full rollout trajectories to a JSONL file (one sample per line).

    Each line contains the decoded prompt / response text, their token ids,
    length metadata, and any requested per-sample metrics (e.g. ``rewards``,
    ``success``). ``rollout_batches`` follows the standard training layout:
    a list of dicts each holding ``tokens`` (full prompt+response sequence),
    ``prompt_lengths`` and ``sequence_lengths`` as parallel per-sample lists.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    metric_keys = metric_keys or []

    with open(path, "w", encoding="utf-8") as f:
        for batch_idx, rb in enumerate(rollout_batches):
            tokens_list = rb["tokens"]
            prompt_lengths = rb["prompt_lengths"]
            seq_lengths = rb["sequence_lengths"]
            for i in range(len(tokens_list)):
                toks = tokens_list[i]
                if hasattr(toks, "tolist"):
                    toks = toks.tolist()
                plen = int(_to_scalar_or_list(prompt_lengths[i]))
                slen = int(_to_scalar_or_list(seq_lengths[i]))
                prompt_ids = toks[:plen]
                response_ids = toks[plen:slen]
                record = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "prompt_length": plen,
                    "sequence_length": slen,
                    "response_length": slen - plen,
                    "prompt": tokenizer.decode(
                        prompt_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "response": tokenizer.decode(
                        response_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": response_ids,
                }
                for k in metric_keys:
                    if k in rb and k not in record:
                        record[k] = _to_scalar_or_list(rb[k][i])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
