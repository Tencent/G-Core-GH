"""Throughput probes for SGLang + Qwen3.6-35B-A3B (MoE).

Three test methods:

- ``test_throughput``: 128 slightly-mutated kinabi prompts, decode to 64k position.
- ``test_throughput_random``: 64 synthetic random-int prompts (63k prefill + 1k decode).
- ``test_throughput_jsonl_presstest``: 40 JSONL rows as prompts, decode to 64k position.

Manual-run only -- not in CI. Requires:
- 8 GPUs visible to this process
- Model checkpoint downloaded into ``hf-hub/``

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=3600 \\
        tests/test_gpatch_v4/test_sgl_qwen36moe_throughput.py
"""

import json
import os
import random
import time
import unittest
from typing import Dict, List

import pytest
import torch
from transformers import AutoTokenizer

from test_sgl_qwen36moe_release_resume_v4 import (
    _all_gpu_pids,
    _env_int,
    _print_gpu_processes,
    _snapshot,
)

sgl = pytest.importorskip("sglang")

_KINABI_DATA_JSON = "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/kinabi/data.json"
_PRESSTEST_JSONL_PATH = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/kinabi/qwen3d6-30b-presstest.jsonl"
)


@unittest.skipUnless(
    torch.cuda.device_count() >= 8,
    f"needs 8 GPUs, have {torch.cuda.device_count()}",
)
class Qwen36MoEThroughputTest(unittest.TestCase):
    """Throughput probe for Qwen3.6-35B-A3B with real kinabi data.

    ``_KINABI_DATA_PATH`` is a single JSON object. The test repeats
    ``json_obj['trajectory']['messages']`` 128 times, applies small
    deterministic random substitutions to message content, applies
    ``apply_chat_template``, then generates each request until the total
    position reaches ``QWEN36_TPT_TARGET_TOKENS`` (default 64k).

    Skipped when the kinabi data file is absent.
    """

    REPO_ID = "hf-hub/Qwen/Qwen3.6-35B-A3B"
    MEM_FRACTION_STATIC = 0.7

    def test_throughput(self):
        BATCH = 128
        """Generate 128 slightly-mutated prompts to the 64k position twice."""
        _KINABI_DATA_PATH = _KINABI_DATA_JSON
        if not os.path.isfile(_KINABI_DATA_PATH):
            self.skipTest(f"kinabi data not found: {_KINABI_DATA_PATH}")

        self.skipTest(f"一般不开")

        config_path = os.path.join(self.REPO_ID, "config.json")
        assert os.path.isfile(config_path), (
            f"missing model config at {config_path}; download "
            f"{self.REPO_ID} into hf-hub/ first"
        )

        with open(_KINABI_DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_messages = raw["trajectory"]["messages"]
        assert isinstance(raw_messages, list) and len(raw_messages) > 0

        # Flatten content to plain strings so Jinja |items won't choke on
        # tool-call dicts or structured content entries.
        messages: List[Dict] = []
        for m in raw_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role not in ("user", "assistant", "system"):
                role = "user"
            messages.append({"role": role, "content": content})

        tokenizer = AutoTokenizer.from_pretrained(
            self.REPO_ID,
            trust_remote_code=True,
        )

        target_tokens = _env_int("QWEN36_TPT_TARGET_TOKENS", 65536)
        iters = 2

        batched_input_ids: List[List[int]] = []
        batched_sp: List[Dict] = []
        prompt_lens: List[int] = []
        gen_lens: List[int] = []
        first_templated_len = 0

        for bi in range(BATCH):
            sample_messages = self._mutate_messages(messages, bi)
            templated: str = tokenizer.apply_chat_template(
                sample_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            tokenized = tokenizer.apply_chat_template(
                sample_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = self._extract_input_ids(tokenized)  # [:200] # 如果要测试短 prompt 就这样 cut 下
            assert len(ids) < target_tokens, (
                f"batch {bi}: prompt already has {len(ids)} tokens, "
                f"target is {target_tokens}"
            )
            gen_tokens = 100000  # target_tokens - len(ids)
            batched_input_ids.append(ids)
            batched_sp.append(
                {
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "min_new_tokens": gen_tokens,
                    "max_new_tokens": gen_tokens,
                    "ignore_eos": True,
                }
            )
            prompt_lens.append(len(ids))
            gen_lens.append(gen_tokens)
            if bi == 0:
                first_templated_len = len(templated)

        print(f"\n===== throughput input =====", flush=True)
        print(f"batch: {BATCH}", flush=True)
        print(f"messages per sample: {len(messages)}", flush=True)
        print(f"first text len(chars): {first_templated_len}", flush=True)
        print(
            f"prompt tokens: min={min(prompt_lens)} max={max(prompt_lens)} "
            f"sum={sum(prompt_lens)}",
            flush=True,
        )
        print(f"target tokens: {target_tokens}", flush=True)
        print(
            f"forced new tokens: min={min(gen_lens)} max={max(gen_lens)} "
            f"sum={sum(gen_lens)}",
            flush=True,
        )

        port = 10000 + os.getpid() % 1000
        server_args = sgl.ServerArgs(
            model_path=self.REPO_ID,
            tp_size=8,
            ep_size=8,
            dp_size=1,
            pp_size=1,
            enable_dp_attention=False,
            dist_init_addr=f"127.0.0.1:{port}",
            nnodes=1,
            node_rank=0,
            base_gpu_id=0,
            mem_fraction_static=self.MEM_FRACTION_STATIC,
            trust_remote_code=True,
        )
        llm = sgl.Engine(server_args=server_args)

        try:
            time.sleep(2)
            _snapshot("after_engine_init_tpt")

            decode_tps_list: List[float] = []
            total_tps_list: List[float] = []
            for it in range(1, iters + 1):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = llm.generate(
                    input_ids=batched_input_ids,
                    sampling_params=batched_sp,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                assert isinstance(outputs, list) and len(outputs) == BATCH, (
                    f"iter {it}: expected {BATCH} outputs, got {type(outputs)} "
                    f"len={len(outputs) if isinstance(outputs, list) else 'N/A'}"
                )
                for oi, output in enumerate(outputs):
                    got = len(output["output_ids"])
                    assert got == gen_lens[oi], (
                        f"iter {it} output[{oi}]: expected {gen_lens[oi]} "
                        f"output tokens, got {got}; EOS short-decode would "
                        f"invalidate throughput"
                    )

                decode_tps = sum(gen_lens) / elapsed
                total_tps = (sum(prompt_lens) + sum(gen_lens)) / elapsed
                decode_tps_list.append(decode_tps)
                total_tps_list.append(total_tps)
                print(
                    f"[throughput] iter={it} elapsed={elapsed:.2f}s "
                    f"decode_tok/s={decode_tps:.1f} "
                    f"total_tok/s={total_tps:.1f}",
                    flush=True,
                )

            print(
                f"[throughput] summary: "
                f"decode_tok/s={decode_tps_list} "
                f"total_tok/s={total_tps_list}",
                flush=True,
            )

        finally:
            engine_pids = _all_gpu_pids() - {os.getpid()}
            llm.shutdown()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                live = _all_gpu_pids() & engine_pids
                if not live:
                    break
                time.sleep(0.5)
            _snapshot("after_shutdown_tpt")
            _print_gpu_processes("after_shutdown_tpt")

    def test_throughput_random(self):
        """Throughput probe with synthetic random-int ``input_ids``.

        Skips the tokenizer / kinabi-data path entirely so the test can run
        wherever the model weights are present. Generates ``BATCH`` prompts
        of length ``QWEN36_TPT_RAND_PROMPT_TOKENS`` filled with random ints
        in ``[1, vocab_size)`` (token 0 is reserved to dodge any special
        PAD-id handling), and forces each request to decode
        ``QWEN36_TPT_RAND_GEN_TOKENS`` more tokens with ``ignore_eos=True``.

        Engine config (TP/EP/dp/pp/mem_fraction) mirrors ``test_throughput``
        so decode-tok/s numbers are directly comparable.
        """
        BATCH = 64

        config_path = os.path.join(self.REPO_ID, "config.json")
        assert os.path.isfile(config_path), (
            f"missing model config at {config_path}; download "
            f"{self.REPO_ID} into hf-hub/ first"
        )

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        vocab_size = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
        assert isinstance(vocab_size, int) and vocab_size > 1, (
            f"cannot find a valid vocab_size in {config_path}; "
            f"checked top-level and text_config.vocab_size"
        )

        prompt_tokens = 63 * 1024
        gen_tokens = 1024
        iters = _env_int("QWEN36_TPT_RAND_ITERS", 2)
        assert prompt_tokens > 0, f"prompt_tokens must be > 0, got {prompt_tokens}"
        assert gen_tokens > 0, f"gen_tokens must be > 0, got {gen_tokens}"
        assert iters >= 1, f"iters must be >= 1, got {iters}"

        # Deterministic seed so reruns produce identical workloads (helpful
        # when comparing throughput numbers across configurations).
        rng = random.Random(20260430)
        batched_input_ids: List[List[int]] = [
            [rng.randrange(1, vocab_size) for _ in range(prompt_tokens)] for _ in range(BATCH)
        ]
        batched_sp: List[Dict] = [
            {
                "temperature": 1.0,
                "top_p": 0.9,
                "min_new_tokens": gen_tokens,
                "max_new_tokens": gen_tokens,
                "ignore_eos": True,
            } for _ in range(BATCH)
        ]
        prompt_lens: List[int] = [prompt_tokens] * BATCH
        gen_lens: List[int] = [gen_tokens] * BATCH

        print(f"\n===== throughput random input =====", flush=True)
        print(f"batch: {BATCH}", flush=True)
        print(f"vocab_size: {vocab_size}", flush=True)
        print(
            f"prompt tokens: min={min(prompt_lens)} max={max(prompt_lens)} "
            f"sum={sum(prompt_lens)}",
            flush=True,
        )
        print(
            f"forced new tokens: min={min(gen_lens)} max={max(gen_lens)} "
            f"sum={sum(gen_lens)}",
            flush=True,
        )

        port = 10000 + os.getpid() % 1000
        server_args = sgl.ServerArgs(
            model_path=self.REPO_ID,
            tp_size=8,
            ep_size=8,
            dp_size=1,
            pp_size=1,
            enable_dp_attention=False,
            dist_init_addr=f"127.0.0.1:{port}",
            nnodes=1,
            node_rank=0,
            base_gpu_id=0,
            mem_fraction_static=self.MEM_FRACTION_STATIC,
            trust_remote_code=True,
        )
        llm = sgl.Engine(server_args=server_args)

        try:
            time.sleep(1)
            _snapshot("after_engine_init_tpt_random")

            decode_tps_list: List[float] = []
            total_tps_list: List[float] = []
            for it in range(1, iters + 1):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = llm.generate(
                    input_ids=batched_input_ids,
                    sampling_params=batched_sp,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                assert isinstance(outputs, list) and len(outputs) == BATCH, (
                    f"iter {it}: expected {BATCH} outputs, got {type(outputs)} "
                    f"len={len(outputs) if isinstance(outputs, list) else 'N/A'}"
                )
                actual_gen_lens: List[int] = []
                for oi, output in enumerate(outputs):
                    got = len(output["output_ids"])
                    actual_gen_lens.append(got)
                    assert got == gen_lens[oi], (
                        f"iter {it} output[{oi}]: expected {gen_lens[oi]} "
                        f"output tokens, got {got}; EOS short-decode would "
                        f"invalidate throughput"
                    )
                total_actual = sum(actual_gen_lens)
                total_expected = sum(gen_lens)
                print(
                    f"[throughput-random] iter={it} decode-check: "
                    f"actual_total={total_actual} expected_total={total_expected} "
                    f"all_match={total_actual == total_expected}",
                    flush=True,
                )

                decode_tps = total_expected / elapsed
                total_tps = (sum(prompt_lens) + total_expected) / elapsed
                decode_tps_list.append(decode_tps)
                total_tps_list.append(total_tps)
                print(
                    f"[throughput-random] iter={it} elapsed={elapsed:.2f}s "
                    f"decode_tok/s={decode_tps:.1f} "
                    f"total_tok/s={total_tps:.1f}",
                    flush=True,
                )

            print(
                f"[throughput-random] summary: "
                f"decode_tok/s={decode_tps_list} "
                f"total_tok/s={total_tps_list}",
                flush=True,
            )

        finally:
            engine_pids = _all_gpu_pids() - {os.getpid()}
            llm.shutdown()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                live = _all_gpu_pids() & engine_pids
                if not live:
                    break
                time.sleep(0.5)
            _snapshot("after_shutdown_tpt_random")
            _print_gpu_processes("after_shutdown_tpt_random")

    @staticmethod
    def _extract_input_ids(tokenized) -> List[int]:
        ids = tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            assert len(ids) == 1, f"expected one prompt, got {len(ids)}"
            ids = ids[0]
        assert len(ids) > 0
        assert isinstance(ids, list) and all(isinstance(x, int) for x in ids
                                            ), (f"expected List[int] input_ids, got {type(ids)}")
        return ids

    @staticmethod
    def _mutate_messages(messages: List[Dict], batch_idx: int) -> List[Dict]:
        rng = random.Random(batch_idx)
        out: List[Dict] = []
        content_indices = [
            i for i, m in enumerate(messages)
            if isinstance(m["content"], str) and len(m["content"]) > 0
        ]
        mutate_indices = set(rng.sample(content_indices, min(4, len(content_indices))))
        for i, m in enumerate(messages):
            content = m["content"]
            if i in mutate_indices:
                marker = f"__rep_{batch_idx:03d}_{i:03d}__"
                replace_len = min(len(marker), max(1, len(content) // 200))
                start = rng.randrange(0, max(1, len(content) - replace_len + 1))
                content = content[:start] + marker + content[start + replace_len:]
            out.append({"role": m["role"], "content": content})
        return out

    def test_throughput_jsonl_presstest(self):
        """Read 40 JSONL rows as 40 prompts, batch generate."""
        jsonl_path = _PRESSTEST_JSONL_PATH
        if not os.path.isfile(jsonl_path):
            self.skipTest(f"presstest jsonl not found: {jsonl_path}")

        self.skipTest(f"一般不开")

        num_lines = _env_int("QWEN36_PRESSTEST_LINES", 40)
        target_tokens = _env_int("QWEN36_TPT_TARGET_TOKENS", 65536)
        iters = _env_int("QWEN36_TPT_ITERS", 2)

        objects: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                objects.append(json.loads(line))
                if len(objects) >= num_lines:
                    break

        assert len(objects) == num_lines, (f"need {num_lines} jsonl rows, got {len(objects)}")

        config_path = os.path.join(self.REPO_ID, "config.json")
        assert os.path.isfile(config_path), (
            f"missing model config at {config_path}; download "
            f"{self.REPO_ID} into hf-hub/ first"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.REPO_ID,
            trust_remote_code=True,
        )

        batched_input_ids: List[List[int]] = []
        batched_sp: List[Dict] = []
        prompt_lens: List[int] = []
        gen_lens: List[int] = []

        for bi, obj in enumerate(objects):
            messages = self._normalize_messages(obj)
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = self._extract_input_ids(tokenized)
            assert len(ids) < target_tokens, (
                f"row {bi}: prompt has {len(ids)} tokens, target is {target_tokens}"
            )
            gen_tokens = target_tokens - len(ids)
            batched_input_ids.append(ids)
            batched_sp.append(
                {
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "min_new_tokens": gen_tokens,
                    "max_new_tokens": gen_tokens,
                    "ignore_eos": True,
                }
            )
            prompt_lens.append(len(ids))
            gen_lens.append(gen_tokens)

        batch = len(objects)
        print(f"\n===== presstest jsonl throughput =====", flush=True)
        print(f"jsonl: {jsonl_path}", flush=True)
        print(f"batch: {batch}", flush=True)
        print(
            f"prompt tokens: min={min(prompt_lens)} max={max(prompt_lens)} "
            f"sum={sum(prompt_lens)}",
            flush=True,
        )
        print(f"target tokens per seq: {target_tokens}", flush=True)
        print(
            f"gen tokens: min={min(gen_lens)} max={max(gen_lens)} "
            f"sum={sum(gen_lens)}",
            flush=True,
        )

        port = 10000 + os.getpid() % 1000
        server_args = sgl.ServerArgs(
            model_path=self.REPO_ID,
            tp_size=2,
            ep_size=2,
            dp_size=1,
            pp_size=1,
            enable_dp_attention=False,
            dist_init_addr=f"127.0.0.1:{port}",
            nnodes=1,
            node_rank=0,
            base_gpu_id=0,
            mem_fraction_static=self.MEM_FRACTION_STATIC,
            trust_remote_code=True,
        )
        llm = sgl.Engine(server_args=server_args)

        try:
            time.sleep(2)
            _snapshot("after_engine_init_presstest")

            decode_tps_list: List[float] = []
            total_tps_list: List[float] = []
            for it in range(1, iters + 1):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = llm.generate(
                    input_ids=batched_input_ids,
                    sampling_params=batched_sp,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                assert isinstance(outputs, list) and len(outputs) == batch, (
                    f"iter {it}: expected {batch} outputs"
                )
                for oi, output in enumerate(outputs):
                    got = len(output["output_ids"])
                    assert got == gen_lens[oi], (
                        f"iter {it} output[{oi}]: expected {gen_lens[oi]} "
                        f"tokens, got {got}"
                    )

                decode_tps = sum(gen_lens) / elapsed
                total_tps = (sum(prompt_lens) + sum(gen_lens)) / elapsed
                decode_tps_list.append(decode_tps)
                total_tps_list.append(total_tps)
                print(
                    f"[presstest] iter={it} elapsed={elapsed:.2f}s "
                    f"decode_tok/s={decode_tps:.1f} "
                    f"total_tok/s={total_tps:.1f}",
                    flush=True,
                )

            print(
                f"[presstest] summary decode_tok/s={decode_tps_list} "
                f"total_tok/s={total_tps_list}",
                flush=True,
            )

        finally:
            engine_pids = _all_gpu_pids() - {os.getpid()}
            llm.shutdown()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                live = _all_gpu_pids() & engine_pids
                if not live:
                    break
                time.sleep(0.5)
            _snapshot("after_shutdown_presstest")
            _print_gpu_processes("after_shutdown_presstest")

    @staticmethod
    def _normalize_messages(obj: Dict) -> List[Dict]:
        raw_msgs = (
            obj.get("conversations") or obj.get("messages") or
            obj.get("trajectory", {}).get("messages")
        )
        assert isinstance(raw_msgs, list) and len(raw_msgs) > 0
        messages: List[Dict] = []
        for m in raw_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role not in ("user", "assistant", "system"):
                role = "user"
            messages.append({"role": role, "content": content})
        return messages


if __name__ == "__main__":
    unittest.main()
