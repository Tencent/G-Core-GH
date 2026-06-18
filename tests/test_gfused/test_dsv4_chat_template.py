# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# nrwu@tencent.com
"""Audit the chat_template used by ``tasks/math_dsv4/finetune_dataset.py``
against the *official* DeepSeek-V4 tokenizer encoder shipped inside each
checkpoint (``encoding/encoding_dsv4.py::encode_messages``).

The template under test is imported as ``DSV4_CHAT_TEMPLATE`` from
``gpatch_v4.models.deepseek_v4.chat_template`` (single source of truth). For each
ckpt and each conversation fixture we render via HF
``apply_chat_template`` and compare the resulting token-id sequence
against the oracle's encoded string fed through the same tokenizer. We
cover both ``thinking_mode="chat"`` (default) and ``thinking_mode="thinking"``,
single-turn / multi-turn, with and without a system prompt.

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/transformers/src:$PYTHONPATH"
    pytest -v -s tests/test_gfused/test_dsv4_chat_template.py
"""

import importlib.util
import os
import sys
import unittest
from typing import List, Optional


# ---------------------------------------------------------------------------
# Single source of truth: load ``DSV4_CHAT_TEMPLATE`` directly from
# ``gpatch_v4/models/deepseek_v4/chat_template.py`` via importlib so the
# test does not import ``gpatch_v4.models.deepseek_v4`` package ``__init__``.
# ---------------------------------------------------------------------------
def _load_chat_template_constant() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    path = os.path.join(
        repo_root,
        "gpatch_v4",
        "models",
        "deepseek_v4",
        "chat_template.py",
    )
    if not os.path.isfile(path):
        raise unittest.SkipTest(f"chat template module missing: {path}")
    spec = importlib.util.spec_from_file_location("_dsv4_chat_template", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DSV4_CHAT_TEMPLATE


DSV4_CHAT_TEMPLATE = _load_chat_template_constant()


# ---------------------------------------------------------------------------
# Checkpoints under audit. Both ship the same ``encoding_dsv4.py`` byte-for-
# byte, but we still audit each independently so a future divergence (e.g.
# different special-token ids) cannot slip through unnoticed.
# ---------------------------------------------------------------------------
DSV4_FLASH_CKPT = "hf-hub/deepseek-ai/DeepSeek-V4-Flash"
DSV4_PRO_CKPT = "hf-hub/deepseek-ai/DeepSeek-V4-Pro"


# ---------------------------------------------------------------------------
# Special-token literals expected by DSV4 tokenizer. Full-width bars (U+FF5C)
# and U+2581 (▁) are mandatory -- ASCII '|' tokenizes differently.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = {
    "bos": "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
    "eos": "<\uff5cend\u2581of\u2581sentence\uff5c>",
    "user": "<\uff5cUser\uff5c>",
    "assistant": "<\uff5cAssistant\uff5c>",
    "think_open": "<think>",
    "think_close": "</think>",
}


# ---------------------------------------------------------------------------
# Three message fixtures covering the cases ``finetune_dataset.py`` actually
# produces (single-turn with/without system prompt) plus a defensive
# multi-turn one to catch loop-boundary mistakes.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

FIXTURE_SINGLE_TURN_WITH_SYSTEM = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 17 * 23?"},
]

FIXTURE_SINGLE_TURN_NO_SYSTEM = [
    {"role": "user", "content": "What is 17 * 23?"},
]

FIXTURE_MULTI_TURN_WITH_SYSTEM = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Now compute 17 * 23."},
]

ALL_FIXTURES = [
    ("single_turn_with_system", FIXTURE_SINGLE_TURN_WITH_SYSTEM),
    ("single_turn_no_system", FIXTURE_SINGLE_TURN_NO_SYSTEM),
    ("multi_turn_with_system", FIXTURE_MULTI_TURN_WITH_SYSTEM),
]


# ---------------------------------------------------------------------------
# Real demo data. ``tasks/math_dsv4/yaml/math_sft_fsdp2.yaml`` points at
# ``hf-hub/AI-MO/NuminaMath-CoT-jsonl/train/`` (relative to the gcore-dev
# repo root). Each row has ``problem`` / ``solution`` / ``messages`` / ``source``
# but ``finetune_dataset.py::SimpleDataset.__getitem__`` only consumes the
# (problem, solution) pair, exactly the way this test does.
#
# We sample the *first* N rows to keep the test fast (a few seconds at
# N=64) while still covering enough surface to catch any data-shape bug
# (very long problems, latex, multi-line solutions, etc.) the toy fixtures
# above would not see.
# ---------------------------------------------------------------------------
NUMINA_MATH_REL_DIR = "hf-hub/AI-MO/NuminaMath-CoT-jsonl/train"
NUMINA_MATH_NUM_SAMPLES = int(os.environ.get("DSV4_DEMO_N", "64"))


# ---------------------------------------------------------------------------
# Oracle loader. ``encoding/encoding_dsv4.py`` lives inside each ckpt dir and
# is intentionally *not* on PYTHONPATH, so we load it via a one-shot
# spec_from_file_location. The same loader is reused across both ckpts.
# ---------------------------------------------------------------------------
def _load_numina_math_samples(n: int):
    """Return up to ``n`` (problem, solution) tuples from the first shard of
    ``NuminaMath-CoT-jsonl``. Skips the test cleanly if the data dir is not
    accessible from the test host.

    We read the .jsonl by hand (instead of ``datasets.load_dataset``) so the
    test does not need ``datasets`` installed nor ``HF_DATASETS_CACHE``
    writable.
    """
    import json

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    train_dir = os.path.join(repo_root, NUMINA_MATH_REL_DIR)
    if not os.path.isdir(train_dir):
        raise unittest.SkipTest(f"demo data dir not found: {train_dir}")
    shards = sorted(
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith(".jsonl")
    )
    if not shards:
        raise unittest.SkipTest(f"no .jsonl shards under: {train_dir}")

    samples = []
    with open(shards[0], "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "problem" in row and "solution" in row:
                samples.append((row["problem"], row["solution"]))
            elif "question" in row and "target" in row:
                samples.append((row["question"], row["target"]))
            else:
                continue
            if len(samples) >= n:
                break
    if not samples:
        raise unittest.SkipTest(f"no usable rows in {shards[0]}")
    return samples


def _load_official_encoding(ckpt_dir: str):
    enc_path = os.path.join(ckpt_dir, "encoding", "encoding_dsv4.py")
    if not os.path.isfile(enc_path):
        raise unittest.SkipTest(f"oracle encoder missing: {enc_path}")

    enc_dir = os.path.dirname(enc_path)
    # Some ckpts ``import`` sibling helpers in the same encoding/ dir, so we
    # temporarily put it on sys.path. Restore afterwards to keep the test
    # process clean.
    sys.path.insert(0, enc_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            f"_dsv4_oracle_{os.path.basename(ckpt_dir)}", enc_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(enc_dir)
        except ValueError:
            pass
    return module


def _build_oracle_encoder(ckpt_dir: str):
    """Return the ckpt's ``encode_messages`` callable (the canonical encoder).

    Verified signature (DSV4-Flash/Pro, identical bytes)::

        def encode_messages(
            messages: List[Dict[str, Any]],
            thinking_mode: str,           # "chat" | "thinking"
            context: Optional[...] = None,
            drop_thinking: bool = True,
            add_default_bos_token: bool = True,
            reasoning_effort: Optional[str] = None,
        ) -> str

    Returns a STRING (not ids). The encoder itself does not call any
    tokenizer; its output is meant to be fed to ``tokenizer(text,
    add_special_tokens=False)`` -- exactly what ``finetune_dataset.py`` does.
    """
    module = _load_official_encoding(ckpt_dir)
    if not hasattr(module, "encode_messages"):
        raise unittest.SkipTest(
            f"{ckpt_dir}/encoding/encoding_dsv4.py has no encode_messages"
        )
    return module.encode_messages


# ---------------------------------------------------------------------------
# Mixin: shared assertions parametrised over a ckpt dir. Concrete subclasses
# below bind ``CKPT_DIR`` to either Flash or Pro.
# ---------------------------------------------------------------------------
class _ChatTemplateAlignmentMixin:
    CKPT_DIR: str = ""  # set by subclass

    # ---- setUp ------------------------------------------------------------
    def setUp(self):
        if not os.path.isdir(self.CKPT_DIR):
            raise unittest.SkipTest(f"ckpt dir not found: {self.CKPT_DIR}")
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"transformers missing: {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.CKPT_DIR, trust_remote_code=True
        )
        # ``finetune_dataset.py`` only injects when ``chat_template is None``;
        # the audited ckpts ship without one, so we always inject here. The
        # template is imported from finetune_dataset (single source of truth).
        self.tokenizer.chat_template = DSV4_CHAT_TEMPLATE

        self.encode_messages = _build_oracle_encoder(self.CKPT_DIR)

    # ---- helpers ----------------------------------------------------------
    def _hf_render(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def _hf_ids(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _oracle_text(
        self,
        messages,
        *,
        thinking_mode: str,
        drop_thinking: bool = True,
    ) -> str:
        """Call the official encoder. Returns a *string*, not ids."""
        return self.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=True,
        )

    def _oracle_ids(
        self,
        messages,
        *,
        thinking_mode: str,
        drop_thinking: bool = True,
    ) -> List[int]:
        text = self._oracle_text(
            messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
        )
        return self._hf_ids(text)

    def _print_diff(
        self,
        label: str,
        hf_text: str,
        hf_ids: List[int],
        oracle_ids: List[int],
        oracle_text: Optional[str],
    ) -> None:
        """Print a side-by-side diff to stdout. Called by every alignment
        test so ``pytest -s`` shows the divergence even on FAIL."""
        bar = "=" * 78
        print(f"\n{bar}\n[{self.CKPT_DIR}] :: {label}\n{bar}")
        print(f"-- HF rendered text ({len(hf_text)} chars) --")
        print(repr(hf_text))
        if oracle_text is not None:
            print(f"-- ORACLE decoded text ({len(oracle_text)} chars) --")
            print(repr(oracle_text))
        print(f"-- HF ids     (n={len(hf_ids)}) --")
        print(hf_ids)
        print(f"-- ORACLE ids (n={len(oracle_ids)}) --")
        print(oracle_ids)
        # Per-position first divergence to make debugging fast.
        m = min(len(hf_ids), len(oracle_ids))
        first_div = next(
            (i for i in range(m) if hf_ids[i] != oracle_ids[i]),
            None,
        )
        if first_div is None and len(hf_ids) != len(oracle_ids):
            first_div = m
        if first_div is not None:
            lo = max(0, first_div - 3)
            hi = min(max(len(hf_ids), len(oracle_ids)), first_div + 6)
            print(
                f"-- first divergence @ pos={first_div}; "
                f"HF[{lo}:{hi}]={hf_ids[lo:hi]} "
                f"ORACLE[{lo}:{hi}]={oracle_ids[lo:hi]}"
            )
        print(bar)

    # ---- prompt-only oracle helper ----------------------------------------
    def _oracle_prompt(
        self, messages, *, thinking_mode: str
    ) -> tuple:
        """Return ``(text, ids)`` for the prompt-only oracle render.

        The official encoder, when the *last* message has role=user, already
        appends ``<｜Assistant｜>`` + the appropriate think token -- exactly
        what ``add_generation_prompt=True`` should produce. So we just trim
        any trailing assistant turn from the fixture and let the encoder do
        its thing.
        """
        prompt_msgs = list(messages)
        if prompt_msgs and prompt_msgs[-1]["role"] == "assistant":
            prompt_msgs = prompt_msgs[:-1]
        text = self._oracle_text(prompt_msgs, thinking_mode=thinking_mode)
        ids = self._hf_ids(text)
        return text, ids

    # ---- alignment tests --------------------------------------------------
    def test_chat_mode_alignment(self):
        """Generation-prompt rendering should match oracle in chat mode
        (oracle ends with ``<｜Assistant｜></think>``)."""
        for name, msgs in ALL_FIXTURES:
            with self.subTest(fixture=name):
                hf_text = self._hf_render(
                    msgs, add_generation_prompt=True, enable_thinking=False
                )
                hf_ids = self._hf_ids(hf_text)
                oracle_text, oracle_ids = self._oracle_prompt(
                    msgs, thinking_mode="chat"
                )
                self._print_diff(
                    f"chat_mode :: {name}",
                    hf_text=hf_text,
                    hf_ids=hf_ids,
                    oracle_ids=oracle_ids,
                    oracle_text=oracle_text,
                )
                self.assertEqual(
                    hf_ids,
                    oracle_ids,
                    f"chat-mode prompt ids diverge for fixture={name!r}",
                )

    def test_thinking_mode_alignment(self):
        """Generation-prompt rendering should match oracle in thinking mode
        (oracle ends with ``<｜Assistant｜><think>``)."""
        for name, msgs in ALL_FIXTURES:
            with self.subTest(fixture=name):
                hf_text = self._hf_render(
                    msgs, add_generation_prompt=True, enable_thinking=True
                )
                hf_ids = self._hf_ids(hf_text)
                oracle_text, oracle_ids = self._oracle_prompt(
                    msgs, thinking_mode="thinking"
                )
                self._print_diff(
                    f"thinking_mode :: {name}",
                    hf_text=hf_text,
                    hf_ids=hf_ids,
                    oracle_ids=oracle_ids,
                    oracle_text=oracle_text,
                )
                self.assertEqual(
                    hf_ids,
                    oracle_ids,
                    f"thinking-mode prompt ids diverge for fixture={name!r}",
                )

    def test_special_tokens_present_as_single_id(self):
        """Each DSV4 special token literal must encode to exactly one id; if
        any of them tokenizes into multiple pieces the entire chat_template
        is silently corrupt (full-width chars are easy to typo)."""
        for name, lit in SPECIAL_TOKENS.items():
            with self.subTest(special=name, literal=lit):
                ids = self.tokenizer(lit, add_special_tokens=False).input_ids
                self.assertEqual(
                    len(ids),
                    1,
                    f"special token {name!r}={lit!r} did not map to a single "
                    f"id, got ids={ids}",
                )

    def test_prompt_is_strict_prefix_of_full(self):
        """Label-masking invariant required by ``tokenize_text``: the ids of
        the prompt-only render must be a strict prefix of the ids of the
        prompt+answer render. If this breaks, every assistant token gets
        masked off-by-N during SFT."""
        for name, msgs in ALL_FIXTURES:
            with self.subTest(fixture=name):
                prompt_text = self._hf_render(
                    msgs, add_generation_prompt=True, enable_thinking=False
                )
                full_msgs = list(msgs) + [
                    {"role": "assistant", "content": "answer = 391"}
                ]
                full_text = self._hf_render(
                    full_msgs,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                prompt_ids = self._hf_ids(prompt_text)
                full_ids = self._hf_ids(full_text)
                self.assertLess(
                    len(prompt_ids),
                    len(full_ids),
                    f"prompt not shorter than full for fixture={name!r}",
                )
                self.assertEqual(
                    full_ids[: len(prompt_ids)],
                    prompt_ids,
                    f"prompt ids are not a prefix of full ids for "
                    f"fixture={name!r}: prompt[:8]={prompt_ids[:8]} "
                    f"full[:8]={full_ids[:8]}",
                )

    def test_numina_math_demo_alignment(self):
        """End-to-end alignment on real demo rows from NuminaMath-CoT-jsonl
        -- exactly the dataset ``tasks/math_dsv4/yaml/math_sft_fsdp2.yaml``
        consumes via ``SimpleDataset``.

        For each (problem, solution) row we replay
        ``SimpleDataset._apply_chat_template`` (chat mode, system prompt
        from the yaml, ``custom_add_eos`` defaults to False so the full
        text is rendered via ``apply_chat_template`` again with the closing
        assistant turn) and assert:

          * HF prompt-only ids == oracle prompt-only ids
          * HF full ids        == oracle full ids
          * HF prompt ids are a strict prefix of HF full ids (label-mask
            invariant required by ``tokenize_text``).

        This is the "all-up" check requested by the user: every check that
        passes here means the *real* training pipeline would emit the
        canonical ckpt-side prompt.
        """
        samples = _load_numina_math_samples(NUMINA_MATH_NUM_SAMPLES)
        prompt_mismatch = 0
        full_mismatch = 0
        prefix_mismatch = 0
        first_failure_repr = None

        for idx, (problem, solution) in enumerate(samples):
            base_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]
            full_msgs = base_msgs + [
                {"role": "assistant", "content": solution}
            ]

            # --- HF side (chat mode == enable_thinking=False, the yaml
            # default for this dataset) ---
            hf_prompt_text = self._hf_render(
                base_msgs,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            hf_full_text = self._hf_render(
                full_msgs,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            # ``finetune_dataset.py`` strips a trailing newline; do the same
            # so we test the actual pipeline output.
            hf_full_text = hf_full_text.rstrip("\n")
            hf_prompt_ids = self._hf_ids(hf_prompt_text)
            hf_full_ids = self._hf_ids(hf_full_text)

            # --- oracle side ---
            oracle_prompt_text = self._oracle_text(
                base_msgs, thinking_mode="chat"
            )
            oracle_full_text = self._oracle_text(
                full_msgs, thinking_mode="chat"
            )
            oracle_prompt_ids = self._hf_ids(oracle_prompt_text)
            oracle_full_ids = self._hf_ids(oracle_full_text)

            # --- compare ---
            if hf_prompt_ids != oracle_prompt_ids:
                prompt_mismatch += 1
                if first_failure_repr is None:
                    first_failure_repr = (
                        f"row {idx}: prompt diverge\n"
                        f"  hf_text   = {hf_prompt_text!r}\n"
                        f"  orcl_text = {oracle_prompt_text!r}"
                    )
            if hf_full_ids != oracle_full_ids:
                full_mismatch += 1
                if first_failure_repr is None:
                    first_failure_repr = (
                        f"row {idx}: full diverge "
                        f"(hf_len={len(hf_full_ids)} "
                        f"orcl_len={len(oracle_full_ids)})"
                    )
            if (
                len(hf_prompt_ids) >= len(hf_full_ids)
                or hf_full_ids[: len(hf_prompt_ids)] != hf_prompt_ids
            ):
                prefix_mismatch += 1

        # A single summary print so ``pytest -v`` shows the volume even on
        # success.
        print(
            f"\n[{self.CKPT_DIR}] :: numina_math demo "
            f"n={len(samples)} prompt_mismatch={prompt_mismatch} "
            f"full_mismatch={full_mismatch} "
            f"prefix_mismatch={prefix_mismatch}"
        )
        if prompt_mismatch or full_mismatch or prefix_mismatch:
            self.fail(
                f"NuminaMath demo alignment failed on "
                f"{prompt_mismatch + full_mismatch + prefix_mismatch} of "
                f"{len(samples)} rows. First failure:\n{first_failure_repr}"
            )


# ---------------------------------------------------------------------------
# Concrete per-ckpt test classes.
# ---------------------------------------------------------------------------
class TestDsv4FlashChatTemplate(_ChatTemplateAlignmentMixin, unittest.TestCase):
    CKPT_DIR = DSV4_FLASH_CKPT


class TestDsv4ProChatTemplate(_ChatTemplateAlignmentMixin, unittest.TestCase):
    CKPT_DIR = DSV4_PRO_CKPT


if __name__ == "__main__":
    unittest.main(verbosity=2)
