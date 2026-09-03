import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpatch_v4/agentic/env_manager/token_mask_utils.py"
)


def _load_module():
    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizer = object
    previous = sys.modules.get("transformers")
    sys.modules["transformers"] = transformers
    try:
        spec = importlib.util.spec_from_file_location(
            "token_mask_utils_reasoning_effort_test",
            MODULE_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = previous


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]


def test_qwen38_reasoning_effort_reaches_chat_template():
    module = _load_module()
    tokenizer = RecordingTokenizer()

    token_ids = module.custom_apply_chat_template(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        tokenizer=tokenizer,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="medium",
    )

    assert token_ids == [1, 2, 3]
    assert tokenizer.calls[-1][1]["enable_thinking"] is True
    assert tokenizer.calls[-1][1]["reasoning_effort"] == "medium"
