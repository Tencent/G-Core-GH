import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpatch_v4/utils/constants.py"
)


def _load_constants_module():
    spec = importlib.util.spec_from_file_location(
        "generate_stop_reason_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_and_context_length_have_distinct_stop_reasons():
    stop_reason = _load_constants_module().GenerateStopReason

    assert stop_reason.MAX_LENGTH is not stop_reason.MAX_GEN_LENGTH
