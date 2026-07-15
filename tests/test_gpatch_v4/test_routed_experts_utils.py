import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "relative_path",
    [
        "gpatch/core/sampler_v3/routed_experts_utils.py",
        "gpatch_v4/generation_backend/routed_experts_utils.py",
    ],
)
def test_process_routed_experts_supports_dense_payload(monkeypatch, relative_path):
    module_path = Path(__file__).parents[2] / relative_path
    spec = importlib.util.spec_from_file_location("routed_experts_utils_under_test", module_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    values = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)
    payload = {
        "schema_version": 3,
        "format": "dense",
        "values": "base64 payload decoded by sglang",
        "shape": list(values.shape),
        "missing_value": -1,
        "invalid_cache_locs": 0,
    }
    monkeypatch.setattr(
        utils,
        "extract_routed_experts_from_meta_info",
        lambda response: values,
    )
    monkeypatch.setattr(utils, "pybase64", object())
    res = types.SimpleNamespace(
        routed_experts=payload,
        prompt_len=2,
        token_ids=[10, 11],
    )

    out = utils.process_routed_experts(res, num_layers=2, moe_router_topk=2)

    assert out.shape == (4, 2, 2)
    assert out.dtype == torch.int32
    assert torch.equal(out[:-1], torch.from_numpy(values))
    assert torch.equal(out[-1], out[-2])
