"""SGLang E2E smoke test for WeLM-v4.5 and Qwen3.5-27B.

The test runs three subcases on one 8-GPU node:
1. WeLM-v4.5 normal generation.
2. WeLM-v4.5 generation with routed experts returned.
3. Qwen3.5-27B normal generation.

Usage:
    pytest -q -s --timeout=2400 \
        tests/test_gpatch_v4/test_sgl_welm_v45_router_replay.py
"""

import asyncio
import importlib
import importlib.util
import os
import socket
import types
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig


WELM_MODEL_PATH = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/wechat/"
    "WeLM-v4.5-80B-A3B-Instruct-0608/"
)
QWEN_MODEL_PATH = "/mnt/wfs/mmhuizhouwfssz/llmmodels/Qwen/Qwen3.5-27B/"
PROMPT = "请用一句话解释为什么天空是蓝色的。"


def _load_process_routed_experts():
    module_path = (
        Path(__file__).parents[2]
        / "gpatch_v4/generation_backend/routed_experts_utils.py"
    )
    spec = importlib.util.spec_from_file_location("routed_experts_utils_e2e", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_routed_experts


def _free_dist_init_addr():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


def _config_attr(config, *names):
    text_config = getattr(config, "text_config", config)
    for name in names:
        value = getattr(text_config, name, None)
        if value is not None:
            return int(value)
    raise AssertionError(f"none of {names} exists in model config")


def _server_args(sgl, model_path, *, return_routed_experts):
    from gpatch_v4.generation_backend.sglang_engine import filter_server_args_kwargs
    kwargs = dict(
        model_path=model_path,
        tp_size=8,
        ep_size=1,
        dp_size=1,
        pp_size=1,
        enable_dp_attention=False,
        dist_init_addr=_free_dist_init_addr(),
        nnodes=1,
        node_rank=0,
        base_gpu_id=0,
        enable_memory_saver=True,
        enable_weights_cpu_backup=True,
        mem_fraction_static=float(os.environ.get("SGL_E2E_GPU_MEM", "0.7")),
        trust_remote_code=True,
    )
    if return_routed_experts:
        kwargs.update(
            enable_over_encoding=True,
            enable_return_routed_experts=True,
        )
    return sgl.ServerArgs(**filter_server_args_kwargs(kwargs))


async def _generate(llm, *, return_routed_experts):
    outputs = await llm.async_generate(
        [PROMPT],
        {
            "temperature": 0.0,
            "min_new_tokens": 4,
            "max_new_tokens": 16,
            "ignore_eos": True,
        },
        return_routed_experts=return_routed_experts,
    )
    assert len(outputs) == 1
    output = outputs[0]
    assert output["output_ids"], "SGLang returned no generated tokens"
    assert output["text"], "SGLang returned empty generated text"
    return output


def _assert_routed_experts(output, model_path):
    raw = output["meta_info"].get("routed_experts")
    assert isinstance(raw, dict), f"expected new routed-experts dict, got {type(raw)}"

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = _config_attr(config, "num_hidden_layers", "n_layer", "num_layers")
    topk = _config_attr(
        config,
        "num_experts_per_tok",
        "moe_router_topk",
        "num_selected_experts",
        "moe_topk",
    )
    result = types.SimpleNamespace(
        routed_experts=raw,
        prompt_len=output["meta_info"]["prompt_tokens"],
        token_ids=output["output_ids"],
    )

    routed_experts = _load_process_routed_experts()(result, num_layers, topk)

    assert routed_experts.shape == (
        result.prompt_len + len(result.token_ids),
        num_layers,
        topk,
    )
    assert routed_experts.dtype == torch.int32
    assert bool((routed_experts >= 0).any()), "routed experts contain no valid expert id"


async def _run_e2e(sgl, welm_model_path, qwen_model_path):
    welm = sgl.Engine(
        server_args=_server_args(
            sgl,
            welm_model_path,
            return_routed_experts=True,
        )
    )
    try:
        # Subcase 1: WeLM-v4.5 normal generation.
        await _generate(welm, return_routed_experts=False)

        # Subcase 2: WeLM-v4.5 generation with the new expert payload.
        output = await _generate(welm, return_routed_experts=True)
        _assert_routed_experts(output, welm_model_path)
    finally:
        welm.shutdown()

    qwen = sgl.Engine(
        server_args=_server_args(
            sgl,
            qwen_model_path,
            return_routed_experts=False,
        )
    )
    try:
        # Subcase 3: Qwen3.5-27B normal generation.
        await _generate(qwen, return_routed_experts=False)
    finally:
        qwen.shutdown()


def test_sglang_generation_e2e():
    welm_model_path = os.environ.get("WELM45_MODEL_PATH", WELM_MODEL_PATH)
    qwen_model_path = os.environ.get("QWEN35_MODEL_PATH", QWEN_MODEL_PATH)
    for model_path in (welm_model_path, qwen_model_path):
        if not os.path.isdir(model_path):
            pytest.skip(f"model directory not found: {model_path}")
    if torch.cuda.device_count() < 8:
        pytest.skip(f"requires 8 GPUs, found {torch.cuda.device_count()}")

    sgl = pytest.importorskip("sglang")

    # To AI: 不要 skip，如果没有 welm in sglang 就让他 fail 就好。

    try:
        patch_torch = importlib.import_module("sglang.srt.patch_torch")
    except ImportError:
        patch_torch = importlib.import_module("sglang.srt.utils.patch_torch")
    patch_torch.monkey_patch_torch_reductions()

    asyncio.run(_run_e2e(sgl, welm_model_path, qwen_model_path))
