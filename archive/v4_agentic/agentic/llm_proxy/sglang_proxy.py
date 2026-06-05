"""
Sglang LLM proxy for agentic env testing (no Ray / SamplerClient required).

**默认**：向本地已启动的 sglang HTTP 服务发 ``POST {base}/generate``，请求体含 ``input_ids``
（与 sglang Native API 一致），避免每次测试在进程内再起引擎。

Configure via :class:`LLMProxyConfig`:

- ``proxy_type``: ``"sglang"``
- ``proxy_config``:

  - ``sglang_base_url`` (str, optional): 服务根地址，无尾斜杠，默认 ``http://127.0.0.1:34492``。
    也可用别名 ``sglang_url``。
  - ``generate_path`` (str, default ``"/generate"``): 路径片段。
  - ``request_timeout`` (float, default 600): HTTP 超时（秒）。
  - ``max_new_tokens`` (int, default 512)、``temperature`` (float, default 0.7)
  - ``return_logprob`` (bool, default True): 是否向服务端请求 logprob。

**可选进程内引擎**（不推荐日常测试）：设 ``use_in_process_engine: true`` 且提供 ``model_path``，
则使用 ``sglang.Engine``（需本机 GPU 等）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from gpatch_v4.agentic.llm_proxy.base_llm_proxy import BaseLLMProxy, register_llm_proxy

logger = logging.getLogger(__name__)

_DEFAULT_HTTP_BASE = "http://127.0.0.1:34492"


@register_llm_proxy("sglang")
class SglangProxy(BaseLLMProxy):
    """Talks to a running sglang server over HTTP (``/generate`` + ``input_ids``), or optional in-process ``Engine``."""

    def __init__(self, sampler_client, llm_proxy_config, tokenizer, env):
        super().__init__(sampler_client, llm_proxy_config, tokenizer, env)
        self._engine = None

    def _cfg(self) -> Dict[str, Any]:
        return self.llm_proxy_config.proxy_config or {}

    def _http_generate_url(self) -> str:
        cfg = self._cfg()
        base = (cfg.get("sglang_base_url") or cfg.get("sglang_url") or _DEFAULT_HTTP_BASE).rstrip("/")
        path = cfg.get("generate_path", "/generate")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        cfg = self._cfg()
        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError(
                "SglangProxy in-process mode requires proxy_config['model_path'] "
                "and proxy_config['use_in_process_engine']=True."
            )

        try:
            from gpatch_v4.patch.sglang_patch import sglang_hack

            sglang_hack()
        except Exception:
            pass

        import sglang as sgl

        kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "tp_size": int(cfg.get("tp_size", 1)),
            "mem_fraction_static": float(cfg.get("mem_fraction_static", 0.85)),
            "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
        }
        if cfg.get("disable_cuda_graph", True):
            kwargs["disable_cuda_graph"] = True

        self._engine = sgl.Engine(**kwargs)

    def _generate_via_http(self, input_ids: list) -> Optional[Dict[str, Any]]:
        cfg = self._cfg()
        sampling_params = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
            "temperature": float(cfg.get("temperature", 0.7)),
        }
        payload: Dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": bool(cfg.get("return_logprob", False)),
        }
        url = self._http_generate_url()
        body = json.dumps(payload).encode("utf-8")
        timeout = float(cfg.get("request_timeout", 600.0))
        req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            logger.error("SglangProxy HTTP %s failed: %s %s", url, e.code, e.reason)
            try:
                logger.error("Body: %s", e.read().decode("utf-8", errors="replace"))
            except Exception:
                pass
            return None
        except URLError as e:
            logger.error("SglangProxy HTTP %s URLError: %s", url, e)
            return None

        try:
            out = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("SglangProxy invalid JSON from %s: %s", url, e)
            return None

        if not isinstance(out, dict):
            return None
        response_ids = out.get("output_ids")
        if response_ids is None:
            return None

        ret: Dict[str, Any] = {"response_ids": response_ids}
        meta = out.get("meta_info") or {}
        tok_lp = meta.get("output_token_logprobs")
        if tok_lp is not None:
            try:
                logprobs = [float(x[0]) for x in tok_lp]
                ret["output_logprobs"] = torch.tensor(logprobs, dtype=torch.float32)
            except (TypeError, IndexError, ValueError):
                pass
        return ret

    def _generate_in_process(self, input_ids: list) -> Optional[Dict[str, Any]]:
        self._ensure_engine()
        cfg = self._cfg()
        sampling_params = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
            "temperature": float(cfg.get("temperature", 0.7)),
        }
        out = self._engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
        )
        if not isinstance(out, dict):
            return None
        response_ids = out.get("output_ids")
        if response_ids is None:
            return None
        ret: Dict[str, Any] = {"response_ids": response_ids}
        meta = out.get("meta_info") or {}
        tok_lp = meta.get("output_token_logprobs")
        if tok_lp is not None:
            try:
                logprobs = [float(x[0]) for x in tok_lp]
                ret["output_logprobs"] = torch.tensor(logprobs, dtype=torch.float32)
            except (TypeError, IndexError, ValueError):
                pass
        return ret

    def generate(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cfg = self._cfg()
        use_in_process = bool(cfg.get("use_in_process_engine", False))

        prompt_token_ids = data["prompt_token_ids"]
        if hasattr(prompt_token_ids, "tolist"):
            input_ids = prompt_token_ids.tolist()
        else:
            input_ids = list(prompt_token_ids)

        if use_in_process:
            return self._generate_in_process(input_ids)
        return self._generate_via_http(input_ids)

    def shutdown(self):
        if self._engine is not None and hasattr(self._engine, "shutdown"):
            self._engine.shutdown()
            self._engine = None
