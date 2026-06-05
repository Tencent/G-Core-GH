"""Synchronous sandbox HTTP client (VERL / sandbox_fusion style)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Tuple

from tasks.retool.async_agent_loop.reward_utils import (
    RETOOL_TOOL_OUTPUT_MAX_CHARS,
    RETOOL_TOOL_OUTPUT_MAX_LINES,
    prepare_code_for_sandbox,
    truncate_tool_output,
)


def execute_code_in_sandbox_sync(
    code: str,
    sandbox_url: str,
    timeout: int = 20,
    memory_limit_mb: int = 1024,
) -> Tuple[str, bool]:
    code = prepare_code_for_sandbox(code)
    payload = {
        "compile_timeout": 10,
        "run_timeout": timeout,
        "code": code,
        "stdin": None,
        "memory_limit_MB": memory_limit_mb,
        "language": "python",
        "files": {},
        "fetch_files": [],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        sandbox_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout + 15) as resp:
            raw = resp.read().decode("utf-8")
        result = json.loads(raw)
        status = result.get("status", "Failed")
        run_result = result.get("run_result", {}) or {}
        compile_result = result.get("compile_result", {}) or {}
        if status == "Success":
            stdout = run_result.get("stdout", "").strip()
            stdout = truncate_tool_output(
                stdout,
                max_lines=RETOOL_TOOL_OUTPUT_MAX_LINES,
                max_chars=RETOOL_TOOL_OUTPUT_MAX_CHARS,
            )
            return (stdout if stdout else "(no output)", True)
        stderr = run_result.get("stderr", "") if run_result else ""
        compile_err = compile_result.get("stderr", "") if compile_result else ""
        error_msg = stderr or compile_err or "Execution failed"
        error_msg = truncate_tool_output(
            error_msg,
            max_lines=RETOOL_TOOL_OUTPUT_MAX_LINES,
            max_chars=RETOOL_TOOL_OUTPUT_MAX_CHARS,
        )
        return f"Error: {error_msg[:500]}", False
    except urllib.error.HTTPError as e:
        # NOTE @yeazhao 这里好像不太一样
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            detail = str(e)
        return f"Error: HTTP {e.code} {detail}", False
    except Exception as e:
        return f"Error: {str(e)[:200]}", False


def _default_sandbox_url() -> str:
    import os
    return os.environ.get(
        "SANDBOX_FUSION_URL",
        f"http://127.0.0.1:{os.environ.get('SANDBOX_LOCAL_PORT', '8008')}/run_code",
    )


if __name__ == "__main__":
    code = "print(1+1)"
    sandbox_url = _default_sandbox_url()
    timeout = 10
    memory_limit_mb = 1024
    out, ok = execute_code_in_sandbox_sync(code, sandbox_url, timeout, memory_limit_mb)
    print(out)
    print(ok)
