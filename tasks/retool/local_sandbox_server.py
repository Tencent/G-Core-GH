"""
Local /run_code sandbox server (FOR DEBUG ONLY).

This implements a minimal subset of the SandboxFusion HTTP protocol expected by
`verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api`.

SECURITY WARNING:
- This executes untrusted code. Do NOT expose this service to untrusted networks.
- Use the real SandboxFusion deployment (or stronger isolation like gVisor/Kata/Firecracker)
  for production.

Run (from verl repo root):
  conda activate retool
  python -m uvicorn recipe.retool.local_sandbox_server:app --host 0.0.0.0 --port 8008
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from typing import Any

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse


def _maybe_set_rlimits(memory_limit_mb: int, cpu_seconds: int) -> None:
    """Best-effort resource limits for the child process (Linux only)."""
    try:
        import resource  # unix only

        # Address space limit (bytes). Note: RLIMIT_AS is not a hard guarantee for RSS.
        if memory_limit_mb and memory_limit_mb > 0:
            mem_bytes = int(memory_limit_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        # CPU time limit (seconds). Wall-time is enforced separately by asyncio timeout.
        if cpu_seconds and cpu_seconds > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds)))
    except Exception:
        # Don't fail the request if rlimit isn't available.
        return


def _compile_python(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<sandbox>", "exec")
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def _get_max_concurrency() -> int:
    # NOTE: This local server is for debug only; keep the default small to avoid host OOM.
    # You can override at runtime:
    #   export SANDBOX_MAX_CONCURRENCY=4
    raw = os.environ.get("SANDBOX_MAX_CONCURRENCY", "64")
    try:
        v = int(raw)
    except Exception:
        v = 8
    return max(1, v)


_sandbox_semaphore = asyncio.Semaphore(
    _get_max_concurrency()
)  # Limit concurrent sandbox subprocesses


def _kill_process_group(pid: int) -> None:
    """Best-effort kill for the whole sandbox process group."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            return


async def _run_python(code: str, stdin: str | None, run_timeout: int,
                      memory_limit_mb: int) -> dict[str, Any]:
    # Acquire semaphore before starting subprocess
    async with _sandbox_semaphore:
        fd, path = tempfile.mkstemp(suffix=".py", prefix="sandbox_", text=True)
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)

            t0 = time.time()
            # IMPORTANT:
            # - Do NOT use preexec_fn with asyncio/uvicorn: it can deadlock in multi-threaded environments.
            # - We still want memory limits for debug safety, so we prefer Linux `prlimit` as a wrapper.
            # - Run subprocess in its own session so we can kill the whole process group on timeout.
            mem_bytes = int(
                memory_limit_mb
            ) * 1024 * 1024 if memory_limit_mb and memory_limit_mb > 0 else 0
            cmd: list[str]
            if shutil.which("prlimit") and (mem_bytes > 0 or (run_timeout and run_timeout > 0)):
                cmd = ["prlimit"]
                if mem_bytes > 0:
                    cmd.append(f"--as={mem_bytes}")
                if run_timeout and run_timeout > 0:
                    # CPU seconds (wall-time still enforced by asyncio timeout below)
                    cmd.append(f"--cpu={max(1, int(run_timeout))}")
                cmd += ["--", sys.executable, path]
            else:
                cmd = [sys.executable, path]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
                start_new_session=True,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=(stdin or "").encode("utf-8")),
                    timeout=max(1, int(run_timeout)),
                )
                dt = time.time() - t0
                return {
                    "status": "Finished" if proc.returncode == 0 else "Error",
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "return_code": int(proc.returncode),
                    "execution_time": dt,
                }
            except asyncio.TimeoutError:
                try:
                    _kill_process_group(proc.pid)
                except Exception:
                    pass
                dt = time.time() - t0
                return {
                    "status": "TimeLimitExceeded",
                    "stdout": "",
                    "stderr": f"TimeLimitExceeded: run_timeout={run_timeout}s\n",
                    "return_code": -1,
                    "execution_time": dt,
                }
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass


app = FastAPI()


@app.post("/run_code")
async def run_code(request: Request):
    """
    Accepts SandboxFusion-compatible payload:
      - code (str)
      - stdin (str|None)
      - compile_timeout (int)
      - run_timeout (int)
      - memory_limit_MB (int)
      - language (str)
    Returns SandboxFusion-like response:
      - status: "Success" | "Failed"
      - compile_result: {...}
      - run_result: {...}
    """
    req = await request.json()
    print(f"recv request {req}")
    code = req.get("code", "")
    stdin = req.get("stdin", None)
    compile_timeout = int(req.get("compile_timeout", 10) or 10)
    run_timeout = int(req.get("run_timeout", 10) or 10)
    memory_limit_mb = int(req.get("memory_limit_MB", 1024) or 1024)
    language = (req.get("language") or "python").lower()

    # Minimal support: python only for debugging.
    if language not in {"python", "python_gpu"}:
        return JSONResponse(
            content={
                "status": "Failed",
                "compile_result":
                    {
                        "status": "Error",
                        "stdout": "",
                        "stderr": f"Unsupported language: {language}\n",
                        "return_code": 1,
                        "execution_time": 0.0,
                    },
                "run_result": None,
            },
            status_code=200,
        )

    # "Compile" stage: syntax check.
    t0 = time.time()
    ok, compile_err = _compile_python(code)
    compile_dt = time.time() - t0
    compile_result: dict[str, Any] = {
        "status": "Finished" if ok else "Error",
        "stdout": "",
        "stderr": compile_err if not ok else "",
        "return_code": 0 if ok else 1,
        "execution_time": compile_dt,
    }

    if not ok:
        return JSONResponse(
            content={
                "status": "Failed",
                "compile_result": compile_result,
                "run_result": None
            },
            status_code=200,
        )

    # Run stage.
    try:
        run_result = await _run_python(
            code=code, stdin=stdin, run_timeout=run_timeout, memory_limit_mb=memory_limit_mb
        )
    except Exception:
        # Catch internal errors to avoid 500 responses, ensuring the client sees the error details.
        err_msg = traceback.format_exc()
        print(f"Error in _run_python:\n{err_msg}", file=sys.stderr)
        run_result = {
            "status": "Error",
            "stdout": "",
            "stderr": f"Internal Sandbox Error:\n{err_msg}",
            "return_code": -1,
            "execution_time": 0.0,
        }

    top_status = "Success" if run_result.get("status") == "Finished" else "Failed"
    return JSONResponse(
        content={
            "status": top_status,
            "compile_result": compile_result,
            "run_result": run_result
        }
    )


if __name__ == "__main__":
    # Convenience entrypoint:
    #   python recipe/retool/local_sandbox_server.py
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("SANDBOX_LOCAL_PORT", "8008")),
        log_level="info"
    )
