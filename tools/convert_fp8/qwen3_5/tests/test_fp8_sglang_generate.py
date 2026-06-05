"""Test FP8 quantized model with SGLang inference engine.

Launches an SGLang server, sends generation requests via the OpenAI-compatible
API, and prints the results.

Usage
-----
    # Auto launch server + generate
    python tools/convert_fp8/qwen3_5/tests/test_fp8_sglang_generate.py --fp8-path /path/to/fp8_model

    # Connect to an already-running server
    python tools/convert_fp8/qwen3_5/tests/test_fp8_sglang_generate.py --server-url http://localhost:30000

    # Custom prompt and TP
    python tools/convert_fp8/qwen3_5/tests/test_fp8_sglang_generate.py \
        --fp8-path /path/to/fp8_model \
        --tp 4 \
        --prompt "Explain quantum computing."
"""

import json
import subprocess
import sys
import time
from argparse import ArgumentParser

import requests

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Write a short Python function that computes fibonacci numbers.",
    "Explain the theory of relativity in one paragraph.",
]


def wait_for_server(base_url: str, timeout: int = 300, interval: int = 5) -> bool:
    """Poll the health endpoint until the server is ready."""

    health_url = f"{base_url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            pass
        print(f"  Waiting for server... ({int(time.time() - start)}s / {timeout}s)")
        time.sleep(interval)
    return False


def generate(base_url: str, prompt: str, max_tokens: int = 128) -> str:
    """Send a chat completion request to the SGLang server."""

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def launch_server(
    model_path: str,
    port: int = 30000,
    tp: int = 1,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch SGLang server as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--quantization",
        "fp8",
        "--port",
        str(port),
        "--tp",
        str(tp),
        "--trust-remote-code",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Launching SGLang server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def main():
    parser = ArgumentParser(description="Test FP8 model with SGLang inference")
    parser.add_argument(
        "--fp8-path", type=str, default=None, help="Path to FP8 checkpoint (auto-launches server)"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="URL of a running SGLang server (skip auto-launch)"
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--prompt", type=str, nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--timeout", type=int, default=300, help="Max seconds to wait for server startup"
    )
    args = parser.parse_args()

    assert args.fp8_path or args.server_url, "Must provide --fp8-path or --server-url"

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    base_url = args.server_url or f"http://localhost:{args.port}"
    server_proc = None

    try:
        if not args.server_url:
            server_proc = launch_server(args.fp8_path, port=args.port, tp=args.tp)
            if not wait_for_server(base_url, timeout=args.timeout):
                print("ERROR: Server failed to start within timeout.")
                server_proc.kill()
                sys.exit(1)

        print(f"\n{'='*60}")
        print(f"SGLang FP8 Generation Test")
        print(f"Server: {base_url}")
        print(f"{'='*60}")

        for i, prompt in enumerate(prompts):
            response = generate(base_url, prompt, max_tokens=args.max_tokens)
            print(f"\n--- Prompt {i+1} ---")
            print(f"Q: {prompt}")
            print(f"A: {response[:500]}{'...' if len(response) > 500 else ''}")

        print("\nAll SGLang generation tests passed.")

    finally:
        if server_proc is not None:
            print("\nShutting down SGLang server...")
            server_proc.kill()
            server_proc.wait()


if __name__ == "__main__":
    main()
