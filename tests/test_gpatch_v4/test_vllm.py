import asyncio
import unittest

import pytest
import ray
import torch

# Skip the entire module when vllm is not installed in the current image.
# Must happen before the top-level ``import vllm`` below or collection fails.
vllm = pytest.importorskip("vllm")

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from random_text import generate_words


async def drain_generator(gen):
    """Consume an async generator and return the final output."""
    output = None
    async for o in gen:
        output = o
    return output


class VllmTestActor:
    async def test_gen(self):
        import nest_asyncio
        nest_asyncio.apply()

        num_gpus_per_node = 8
        nnodes = 1

        tp_size = 8 * nnodes

        repo_id = 'hf-hub/Qwen/Qwen3-30B-A3B'

        engine_args = AsyncEngineArgs(
            model=repo_id,
            dtype='bfloat16',
            distributed_executor_backend="ray",
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            trust_remote_code=True,
            enable_sleep_mode=True,
        )
        llm = AsyncLLM.from_engine_args(engine_args)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = vllm.SamplingParams(
            temperature=1.0,
            top_p=0.9,
            max_tokens=128,
        )

        gens = [
            llm.generate(p, sampling_params, request_id=f"first-{i}")
            for i, p in enumerate(prompts)
        ]
        outputs = await asyncio.gather(*[drain_generator(g) for g in gens])
        assert len(outputs) == 4
        print('-' * 80)
        print('FIRST GENERATION')
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output.outputs[0].text}", flush=True)
        print('-' * 80)

        await llm.reset_prefix_cache()
        await llm.sleep()
        await llm.wake_up()

        gens = [
            llm.generate(p, sampling_params, request_id=f"second-{i}")
            for i, p in enumerate(prompts)
        ]
        outputs = await asyncio.gather(*[drain_generator(g) for g in gens])
        assert len(outputs) == 4
        print('-' * 80)
        print('SECOND GENERATION')
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output.outputs[0].text}", flush=True)
        print('-' * 80)

    async def test_gen_very_long(self):
        import nest_asyncio
        nest_asyncio.apply()

        num_gpus_per_node = 8
        nnodes = 1

        tp_size = 8 * nnodes

        repo_id = 'hf-hub/Qwen/Qwen3-Coder-30B-A3B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

        engine_args = AsyncEngineArgs(
            model=repo_id,
            dtype='bfloat16',
            distributed_executor_backend="ray",
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            trust_remote_code=True,
            enable_sleep_mode=True,
        )
        llm = AsyncLLM.from_engine_args(engine_args)

        await llm.reset_prefix_cache()
        await llm.sleep()
        await llm.wake_up()

        prompt = generate_words(4 * 1024)

        repeat = 16
        sampling_params = vllm.SamplingParams(
            n=1,
            temperature=1.0,
            top_p=0.9,
            min_tokens=30 * 1024,
            max_tokens=31 * 1024,
        )

        gens = [
            llm.generate(prompt, sampling_params, request_id=f"long-{i}") for i in range(repeat)
        ]
        outputs = await asyncio.gather(*[drain_generator(g) for g in gens])
        assert len(outputs) == repeat

        print('-' * 80)
        print('FIRST GENERATION')
        for output in outputs:
            print(f"Generated text: {output.outputs[0].text}", flush=True)
        print('-' * 80)


class VllmTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        ray.shutdown()

    def test_gen(self):
        test_actor = ray.remote(VllmTestActor).remote()
        test_actor.test_gen.remote()

    def test_gen_very_long(self):
        test_actor = ray.remote(VllmTestActor).remote()
        test_actor.test_gen_very_long.remote()
