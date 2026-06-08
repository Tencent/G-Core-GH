"""Minimal sglang smoke test for the WeLM-v4 model.

Mirrors the runnable example at
``sglang/examples/nrwu/test_welmv4.py`` but wraps it into a ray-actor-based
unittest consistent with the surrounding ``tests/test_gpatch_v4`` suite
(see ``test_sgl.py``).

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_sgl_welm_v4.py
"""

import unittest

import ray

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray


class WelmV4SglActor:
    async def test_gen(self):
        # NOTE: sglang 在 module 顶层 ``asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())``，
        # 一旦在 driver 进程 import 过 sglang，``asyncio.run`` 就会创出 uvloop loop，
        # 与 ``nest_asyncio.apply()`` 不兼容（uvloop 不是 BaseEventLoop 子类）。
        # 解决：把 sglang 调用放进独立 ray actor 进程（参考 test_sgl.py），actor 自身
        # event loop 由 ray 管理，且 actor 进程是 fresh import。
        import nest_asyncio
        import sglang as sgl
        from transformers import AutoTokenizer

        nest_asyncio.apply()

        dist_init_addr = '127.0.0.1:10000'
        nnodes = 1
        node_rank = 0

        tp_size = 8 * nnodes
        ep_size = 8 * nnodes
        dp_size = 1
        pp_size = 1

        repo_id = "hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240"

        prompts = [
            "Hello, what is your name?",
            "Who is the president of the United States?",
            "What is the capital of France?",
            "What is the future of AI?",
        ]
        sampling_params = {"temperature": 1.0, "top_p": 0.9}

        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        for pi, p in enumerate(prompts):
            chat = [
                {
                    "role": "user",
                    "content": p
                },
            ]
            prompts[pi] = tokenizer.apply_chat_template(
                chat,
                add_special_tokens=False,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )

        server_args = sgl.ServerArgs(
            model_path=repo_id,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            pp_size=pp_size,
            enable_dp_attention=(dp_size > 1),
            dist_init_addr=dist_init_addr,
            nnodes=nnodes,
            node_rank=node_rank,
            base_gpu_id=0,
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            mem_fraction_static=0.7,
            trust_remote_code=True,
            enable_kv_mirror=True,
            enable_over_encoding=True,
            enable_return_routed_experts=True,
            disable_piecewise_cuda_graph=True,
        )
        llm = sgl.Engine(server_args=server_args)

        outputs = await llm.async_generate(prompts, sampling_params)
        assert len(outputs) == len(prompts)
        print('-' * 80)
        print('FIRST GENERATION')
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output['text']}", flush=True)
        print('-' * 80)

        llm.release_memory_occupation()
        llm.resume_memory_occupation()

        outputs = await llm.async_generate(prompts, sampling_params)
        assert len(outputs) == len(prompts)
        print('-' * 80)
        print('SECOND GENERATION')
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output['text']}", flush=True)
        print('-' * 80)


class WelmV4Test(unittest.TestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_gen(self):
        test_actor = ray.remote(WelmV4SglActor).remote()
        test_actor.test_gen.remote()
