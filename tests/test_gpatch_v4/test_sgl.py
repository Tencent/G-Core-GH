import asyncio
import shutil
import unittest

import lipsum
import numpy as np
import pynvml
import pytest
import ray
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

sgl = pytest.importorskip("sglang")
from sglang.utils import async_stream_and_merge, stream_and_merge

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

try:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions
except:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions


class SglTestActor:
    async def test_on_real_data(self):
        # hardcode 了路径，临时 check 下 kinabi 的数据，一般不 test。
        data_l = torch.load(
            '/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/llm-tools/input_ids.pt'
        )
        n_data = len(data_l)
        repeat = 16

        import nest_asyncio
        nest_asyncio.apply()

        num_gpus_per_node = 8
        dist_init_addr = '127.0.0.1:10000'
        nnodes = 1
        node_rank = 0

        tp_size = 8 * nnodes
        ep_size = 1
        dp_size = 1
        pp_size = 1

        repo_id = 'hf-hub/Qwen/Qwen3-Coder-30B-A3B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

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
            skip_tokenizer_init=True,
            mem_fraction_static=0.7,
            max_running_requests=32,
        )
        llm = sgl.Engine(server_args=server_args)

        # 模拟 overhead
        overhead = []
        for i in range(8):
            z = torch.zeros((27000 * 1000 * 1000), dtype=torch.int8, device=f'cuda:{i}')
            overhead.append(z)

        while True:
            llm.release_memory_occupation()
            llm.resume_memory_occupation()

            outp_cos = []
            for data in data_l:
                input_ids = data['input_ids']
                prompt = tokenizer.decode(input_ids)
                sys_prompt = prompt[40:680]
                user_prompt = prompt[680 + 28:-33]

                for rep_i in range(repeat):
                    # await asyncio.sleep(0.1)
                    # print(f'issue a new req')
                    outp_co = asyncio.create_task(
                        llm.async_generate(
                            input_ids=data['input_ids'],
                            sampling_params=data['sampling_params'],
                        )
                    )
                    outp_cos.append(outp_co)

            for t in outp_cos:
                output = await t
                output_ids = output['output_ids']
                output_text = tokenizer.decode(output_ids)
                # print('-' * 80)
                # print(f"Generated text: {output_text}", flush=True)
                # print('-' * 80)

    async def test_gen(self):
        import nest_asyncio
        nest_asyncio.apply()

        num_gpus_per_node = 8
        dist_init_addr = '127.0.0.1:10000'
        nnodes = 1
        node_rank = 0

        tp_size = 8 * nnodes
        ep_size = 8 * nnodes
        dp_size = 1
        pp_size = 1

        repo_id = 'hf-hub/Qwen/Qwen3-30B-A3B'

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
        )
        llm = sgl.Engine(server_args=server_args)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = {"temperature": 1.0, "top_p": 0.9}

        outputs = await llm.async_generate(prompts, sampling_params)
        assert len(outputs) == 4
        print('-' * 80)
        print(f'FIRST GENERATION')
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output['text']}", flush=True)
        print('-' * 80)

        llm.release_memory_occupation()
        llm.resume_memory_occupation()

        outputs = await llm.async_generate(prompts, sampling_params)
        assert len(outputs) == 4
        print('-' * 80)
        print(f'SECOND GENERATION')
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output['text']}", flush=True)
        print('-' * 80)

    async def test_gen_very_long(self):
        import nest_asyncio
        nest_asyncio.apply()

        num_gpus_per_node = 8
        dist_init_addr = '127.0.0.1:10000'
        nnodes = 1
        node_rank = 0

        tp_size = 8 * nnodes
        ep_size = 1  # 8 * nnodes
        dp_size = 1
        pp_size = 1

        repo_id = 'hf-hub/Qwen/Qwen3-Coder-30B-A3B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

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
        )
        llm = sgl.Engine(server_args=server_args)
        llm.release_memory_occupation()
        llm.resume_memory_occupation()

        # 对于 qwen3 的 tokenizer，lorem ipsum 大约是 1 word == 2 tokens 。
        prompt = lipsum.generate_words(4 * 1024)
        prompt_input_ids = tokenizer(prompt)['input_ids']

        repeat = 16
        outp_cos = []
        for rep_i in range(repeat):
            sampling_params = {
                'n': 1,
                "temperature": 1.0,
                "top_p": 0.9,
                'min_new_tokens': 30 * 1024,
                'max_new_tokens': 31 * 1024,
            }
            outp_co = llm.async_generate(prompt, sampling_params)
            outp_cos.append(outp_co)

        outputs = await asyncio.gather(*outp_cos)
        assert len(outputs) == repeat

        print('-' * 80)
        print(f'FIRST GENERATION')
        for output in outputs:
            print(f"Generated text: {output['text']}", flush=True)
        print('-' * 80)


class SglTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    '''
    def test_on_real_data(self):
        test_actor = SglTestActor()
        asyncio.run(test_actor.test_on_real_data())
    '''

    def test_gen(self):
        test_actor = ray.remote(SglTestActor).remote()
        test_actor.test_gen.remote()

    def test_gen_very_long(self):
        test_actor = ray.remote(SglTestActor).remote()
        test_actor.test_gen_very_long.remote()
