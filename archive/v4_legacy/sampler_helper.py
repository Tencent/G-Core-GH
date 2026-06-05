# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import copy
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch

from gpatch.rpc import once_rpc
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import log, logging_rank0


@dataclass
class SamplerCallbackHook:
    engine_provider: Callable = None
    gen_rollouts: Callable[[Dict[str, Any]], Dict[str, Any]] = None
    get_sampling_params: Callable = None


class SamplerGenerateMixin:
    @torch.no_grad()
    def generate_rollouts(self, batch_data, sampling_repeat_n):
        if False:
            # hook func or importlib
            pass
        else:
            rollout_batch = self.default_generate_func(batch_data, sampling_repeat_n)

        return rollout_batch

    @torch.no_grad()
    async def default_generate_func(self, batch_data, sampling_repeat_n):
        prompt_token_ids = batch_data["prompt_token_ids"]
        lpad_lens = batch_data["lpad_lens"]
        gt_label = batch_data["gt_label"]

        def get_sampling_params(engine, sampler_config):
            stop_at_token_id = self.tokenizer.eos_token_id
            return engine.get_sampling_params(
                n=1,
                temperature=sampler_config.infer_engine.temperature,
                top_k=sampler_config.infer_engine.top_k if sampler_config.infer_engine.top_k > 0 else -1,
                top_p=sampler_config.infer_engine.top_p,
                max_tokens=sampler_config.infer_engine.generate_max_tokens,
                stop_token_ids=[stop_at_token_id],
                seed=sampler_config.infer_engine.seed,
            )

        sampling_params = get_sampling_params(self.infer_engine, self.config.sampler_config)
        res_gens = []
        for i in range(len(prompt_token_ids)):
            for j in range(sampling_repeat_n):
                tmp_sampling_params = copy.deepcopy(sampling_params)
                tmp_sampling_params.sampling_seed += i * sampling_repeat_n + j
                gen = self.infer_engine.async_generate(prompt_token_ids[i], tmp_sampling_params, str(uuid.uuid4().hex))
                res_gens.append(gen)

        gen_outputs = await self.infer_engine.wait_and_get_async_generate_output(res_gens)

        tokens = []
        seq_lengths = []
        max_response_len = 0
        prompt_lens = []
        gt_label_list = []

        for gi, gen_out in enumerate(gen_outputs):
            i = gi // sampling_repeat_n
            j = gi % sampling_repeat_n
            assert len(gen_out.outputs) == 1
            output_tokens = list(gen_out.outputs[0].token_ids)
            token = prompt_token_ids[i]['prompt_token_ids'] + output_tokens
            assert len(token) <= self.config.train_config.seq_length
            tokens.append(torch.tensor(token, dtype=torch.long))
            seq_lengths.append(torch.tensor(len(token), dtype=torch.long))
            max_response_len = max(max_response_len, len(token))
            prompt_lens.append(lpad_lens[i])
            gt_label_list.append(gt_label[i])

        rollout_batch = {
            'tokens': tokens,
            'sequence_lengths': seq_lengths,
            'prompt_lengths': prompt_lens,
            'gt_label': gt_label_list,
        }
        return rollout_batch


class SamplerServerTestMixin:
    def register_test_routes(self, app):
        @app.post("/test_generate")
        @once_rpc(**self.monitor_kwargs)
        async def test_generate(req_dict):
            prompts = req_dict.pop("prompts")
            if not isinstance(prompts, list):
                prompts = [prompts]
            res_gens = []
            sampling_params = self.infer_engine.get_sampling_params(temperature=0., top_k=1, seed=123, n=1)
            for prompt in prompts:
                input_dict = {
                    'prompt_token_ids': self.tokenizer(prompt, add_special_tokens=False).input_ids,
                }
                res_gens.append(self.infer_engine.async_generate(input_dict, sampling_params, str(uuid.uuid4().hex)))
            outputs = await self.infer_engine.wait_and_get_async_generate_output(res_gens)

            if self.config.sampler_config.infer_engine_impl == 'sglang':
                for output in outputs:
                    output.outputs[0].text = self.tokenizer.decode(
                        output.outputs[0].token_ids, skip_special_tokens=False
                    )
            text_outputs = [prompt + output.outputs[0].text for prompt, output in zip(prompts, outputs)]
            output_token_ids = [output.outputs[0].token_ids for output in outputs]
            log(f"test_generate {text_outputs=} {output_token_ids=}")
            return {"ret": "ok", "text_outputs": text_outputs, "output_token_ids": output_token_ids}

        @app.post("/save_engine_ckpt")
        @once_rpc(**self.monitor_kwargs)
        async def save_engine_ckpt(req_dict):
            save_ckpt_dir = f"{req_dict['save_ckpt_dir']}_{torch.distributed.get_rank()}"
            if not os.path.exists(save_ckpt_dir):
                os.makedirs(save_ckpt_dir, exist_ok=True)

            ret = self.infer_engine.save_engine_ckpt(save_ckpt_dir)
            log(f"saved engine ckpt to {save_ckpt_dir}", rank=0)
            return {"save_ckpt": ret}
