"""
Inference Actor for pure inference workload
"""
import asyncio
import inspect
import json
import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List

import torch.distributed as dist
from torch.utils.data import DataLoader

from megatron.core import mpu

from gpatch_v4.actor.mixin import TokenizerMixin
from gpatch_v4.client.infer_client import InferenceClient
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.parallel_state import init_pg, initlize_parallel_state
from gpatch_v4.utils import import_fn_from_path, is_same_tokenizer, log, logging_rank0


class InferenceWorker(TokenizerMixin):
    async def init(self, config):
        self.config = config

        fake_dist_config = DistConfig()
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:19500",
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=fake_dist_config.torch_dist_timeout_minutes),
        )
        initlize_parallel_state(config, fake_dist_config)
        init_pg(fake_dist_config)

        self.build_tokenizer()
        self.tokenizer = self.actor_tokenizer
        is_same_tokenizer(self.tokenizer, self.sampler_tokenizers[0])
        self.build_dataset_and_dataloader()

        logging_rank0(
            f"{self.__class__.__name__} initialized with {len(self.train_dataset)} samples"
        )

    def build_dataset_and_dataloader(self):
        """Build dataset and dataloader from config."""
        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) == 4,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )

        if cond1:
            fn_ret = fn(
                config=self.config,
                tokenizer=self.tokenizer,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
            )
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

    async def setup_client(self):
        """Setup sampler client for inference."""
        self.sampler_client = InferenceClient(self.config)

    async def sampler_gen_out(self, rbs: List[Dict[str, List[Any]]], sampler_idx):
        """Generate outputs from sampler."""
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            co = self.sampler_client.generate(sampler_idx, rbi, rollout_batch)
            cos.append(co)
        return await asyncio.gather(*cos)

    async def _inference(self):
        """Main inference loop."""
        batched_request_data = []
        dataloader_len = len(self.train_dataloader)
        train_iter = iter(self.train_dataloader)

        # Collect all batches
        for _ in range(dataloader_len):
            data = next(train_iter)
            batched_request_data.append(data)

        logging_rank0(
            f"Inference info: {dataloader_len=}, dataset_len={len(self.train_dataset)}, "
            f"local_batch_size={len(batched_request_data)}"
        )

        # Generate outputs
        sampler_idx = 0

        rbs = await self.sampler_gen_out(batched_request_data, sampler_idx)

        # Expand rollout batches
        #expanded_rbs = expand_rollout_batches(rbs)
        #expected_len = dataloader_len * self.config.training.train_mbs * repeat_n
        #assert len(expanded_rbs) == expected_len, f"{len(expanded_rbs)} != {expected_len}"

        # Save results
        self._save_results(rbs)

        return True

    def _save_results(self, rbs: List[Dict[str, Any]]):
        """Save inference results to file."""
        output_dir = self.config.infer_result.output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"inference_results_{timestamp}_rank{mpu.get_data_parallel_rank()}.jsonl"
        )

        total_samples = sum(len(rb.get('prompt', [])) for rb in rbs)
        logging_rank0(f"Saving {total_samples} results (from {len(rbs)} batches) to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for rb in rbs:
                prompts = rb.get('prompt', [])
                responses = rb.get('response', [])
                prompt_ids = rb.get('prompt_ids', [])
                response_ids = rb.get('response_ids', [])
                queries = rb.get('query', [])
                search_res_list = rb.get('search_res', [])

                for i in range(len(prompts)):
                    result = {
                        'query': queries[i] if i < len(queries) else '',
                        'search_res': search_res_list[i] if i < len(search_res_list) else '',
                        'prompt': prompts[i] if i < len(prompts) else '',
                        'response': responses[i] if i < len(responses) else '',
                        'prompt_ids': prompt_ids[i] if i < len(prompt_ids) else [],
                        'response_ids': response_ids[i] if i < len(response_ids) else [],
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

        logging_rank0(f"Results saved to {output_file}")

    async def inference(self):
        """Run inference with error handling."""
        try:
            await self._inference()
            logging_rank0("Inference completed successfully")
            return True
        except Exception as e:
            log(f"Inference error: {e}")
            traceback.print_exc()
            return False
