# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com
from copy import deepcopy
from dataclasses import dataclass
from tkinter import N
from typing import Dict, Sequence
from types import SimpleNamespace

import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor

from megatron_datasets.mm_dataset import (
    MultiModalDataset,
    fetch_images,
    convert_conversations,
    remove_bos,
)
from megatron_datasets.utils import print_rank_0, get_iterator
from gpatch.core.utils import split_data_cp_rank


class Gemma3Dataset(MultiModalDataset):
    def __init__(
        self,
        sliding_window,
        use_for_hf,
        mask_history,
        use_grpo,
        num_attention_heads,
        cp_rank,
        cp_size,
        context_parallel_heads_kv_stride,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sliding_window = sliding_window
        self.use_for_hf = use_for_hf
        self.mask_history = mask_history
        self.use_grpo = use_grpo
        self.attn_head_size = num_attention_heads
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        if context_parallel_heads_kv_stride is not None:
            self.attn_head_size = context_parallel_heads_kv_stride

    def gen_attn_mask(self, token_type_ids: torch.Tensor):
        seq_length = len(token_type_ids)
        mask_pair = []
        last_val = 0
        for i in range(len(token_type_ids)):
            if token_type_ids[i] != last_val:
                if token_type_ids[i] == 1:
                    mask_pair.append([i, -1])
                else:
                    mask_pair[-1][-1] = i
                last_val = token_type_ids[i]

        token_type_mask = torch.zeros((seq_length, seq_length), dtype=torch.long, device="cpu")
        for pair in mask_pair:
            token_type_mask[pair[0]:pair[1], pair[0]:pair[1]] = 1

        attn_mask_src = torch.tril(
            torch.ones(
                (seq_length, seq_length),
                dtype=torch.long,
                device="cpu",
            )
        )
        attn_mask = (attn_mask_src | token_type_mask).to(torch.bool)

        slice_mask = torch.tril(
            torch.ones(
                (seq_length, seq_length),
                dtype=torch.long,
                device="cpu",
            ),
            diagonal=-self.sliding_window,
        )

        sliding_window_attention_mask = ((attn_mask_src - slice_mask) | token_type_mask).to(
            torch.bool
        )
        attn_mask = torch.zeros(attn_mask.shape, dtype=torch.bfloat16).masked_fill_(
            attn_mask.logical_not(), float("-inf")
        ).unsqueeze(0)
        attn_mask = attn_mask.expand(self.attn_head_size, *attn_mask.shape[1:])
        sliding_window_attention_mask = torch.zeros(
            sliding_window_attention_mask.shape, dtype=torch.bfloat16
        ).masked_fill_(sliding_window_attention_mask.logical_not(), float("-inf")).unsqueeze(0)
        sliding_window_attention_mask = sliding_window_attention_mask.expand(
            self.attn_head_size,
            *sliding_window_attention_mask.shape[1:],
        )
        # 一般来说，这里的attn-mask就很大了，所以在ds内切分
        # 但grpo不能切分，因为`cp_rank > 0 and tp_rank == 0`时没有读到
        if self.cp_size > 1 and not self.use_grpo:
            attn_mask = split_data_cp_rank(attn_mask, self.cp_size, 1, self.cp_rank)
            sliding_window_attention_mask = split_data_cp_rank(
                sliding_window_attention_mask, self.cp_size, 1, self.cp_rank
            )

        return attn_mask, sliding_window_attention_mask

    def convert_img2tensor(self, imgs):
        res = []
        for img in imgs:
            np_array = np.array(img)
            res.append(torch.from_numpy(np_array))
        return res

    def convert_example(
        self,
        example,
        conversations,
        imgs,
        domain_states,
        tools=None,
        answer=None,
    ):
        add_generation_prompt = False
        if self.use_grpo and conversations[-1]['role'] == "assistant":
            conversations = conversations[:-1]
        if self.use_grpo:
            assert conversations[-1]['role'] != "assistant"
            add_generation_prompt = True
        all_text = self.processor.apply_chat_template(
            conversations,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        all_text = remove_bos(all_text)

        all_text_dict = self.processor(
            text=[all_text],
            images=imgs,
            return_tensors="pt",
        )

        input_ids = all_text_dict['input_ids'].squeeze(0)
        if self.image_token_id is not None:
            assert self.tokenizer._tokenizer.image_token_id == self.image_token_id
        total_image_token = (input_ids == self.tokenizer._tokenizer.image_token_id
                            ).sum().cpu().item()
        token_type_ids = all_text_dict['token_type_ids'].squeeze(0).tolist()
        if 'pixel_values' in all_text_dict:
            pixel_values = all_text_dict['pixel_values']
        else:
            pixel_values = None
        labels = input_ids.clone()
        label_mask = self.gen_label_mask(conversations, imgs, tools)
        if self.use_grpo:
            assert len(label_mask) == 1 and label_mask[0][0] == 0
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100
        prompt_len = label_mask[-1][-1]

        # padding to max length
        input_ids = input_ids.tolist()
        labels = labels.tolist()
        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(token_type_ids)
        if len(input_ids) < self.max_seq_len + 1:
            input_ids += [self.tokenizer._tokenizer.pad_token_id
                         ] * (self.max_seq_len + 1 - len(input_ids))
            token_type_ids += [0] * (self.max_seq_len + 1 - len(token_type_ids))
            labels += [-100] * (self.max_seq_len + 1 - len(labels))
        input_ids = input_ids[:-1]
        token_type_ids = token_type_ids[:-1]
        if self.use_for_hf:
            labels = labels[:-1]
        else:
            labels = labels[1:]
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            token_type_ids = token_type_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]

        attn_mask, sliding_window_attention_mask = self.gen_attn_mask(token_type_ids)
        example["attention_mask"] = attn_mask
        example["sliding_window_attention_mask"] = sliding_window_attention_mask
        example["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        example["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.int64)
        example["labels"] = torch.tensor(labels, dtype=torch.int64)
        if pixel_values is None:
            example["pixel_values"] = pixel_values
        else:
            example["pixel_values"] = pixel_values.to(torch.bfloat16)
        image_input_mask = (example["input_ids"] == self.tokenizer._tokenizer.image_token_id)

        # 增加样本计数
        domain_states.domain_lines += example["domain_line"]
        sum_image_token = image_input_mask.sum().cpu().item()
        if self.use_grpo:
            all_ignore = False
        else:
            all_ignore = torch.all(example["labels"] == -100).item()
        assert total_image_token >= sum_image_token
        # 跳过样本
        if total_image_token > sum_image_token or all_ignore:
            print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}")
            # print(f"Abort Sample {json.dumps(json_data, ensure_ascii=False)}")
            return None

        example["domain_line"] = torch.tensor(domain_states.domain_lines, dtype=torch.int64)
        example["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        domain_states.domain_lines = 0
        if self.use_for_hf:
            for key in [
                'train', 'domain_id', 'worker_id', 'domain_epoch', 'domain_cand_off',
                'sliding_window_attention_mask', 'domain_line', 'prompt_len'
            ]:
                del example[key]

        return example

    def __iter__(self):
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            # json_data 就是从jsonl读出的数据
            json_data = example["json_data"]
            imgs = None
            if 'images' in json_data and len(json_data['images']) > 0:
                imgs = fetch_images(json_data['images'], self.tar_dir, self.lmdb_port)
                imgs_valid = True
                for img in imgs:
                    if img is None:
                        imgs_valid = False
                        break
                if not imgs_valid:
                    domain_states.domain_lines += example["domain_line"]
                    print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}[invalid image]")
                    continue
            conversations = convert_conversations(json_data['conversations'])
            tools = None
            if 'tools' in json_data:
                tools = json_data['tools']
            answer = None
            if 'label' in json_data:
                answer = json_data['label']
            assert len(conversations) > 1
            del example["json_data"]

            example_copy = deepcopy(example)
            example_copy = self.convert_example(
                example_copy, conversations, imgs, domain_states, tools, answer
            )
            if example_copy is None:
                continue

            if self.use_grpo:
                example_copy["json_data"] = json_data
                imgs_np_array = None
                if imgs is not None:
                    imgs_np_array = [np.array(img) for img in imgs]
                example_copy["imgs_np_array"] = imgs_np_array
            yield example_copy


class Gemma3DatasetDPO(Gemma3Dataset):
    def __iter__(self):
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            # json_data 就是从jsonl读出的数据
            json_data = example["json_data"]
            imgs = None
            if 'images' in json_data:
                imgs = fetch_images(json_data['images'], self.tar_dir, self.lmdb_port)
            conversations_chosen = deepcopy(json_data['conversations'])
            conversations_chosen.append(json_data['chosen'])
            conversations_rejected = deepcopy(json_data['conversations'])
            conversations_rejected.append(json_data['rejected'])
            conversations_chosen = convert_conversations(conversations_chosen)
            conversations_rejected = convert_conversations(conversations_rejected)
            assert len(conversations_rejected) > 1 and len(conversations_chosen) > 1
            del example["json_data"]

            tools = None
            if 'tools' in json_data:
                tools = json_data['tools']

            example_chosen = deepcopy(example)
            example_chosen = self.convert_example(
                example_chosen, conversations_chosen, imgs, domain_states, tools
            )
            example_rejected = deepcopy(example)
            example_rejected = self.convert_example(
                example_rejected, conversations_rejected, imgs, domain_states, tools
            )

            if example_chosen is None or example_rejected is None:
                assert domain_states.domain_lines >= 2 * (example["domain_line"])
                domain_states.domain_lines -= example["domain_line"]
                continue
            # example_chosen先运行，它的domain_line是正确的
            if "domain_line" in example_chosen:
                # 只需要记录一个domain_line
                example_rejected["domain_line"] = torch.tensor(0, dtype=torch.int64)
            yield example_chosen
            yield example_rejected


@dataclass
class DataCollatorForGemma3(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        pixel_values = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
            del instance["pixel_values"]

        res = default_collate(instances)
        if len(pixel_values) > 0:
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["has_imgs"] = torch.tensor([1], dtype=torch.int64)
        else:
            res["has_imgs"] = torch.tensor([0], dtype=torch.int64)
        return res


@dataclass
class DataCollatorForGemma3GRPO(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        pixel_values = []
        json_data_list = []
        imgs_np_array_list = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
            del instance["pixel_values"]

            json_data_list.append(instance["json_data"])
            del instance["json_data"]
            imgs_np_array_list.append(instance["imgs_np_array"])
            del instance["imgs_np_array"]

        res = default_collate(instances)
        if len(pixel_values) > 0:
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
        res["json_data_list"] = json_data_list
        res["imgs_np_array_list"] = imgs_np_array_list
        return res


@dataclass
class DataCollatorForGemma3DPO(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) % 2 == 0
        # 正负样本交错出现，要换好顺序
        instances = instances[::2] + instances[1::2]
        pixel_values = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
            del instance["pixel_values"]

        res = default_collate(instances)
        if len(pixel_values) > 0:
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["has_imgs"] = torch.tensor([1], dtype=torch.int64)
        else:
            res["has_imgs"] = torch.tensor([0], dtype=torch.int64)
        return res


def get_processor(args):
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "token": None,
    }
    processor = AutoProcessor.from_pretrained(args.processor_path, **init_kwargs)
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    cp_rank=0,
    cp_size=1,
    use_for_hf=False,
    is_dpo=False,
):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    retention_rates_per_domains = args.px_retention_rates_per_domain
    domain_names = args.px_train_data_domain_names
    enable_pareto = args.px_train_apply_pareto
    pareto_alpha = args.px_train_pareto_alpha
    pareto_scale = args.px_train_pareto_scale
    pareto_score_scale = args.train_pareto_score_scale
    processor = get_processor(args)
    sliding_window = args.sliding_window
    mask_history = args.mask_history
    use_grpo = args.use_grpo
    if use_grpo:
        assert mask_history, f"mask_history must be True when use grpo"

    dataset_class = Gemma3Dataset
    if is_dpo:
        dataset_class = Gemma3DatasetDPO

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = dataset_class(
        sliding_window,
        use_for_hf,
        mask_history,
        use_grpo,
        args.num_attention_heads // args.tensor_model_parallel_size,
        cp_rank,
        cp_size,
        args.context_parallel_heads_kv_stride,
        tokenizer,
        args.decoder_seq_length,
        train_path_likes,
        domain_probabilities,
        domain_names,
        args.global_batch_size,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        num_workers=args.num_workers,
        access_policy_interleave=False,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        train=True,
        retention_rates_per_domains=retention_rates_per_domains,
        unsplit_eval_data=False,
        enable_pareto=enable_pareto,
        pareto_alphas=pareto_alpha,
        pareto_scales=pareto_scale,
        pareto_score_scales=pareto_score_scale,
        top_domains_to_cut=args.px_top_domains_to_cut,
        processor=processor,
        tar_dir=args.tarfile_path,
        lmdb_port=args.lmdb_port,
        image_token_id=args.image_token_id,
    )

    eval_ds = None
    if eval_path_likes is not None:
        # NOTE(guanyouhe): 尝未测试每次eval是否都一样
        eval_ds = dataset_class(
            sliding_window,
            use_for_hf,
            mask_history,
            use_grpo,
            args.num_attention_heads // args.tensor_model_parallel_size,
            cp_rank,
            cp_size,
            args.context_parallel_heads_kv_stride,
            tokenizer,
            args.decoder_seq_length,
            eval_path_likes,
            [1.0],  # 写入概率
            args.px_eval_data_domain_names,
            args.global_batch_size,
            train_data_consuming_progresses=None,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=args.num_workers,
            access_policy_interleave=False,
            shuffle_buffer_size=args.px_shuffle_buffer_size,
            seed=args.seed,
            train=False,
            retention_rates_per_domains=retention_rates_per_domains,
            unsplit_eval_data=False,
            enable_pareto=enable_pareto,
            pareto_alphas=pareto_alpha,
            pareto_scales=pareto_scale,
            pareto_score_scales=pareto_score_scale,
            top_domains_to_cut=args.px_top_domains_to_cut,
            processor=processor,
            tar_dir=args.tarfile_path,
            lmdb_port=args.lmdb_port,
            image_token_id=args.image_token_id,
        )
        assert args.px_reset_dataloader_at_start_of_eval, "需要--px-reset-dataloader-at-start-of-eval来保保证每次eval的数据是一样的"
    test_ds = None

    return train_ds, eval_ds, test_ds


def build_train_valid_test_data_iter(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    cp_rank=0,
    cp_size=1,
    use_for_hf=False,
    is_dpo=False,
):
    train_ds, eval_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        rank,
        dp_rank,
        dp_size,
        cp_rank,
        cp_size,
        use_for_hf=use_for_hf,
        is_dpo=is_dpo,
    )
    batch_size = args.micro_batch_size
    if is_dpo:
        collate_func = DataCollatorForGemma3DPO()
    elif args.use_grpo:
        collate_func = DataCollatorForGemma3GRPO()
        batch_size = args.ppo_rollout_micro_batch_size
    else:
        collate_func = DataCollatorForGemma3()
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_func,
        prefetch_factor=args.px_dataloader_prefetch_factor,
    )

    eval_dataloader = None
    if eval_ds is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    if use_for_hf:
        return train_dataloader, eval_dataloader, test_dataloader
    return get_iterator(train_dataloader), get_iterator(eval_dataloader
                                                       ), get_iterator(test_dataloader)


if __name__ == "__main__":
    dp_rank = 0
    # 先保存args，再load，与llama-factory/transformers对齐时，tp/pp这些要为1
    args = torch.load(f"../Megatron-LM/ckpt_ds/{dp_rank}.pt")
    from megatron.training.tokenizer import build_tokenizer
    tokenizer = build_tokenizer(args)
    print(f"{tokenizer=}")
    train_dataloader, eval_dataloader, test_dataloader = build_train_valid_test_data_iter(
        args,
        tokenizer,
        dp_rank=dp_rank,
        dp_size=8,
        use_for_hf=True,
    )
    print(f"{train_dataloader=} {eval_dataloader=} {test_dataloader=}")
    train_dataloader = iter(train_dataloader)
    for _ in range(10):
        test = next(train_dataloader)
        print(test.keys())
    print(f"Done!")
