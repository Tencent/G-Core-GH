# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com
import torch

from megatron_datasets.mega_indexed_jsonl_dataset_v3 import (
    MegaIndexedJsonlDatasetV3,
    get_consumed_by_this_worker,
    get_consumed_in_this_domain,
)

from megatron_datasets.utils import (
    print_rank_0,
)


class Qwen2Dataset(MegaIndexedJsonlDatasetV3):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        path_likes,
        domain_probabilities,
        domain_names,
        global_batch_size,
        train_data_consuming_progresses=None,
        rank=0,
        dp_rank=0,
        dp_size=1,
        num_workers=1,
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        retention_rates_per_domains=None,
        unsplit_eval_data=False,
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
        top_domains_to_cut=1,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            path_likes=path_likes,
            domain_probabilities=domain_probabilities,
            domain_names=domain_names,
            global_batch_size=global_batch_size,
            train_data_consuming_progresses=train_data_consuming_progresses,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=num_workers,
            access_policy_interleave=access_policy_interleave,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            train=train,
            retention_rates_per_domains=retention_rates_per_domains,
            unsplit_eval_data=unsplit_eval_data,
            enable_pareto=enable_pareto,
            pareto_alphas=pareto_alphas,
            pareto_scales=pareto_scales,
            pareto_score_scales=pareto_score_scales,
            top_domains_to_cut=top_domains_to_cut,
        )

    def tokenize_text(self, message):
        all_text = self.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            [message[:-1]], tokenize=False, add_generation_prompt=True
        )
        all_text = all_text[0]
        prompt_text = prompt_text[0]
        assert len(all_text) > len(prompt_text)

        answer = all_text[len(prompt_text):]
        prompt_tokenized = self.tokenizer(prompt_text, padding=False)
        answer_tokenized = self.tokenizer(answer, padding=False)

        input_ids = prompt_tokenized.input_ids + answer_tokenized.input_ids
        labels = [-100] * len(prompt_tokenized.input_ids) + answer_tokenized.input_ids

        if len(input_ids) < self.max_seq_len + 1:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len + 1 - len(input_ids))
            labels += [-100] * (self.max_seq_len + 1 - len(labels))

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len - 1:]
            labels = labels[-self.max_seq_len - 1:]

        return input_ids, labels

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True

        while True:
            domain_id = self.global_batch_domain_id[self.domain_cand_off]
            ds = self.ds_list[domain_id]

            try:
                idx = next(ds)
            except StopIteration:
                # 该 domain 空了，重开一次
                self.ds_list[domain_id] = iter(
                    self.create_dataset(domain_id, self.path_likes[domain_id], new_epoch=True)
                )
                ds = self.ds_list[domain_id]
                idx = next(ds)

            fname = idx['data_file_name']
            offset = idx['offset']
            length = idx['length']
            assert idx['domain_id'].item() == domain_id
            worker_id = idx['worker_id']

            example = self.read_and_parse_obj_from_jsonl(fname, offset, length)
            input_ids, labels = self.tokenize_text(example)
            domain_epoch = get_consumed_by_this_worker(
                get_consumed_in_this_domain(self.consumed_by_this_rank, domain_id), worker_id
            ).epoch
            ret_d = {
                'input_ids': torch.tensor(input_ids, dtype=torch.int64),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'train': self.train,
                'domain_id': torch.tensor(domain_id, dtype=torch.int64),
                'worker_id': torch.tensor(worker_id, dtype=torch.int64),
                'domain_epoch': torch.tensor(domain_epoch, dtype=torch.int64),
                'domain_line': 1,
                'domain_cand_off': self.domain_cand_off
            }
            self.domain_cand_off = (self.domain_cand_off + 1) % len(self.global_batch_domain_id)
            yield ret_d

        assert False, 'never reachable'


def build_train_valid_test_datasets(args, tokenizer, rank=0, dp_rank=0, dp_size=1):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    retention_rates_per_domains = args.px_retention_rates_per_domain
    domain_names = args.px_train_data_domain_names
    enable_pareto = args.px_train_apply_pareto
    pareto_alpha = args.px_train_pareto_alpha
    pareto_scale = args.px_train_pareto_scale
    pareto_score_scale = args.train_pareto_score_scale

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = Qwen2Dataset(
        tokenizer,
        args.seq_length,
        train_path_likes,
        domain_probabilities,
        domain_names,
        args.global_batch_size,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
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
    )

    eval_ds = None
    if eval_path_likes is not None:
        # NOTE(guanyouhe): 尝未测试每次eval是否都一样
        eval_ds = Qwen2Dataset(
            tokenizer,
            args.seq_length,
            eval_path_likes,
            [1.0],  # 写入概率
            args.px_eval_data_domain_names,
            args.global_batch_size,
            train_data_consuming_progresses=None,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
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
        )
        assert args.px_reset_dataloader_at_start_of_eval, "需要--px-reset-dataloader-at-start-of-eval来保保证每次eval的数据是一样的"
    test_ds = None

    return train_ds, eval_ds, test_ds
