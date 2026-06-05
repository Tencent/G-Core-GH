# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com
import torch

from megatron_datasets.mega_indexed_jsonl_dataset_v3 import (
    MegaIndexedJsonlDatasetV3,
    get_consumed_by_this_worker,
    get_consumed_in_this_domain,
)


class MegaIndexedJsonlDatasetMM(MegaIndexedJsonlDatasetV3):
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

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        if self.ds_list is None:
            self.ds_list = self.make_dasets_for_each_domain()

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
            domain_epoch = get_consumed_by_this_worker(
                get_consumed_in_this_domain(self.consumed_by_this_rank, domain_id), worker_id
            ).epoch
            ret_d = {
                'json_data': example,
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
