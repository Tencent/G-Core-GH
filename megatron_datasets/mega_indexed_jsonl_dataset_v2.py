# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com
# NOTE: DEPRECATED, DON'T USE IT.

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from types import SimpleNamespace
import json
import copy

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from datasets import load_dataset

from megatron_datasets.utils import print_rank_0

from megatron_datasets.indexed_jsonl_pretrain_dataset import PretrainDataset


def get_domain_epoch_and_line(consuming_progresses, rank, num_domains):
    domain_consumed = [
        SimpleNamespace(epoch=0, line=0, curr_pointer=-1) for i in range(num_domains)
    ]

    if consuming_progresses is not None:
        prg = consuming_progresses.setdefault(rank, domain_consumed)
        print(f"get_domain_epoch_and_line rank {rank} {prg}")
        return copy.deepcopy(prg)

    return domain_consumed


def update_domain_consumed(consuming_progresses, rank, num_domains, data):
    if data is None:
        return

    if not data['train'][0].item():
        return
    domain_consumed = [
        SimpleNamespace(epoch=0, line=0, curr_pointer=-1) for i in range(num_domains)
    ]
    prg = consuming_progresses.setdefault(rank, domain_consumed)

    curr_pointer = data["curr_pointer_position"][-1].item()
    for domain_id in range(num_domains):
        prev_epoch = prg[domain_id].epoch
        # 所有 domain 的 curr_pointer 一起更新，免得出现错乱
        prg[domain_id].curr_pointer = curr_pointer
        mask = data["domain_id"] == domain_id

        if data["domain_epoch"][mask].numel() == 0:
            continue
        max_epoch = data["domain_epoch"][mask].max().item()
        if max_epoch != prev_epoch:
            prg[domain_id].epoch = max_epoch
            prg[domain_id].line = 0
        max_epoch_mask = (data["domain_id"] == domain_id) & (data["domain_epoch"] == max_epoch)
        n_packed_sum = data["n_packed"][max_epoch_mask].sum().item()
        prg[domain_id].line += n_packed_sum


def adjust_domain_id_list_for_dp(gbs, dp_rank, dp_world_size, domain_id_list):
    fake_ring_domain_id_list = domain_id_list + domain_id_list
    offset = gbs // dp_world_size
    start_index = offset * dp_rank
    end_index = start_index + gbs
    return fake_ring_domain_id_list[start_index:end_index]


def generate_global_batch_domain_id(
    gbs, domains_probs, dp_rank, dp_world_size, top_domains_to_cut=1
):
    assert gbs > dp_world_size and gbs % dp_world_size == 0
    domains_num = len(domains_probs)
    domain_samples = [max(1, int(p * gbs)) for p in domains_probs]
    total_samples = sum(domain_samples)

    # 如果总样本数大于/小于全局批次大小，从样本数最大 cut_domain 个域中减去/加上对应数量的样本额度
    if total_samples != gbs:
        top_domain_indices = sorted(
            range(len(domain_samples)), key=lambda i: domain_samples[i], reverse=True
        )[:top_domains_to_cut]
        differences = abs(total_samples - gbs)
        assert differences > 0
        for index in top_domain_indices:
            cut_nums = differences // top_domains_to_cut + int(
                index < differences % top_domains_to_cut
            )
            assert cut_nums < domain_samples[index], f"somethong wrong"
            # 上面 assert 会触发的情况是 domain多，gbs 小，比如domains_num = 20, gbs = 10, 分都不够分，
            # 那么需要认为调整，或者完全没有必要用此 dataset
            if total_samples < gbs:
                domain_samples[index] += cut_nums
            else:
                domain_samples[index] -= cut_nums

    domain_id_list = []
    for domain_id, num_samples in enumerate(domain_samples):
        domain_id_list.extend([domain_id] * num_samples)
    assert len(domain_id_list) == gbs, f"samples num mismatch"

    # TODO(xiaotaoliu): 跟宇哥讨论是否对 domain_id_list 进行 shuffle？好像必要性不太高
    return adjust_domain_id_list_for_dp(gbs, dp_rank, dp_world_size, domain_id_list=domain_id_list)


def add_domain_id(domain_id, example):
    example["domain_id"] = torch.tensor(domain_id, dtype=torch.int64)
    return example


def tokenize_text(tokenizer, text):
    y = tokenizer(f'{tokenizer.bos_token}{text}{tokenizer.eos_token}', add_special_tokens=False)
    input_ids = y.input_ids
    attention_mask = y.attention_mask
    return input_ids, attention_mask


class MegaIndexedJsonlDatasetV2(TorchIterableDataset):
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
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        retention_rates_per_domains=[],
        unsplit_eval_data=False,
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
        top_domains_to_cut=1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        assert isinstance(path_likes, list)
        if domain_probabilities is not None:
            assert len(domain_probabilities) == len(path_likes)
        self.path_likes = path_likes  # `hf.datasets.PathLike`
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.rank = rank
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.global_batch_size = global_batch_size
        assert top_domains_to_cut <= len(domain_names)
        self.top_domains_to_cut = top_domains_to_cut

        # `access_policy_interleave` 不是指混合数据集 interleave。
        self.access_policy_interleave = access_policy_interleave
        assert self.access_policy_interleave is False, "not implemented yet"

        self.train = train
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_data_consuming_progresses = train_data_consuming_progresses
        self.domain_consumed = get_domain_epoch_and_line(
            train_data_consuming_progresses, self.rank, len(path_likes)
        )

        self.retention_rates_per_domains = retention_rates_per_domains
        self.unsplit_eval_data = unsplit_eval_data
        self.flag_on_pareto_sampling = False
        self.enable_pareto = enable_pareto
        self.pareto_alphas = pareto_alphas
        self.pareto_scales = pareto_scales
        self.pareto_score_scales = pareto_score_scales
        if len(self.enable_pareto) > 0:
            assert self.train, f"only train dataset can enable pareto"
            assert len(domain_probabilities) == len(self.enable_pareto)
            assert len(domain_probabilities) == len(self.pareto_alphas)
            assert len(domain_probabilities) == len(self.pareto_scales)
            assert len(domain_probabilities) == len(self.pareto_score_scales)
            self.flag_on_pareto_sampling = True

        self.print_domain_id_map()
        self.ds_list = self.make_ds_list()

        # TODO(xiaotaoliu): 热启相关，需要记录 curr_pointer_position 和 global_batch_domain_id
        # 如果 global_batch_domain_id 不做 shuffle 的话，那么只需要记录 curr_pointer_position
        # TODO: eval data 没有 domain_probabilities，考虑一下怎么做
        self.global_batch_domain_id = generate_global_batch_domain_id(
            gbs=global_batch_size,
            domains_probs=domain_probabilities,
            dp_rank=dp_rank,
            dp_world_size=dp_size,
            top_domains_to_cut=top_domains_to_cut
        )
        if self.dp_rank in [0, 1]:
            print(f"Dataset init global_batch_domain_id {dp_rank} {self.global_batch_domain_id}")

        # 继续开始读数据，应从上一步 pointer 进行叠加 1
        self.curr_pointer_position = (
            self.domain_consumed[0].curr_pointer + 1
        ) % self.global_batch_size

        self.in_iter = False
        self.eval_file_cache = {}
        self.train_file_cache = {}
        self.skip_consumed_data()

    def print_domain_id_map(self):
        domain_id_map = []
        for domain_id, path_like in enumerate(self.path_likes):
            d = {
                'domain_id': domain_id,
                'domain_name': self.domain_names[domain_id],
                'domain_path_like': path_like,
            }
            if self.domain_probabilities:
                d['domain_probabilities'] = self.domain_probabilities[domain_id]
            if self.flag_on_pareto_sampling:
                d['enable_pareto'] = self.enable_pareto[domain_id]
                d['pareto_alphas'] = self.pareto_alphas[domain_id]
                d['pareto_scales'] = self.pareto_scales[domain_id]
                d['pareto_score_scales'] = self.pareto_score_scales[domain_id]
            domain_id_map.append(d)
        domain_id_map_str = 'MegaIndexedJsonlDatasetV2 id / domain mapping ' + json.dumps(
            domain_id_map, indent=4
        )
        print_rank_0(domain_id_map_str)

    def create_ds(self, domain_id, path_like, domain_epoch=0):
        if self.train:
            sample_rate = self.retention_rates_per_domains[domain_id]
            enable_pareto, pareto_alpha, pareto_scale, pareto_score_scale = False, None, None, None
            if self.flag_on_pareto_sampling:
                enable_pareto = self.enable_pareto[domain_id]
                pareto_alpha = self.pareto_alphas[domain_id]
                pareto_scale = self.pareto_scales[domain_id]
                pareto_score_scale = self.pareto_score_scales[domain_id]
            ## 区分开需要 pareto 抽样和不需要 pareto 抽样的数据
            # 不需要抽样的数据就不再重新做预处理了，兼容老的数据
            if enable_pareto:
                ds = load_dataset(
                    path_like,
                    split='train',
                    streaming=True,
                    trust_remote_code=True,
                    dp_rank=self.dp_rank,
                    dp_size=self.dp_size,
                    access_policy_interleave=self.access_policy_interleave,
                    sample_rate=sample_rate,
                    enable_pareto=enable_pareto,
                    pareto_alpha=pareto_alpha,
                    pareto_scale=pareto_scale,
                    pareto_score_scale=pareto_score_scale,
                )
            else:
                ds = load_dataset(
                    path_like,
                    split='train',
                    streaming=True,
                    trust_remote_code=True,
                    dp_rank=self.dp_rank,
                    dp_size=self.dp_size,
                    access_policy_interleave=self.access_policy_interleave,
                    sample_rate=sample_rate,
                )
        else:
            ds = load_dataset(
                path_like,
                split='train',
                streaming=True,
                trust_remote_code=True,
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                access_policy_interleave=self.access_policy_interleave,
                unsplit_data=self.unsplit_eval_data
            )
        ds = ds.map(partial(add_domain_id, domain_id))
        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed + domain_epoch)

        return ds

    def make_ds_list(self, ):
        ds_list = []
        for domain_id, path_like in enumerate(self.path_likes):
            domain_epoch = self.domain_consumed[domain_id].epoch
            ds = self.create_ds(domain_id, path_like, domain_epoch=domain_epoch)
            ds_list.append(iter(ds))

        return ds_list

    def read_and_parse_obj_from_jsonl(self, fname, offset, length):
        # no need for a file handler cache since it's handled by the os kernel
        if not self.train:
            file_cache = self.eval_file_cache
        else:
            file_cache = self.train_file_cache

        if fname in file_cache.keys():
            inf = file_cache[fname]
        else:
            inf = open(fname, 'rb')
            file_cache[fname] = inf
        inf.seek(offset)
        line = inf.read(length)
        obj = json.loads(line)

        return obj

    def log_skip(self, domain_id, domain_name, to_skip):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{time_str}] skip consumed lines dname {domain_name}' \
                + f' rank {torch.distributed.get_rank()}' \
                + f' dp_rank {self.dp_rank}' \
                + f' to_skip {to_skip}')

    def skip_consumed_data(self):
        for domain_id, domain_name in enumerate(self.domain_names):
            comsumed = self.domain_consumed[domain_id].line

            if comsumed > 0:
                self.log_skip(domain_id, domain_name, to_skip=comsumed)
                while comsumed > 0:
                    try:
                        next(self.ds_list[domain_id])
                        comsumed -= 1
                        if comsumed % 10000 == 0:
                            self.log_skip(domain_id, domain_name, to_skip=comsumed)
                    except StopIteration:
                        break
                print(f'done to_skip dname {domain_name} rank {torch.distributed.get_rank()}')
                self.domain_consumed[domain_id].line = 0

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True

        bufs = [
            SimpleNamespace(input_ids=[], n_packed=0)
            for domain_id, _ in enumerate(self.domain_names)
        ]

        while True:
            domain_id = self.global_batch_domain_id[self.curr_pointer_position]
            ds = self.ds_list[domain_id]

            # 先在 buffer 里面看看是否有足够的数据能弹出
            while len(bufs[domain_id].input_ids) < self.max_seq_len + 1:
                try:
                    idx = next(ds)
                except StopIteration:
                    self.domain_consumed[domain_id].epoch += 1
                    domain_epoch = self.domain_consumed[domain_id].epoch
                    # 该 domain 满了，重开一次
                    self.ds_list[domain_id] = iter(
                        self.create_ds(
                            domain_id, self.path_likes[domain_id], domain_epoch=domain_epoch
                        )
                    )
                    ds = self.ds_list[domain_id]
                    idx = next(ds)

                fname = idx['data_file_name']
                offset = idx['offset']
                length = idx['length']
                assert domain_id == idx['domain_id'].item()
                example = self.read_and_parse_obj_from_jsonl(fname, offset, length)
                # TODO: n_packed 添加到 buf 里
                bufs[domain_id].n_packed += 1
                if example.get('deleted', False):  # 被黑名单的垃圾数据
                    continue
                text = example['content']
                doc_id = example['docid']
                _input_ids, _ = tokenize_text(self.tokenizer, text)
                bufs[domain_id].input_ids += _input_ids

            input_ids = bufs[domain_id].input_ids[:self.max_seq_len + 1]
            labels = copy.deepcopy(input_ids)
            ret_d = {
                'input_ids':
                    torch.tensor(input_ids, dtype=torch.int64),
                'labels':
                    torch.tensor(labels, dtype=torch.int64),
                'train':
                    self.train,
                'domain_id':
                    torch.tensor(domain_id, dtype=torch.int64),
                'n_packed':
                    torch.tensor(bufs[domain_id].n_packed, dtype=torch.int64),
                'domain_epoch':
                    torch.tensor(self.domain_consumed[domain_id].epoch, dtype=torch.int64),
                'curr_pointer_position':
                    self.curr_pointer_position
            }
            bufs[domain_id].input_ids = bufs[domain_id].input_ids[self.max_seq_len:]
            bufs[domain_id].n_packed = 0
            self.curr_pointer_position = (self.curr_pointer_position +
                                          1) % len(self.global_batch_domain_id)
            # TODO: 如果 curr_pointer_position 达到 global_batch_domain_id 的尾巴
            # 是否要对 global_batch_domain_id 重新生成，或者 shuffle？
            # 其实主要是为了 shuffle global_batch_domain_id 的id?
            # if self.curr_pointer_position == 0:
            #     # shuffle global_batch_domain_id
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
    assert args.num_workers <= 1

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = MegaIndexedJsonlDatasetV2(
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
        access_policy_interleave=False,  # todo: 测试 nr 对这个flag 的去处是否有影响
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
    # eval data 不必再单独设计，复用之前的
    eval_ds = None
    if eval_path_likes is not None:
        eval_samples = args.eval_iters * args.global_batch_size // dp_size
        eval_ds = PretrainDataset(
            tokenizer,
            args.seq_length,
            eval_path_likes,
            None,
            args.px_eval_data_domain_names,
            train_data_consuming_progresses=None,
            train=False,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            access_policy_interleave=args.px_indexed_jsonl_dataset_access_policy_interleave,
            shuffle_buffer_size=0,
            eval_samples=eval_samples,
            seed=0
        )
    test_ds = None

    return train_ds, eval_ds, test_ds
