# coding=utf8
# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com
'''
预处理数据，对数据的唯一要求就是 jsonl（每一 line 一个 json）。
对比之前的预处理：
1. 处理速度大幅度加快。
2. 字段没有限制，比较灵活。
'''

from dataclasses import dataclass
import argparse
import glob
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import gzip

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from tqdm import tqdm
import numpy as np
import torch

EACH_INDEX_SIZE = 12
EACH_INDEX_WITH_SCORE_SIZE = 16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='preprocess-indexed-jsonl-dataset')
    group.add_argument('--data_folder', type=str, required=True, help='数据文件的文件夹')
    group.add_argument('--decompress', action='store_true', help='解压数据')
    group.add_argument(
        '--decompress_postfix', type=str, default='gz', help='压缩数据文件的 postfix，默认是 gz。'
    )
    group.add_argument(
        '--data_file_postfix', type=str, default='jsonl', help='数据文件的 postfix，默认是 jsonl。'
    )
    group.add_argument(
        '--idx_file_buf_sz',
        type=int,
        default=1024,
        help='写索引文件的 buffer size，攒够 buffer size 个会写文件。'
    )
    group.add_argument(
        '--ensure_each_line_forms_valid_json',
        action='store_true',
        help='每个 line 都 json.loads 一下，确保确实是 json，打开来 debug'
    )
    group.add_argument(
        '--domain_name',
        type=str,
        default='default-data-domain-name',
        help='数据 domain 的 name，让 debug 信息好看些。'
    )
    group.add_argument('--save-pareto-score', action='store_true', help='是否保存 score 值')
    group.add_argument('--rm-compress-file', action='store_true', help='是否删除压缩文件')
    args = parser.parse_args()
    return args


def decompress_each_file(args, fname, postfix):
    with gzip.open(fname, 'rb') as fin:
        with open(fname[:-len(postfix) - 1], 'wb') as fout:
            shutil.copyfileobj(fin, fout)
    if args.rm_compress_file and fname.endswith(args.decompress_postfix):
        os.remove(fname)


def append_to_idx_file(args, idxf, packed_idxs):
    real_each_index_size = EACH_INDEX_SIZE
    if args.save_pareto_score:
        real_each_index_size = EACH_INDEX_WITH_SCORE_SIZE

    assert len(packed_idxs) % real_each_index_size == 0
    idxf.write(packed_idxs)


def for_each_data_file(args, fname):
    idx_fname = fname + '.idx'
    with open(fname, 'rb') as dataf, open(idx_fname, 'wb') as idxf:
        pack_buf = []
        off = 0
        # 计算 pareto score 的 weights
        weights = [0.2, 0.4, 0.7, 1.0]
        max_score = 100.0
        for li, line in tqdm(enumerate(dataf)):
            score = 1.0
            real_each_index_size = EACH_INDEX_SIZE
            if args.save_pareto_score:
                assert args.ensure_each_line_forms_valid_json
                real_each_index_size = EACH_INDEX_WITH_SCORE_SIZE

            if args.ensure_each_line_forms_valid_json:
                item = json.loads(line)

                if args.save_pareto_score:
                    if "pretrain_quality_info" in item:
                        scores = json.loads(item["pretrain_quality_info"])["model_predict_scores"]
                        if scores is None or len(scores) != 4:
                            score = max_score
                        else:
                            score = np.sum(
                                [score * weight for score, weight in zip(scores, weights)]
                            ) - 0.2
                    else:
                        score = max_score

            # https://docs.python.org/3/library/struct.html
            if args.save_pareto_score:
                packed = struct.pack('<qif', off, len(line), score)
            else:
                packed = struct.pack('<qi', off, len(line))

            assert len(packed) == real_each_index_size
            pack_buf.append(packed)
            off += len(line)

            if len(pack_buf) % args.idx_file_buf_sz == 0:
                packed_idxs = b''.join(pack_buf)
                pack_buf.clear()
                append_to_idx_file(args, idxf, packed_idxs)

        if len(pack_buf) > 0:
            packed_idxs = b''.join(pack_buf)
            pack_buf.clear()
            append_to_idx_file(args, idxf, packed_idxs)


def make_data_file_list(args, data_fnames, idx_fnames):
    real_each_index_size = EACH_INDEX_SIZE
    if args.save_pareto_score:
        real_each_index_size = EACH_INDEX_WITH_SCORE_SIZE

    data_folder = args.data_folder.rstrip('/')
    rel_data_fnames = [os.path.basename(fname) for fname in data_fnames]
    rel_idx_fnames = [os.path.basename(fname) for fname in idx_fnames]
    szs = [os.path.getsize(idx_fname) for idx_fname in idx_fnames]
    nums = [fsz // real_each_index_size for fsz in szs]
    total_num = sum(nums)
    metadata = {
        'domain_name': args.domain_name,
        'data_files': rel_data_fnames,
        'idx_files': rel_idx_fnames,
        'nums': nums,
        'total_num': total_num,
    }
    out = os.path.join(data_folder, 'metadata.json')
    with open(out, 'w') as outf:
        outf.write(json.dumps(metadata, ensure_ascii=False))


def make_dataset_builder(args):
    data_folder = args.data_folder.rstrip('/')
    name = os.path.basename(data_folder)
    out_py = os.path.join(data_folder, f'{name}.py')

    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)
    shutil.copy(os.path.join(cur_dir, 'indexed_jsonl_dataset_builder_template.py'), out_py)


def main():
    args = get_args()
    try:
        comm = MPI.COMM_WORLD  # 假定使用 mpi，其实用 torchrun 也可以，但是 mpi 更加方便。
        rank = comm.Get_rank()
        size = comm.Get_size()
    except:
        rank = 0
        size = 1
    print(f"begin preprocess {rank} {size}", flush=True)
    print(f"inspect args  {args}", flush=True)

    decompress_fnames = [
        fname
        for fname in glob.glob(os.path.join(args.data_folder, '*.' + args.decompress_postfix))
    ]
    if args.decompress:
        for fname in decompress_fnames[rank:len(decompress_fnames):size]:
            decompress_each_file(args, fname, args.decompress_postfix)

    try:
        os.remove(os.path.join(args.data_folder, 'metadata.json'))
    except OSError:
        pass
    comm.Barrier()
    data_fnames = [
        fname for fname in glob.glob(os.path.join(args.data_folder, '*.' + args.data_file_postfix))
    ]
    data_fnames = sorted(data_fnames)
    idx_fnames = [fname + '.idx' for fname in data_fnames]

    # 造 index（索引）
    comm.Barrier()
    print('making indexes')
    for fname in data_fnames[rank:len(data_fnames):size]:
        for_each_data_file(args, fname)

    # metadata
    comm.Barrier()
    if rank == 0:
        print('making metadata')
        make_data_file_list(args, data_fnames, idx_fnames)

    # 造 dataset builder py
    comm.Barrier()
    if rank == 0:
        print('making dataset builder')
        make_dataset_builder(args)


if __name__ == '__main__':
    main()
