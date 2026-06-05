import os
import argparse
from datetime import timedelta
import json
import time

import torch
from tqdm import tqdm
import transformers
from PIL import Image

from gpatch.core.utils import print_with_rank_and_datetime
from tasks.glm4v.glm4vl_dataset_map import Glm4vTokenizer, Glm4vDatasetMap
from megatron.training.tokenizer.multimodal_tokenizer import MultimodalTokenizer
from transformers import AutoProcessor, AutoTokenizer


def get_filter_args_filename(output_jsonl_dir: str):
    return os.path.join(output_jsonl_dir, "glmvl_filter_args.json")


def check_files(input_jsonl_files: str, output_jsonl_dir: str) -> list[str]:
    for i_file in input_jsonl_files:
        assert os.path.exists(i_file) and os.path.isfile(i_file)
    if torch.distributed.get_rank() == 0:
        fitler_args_file = get_filter_args_filename(output_jsonl_dir)
        if os.path.exists(output_jsonl_dir):
            assert os.path.isdir(output_jsonl_dir)
        else:
            os.makedirs(output_jsonl_dir, exist_ok=True)
        for i_file in input_jsonl_files:
            o_file = os.path.join(output_jsonl_dir, os.path.basename(i_file))
            assert o_file != fitler_args_file, f"your show rename the input file name"
            assert not os.path.exists(o_file), f"the outfile: {o_file} should not be exists"

        # the args filter file show not be exit
        assert not os.path.exists(
            o_file
        ), f"the fitler args file: {fitler_args_file} should not be exists"

    return input_jsonl_files


class SampleFilter:
    def __init__(self, dataset_map, seqlen):
        self.dataset_map = dataset_map
        self.seqlen = seqlen

    def __call__(self, sample, print_filter_msg=False):
        sample_json = json.loads(sample)
        imgs = []
        if "images" in sample_json:
            for image in sample_json["images"]:
                sizes = (image['width'], image['height'])
                imgs.append(Image.new('RGB', sizes, (255, 255, 255)))
        sample_json["__images_feat__"] = imgs
        # assert False, f"sample_json: {sample_json}"
        res = self.dataset_map.process(sample_json)
        valid = not isinstance(res, str)
        if not valid:
            return False, res
        tokenizer_len = res['tokenizer_len'].item()
        json_data = json.loads(sample)
        assert "tokenizer_len" not in json_data
        json_data["tokenizer_len"] = tokenizer_len
        return True, json.dumps(json_data, ensure_ascii=False) + "\n"


def build_filter(args):
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if isinstance(tokenizer, MultimodalTokenizer):
        tokenizer.__class__ = Glm4vTokenizer
        tokenizer.init_glm_4v()
    else:
        tokenizer = Glm4vTokenizer(tokenizer)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)

    dataset_map = Glm4vDatasetMap(
        False,
        args.use_grpo,
        tokenizer,
        args.seq_length,
        processor=processor,
        image_token_id=None,
        mask_history=True,
    )

    filter = SampleFilter(dataset_map, args.seq_length)

    return filter


def get_filter_args():
    parser = argparse.ArgumentParser(
        description="qwen2vl/qwen2.5vl sampler filter",
        allow_abbrev=False,
        conflict_handler='resolve'
    )
    parser.add_argument('--hf-model-path', type=str, required=True, help="The hf_model_path path")

    parser.add_argument('--use-grpo', action='store_true', help='use grpo')
    parser.add_argument(
        '--seq-length', type=int, default=None, help='Maximum sequence length to process.'
    )

    parser.add_argument("--input-jsonl-files", type=str, default=None, nargs='*', help="可直接用正规表达式")
    parser.add_argument("--output-jsonl-dir", type=str, default=None, help="需要文件夹内没有对应的文件")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    # 8h
    torch.distributed.init_process_group(backend='gloo', timeout=timedelta(seconds=8 * 60 * 60))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    args = get_filter_args()
    filter = build_filter(args)
    input_jsonl_files = check_files(args.input_jsonl_files, args.output_jsonl_dir)

    print_with_rank_and_datetime(f"begin to convert", 0)
    tic = time.time()
    for input_jsonl_file in tqdm(
        input_jsonl_files, desc=f"convert files [rank: {rank:4d}]", disable=(rank != 0)
    ):
        filtered_samples = []
        with open(input_jsonl_file) as f:
            samples = f.readlines()

        samples_len = len(samples)
        each_rank_size = (samples_len + world_size - 1) // world_size
        begin_idx = min(samples_len, rank * each_rank_size)
        end_idx = min(samples_len, (rank + 1) * each_rank_size)
        valid_cnt = 0
        total_cnt = 0
        for sample in tqdm(
            samples[begin_idx:end_idx],
            desc=f"filter sampler [rank: {rank:4d}]",
            # disable=(rank != 0),
        ):
            valid, new_samle = filter(sample)
            if valid:
                valid_cnt = valid_cnt + 1
                filtered_samples.append(new_samle)
            else:
                print(f"invalid sample {sample} : {new_samle}")
            total_cnt = total_cnt + 1

        torch.distributed.barrier()
        print(f"debug {valid_cnt} vs {total_cnt}")
        gather_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gather_list, filtered_samples)
        if rank == 0:
            all_samples = []
            for samples in gather_list:
                all_samples.extend(samples)

            o_file = os.path.join(args.output_jsonl_dir, os.path.basename(input_jsonl_file))
            with open(o_file, "w") as f:
                f.write("".join([sample for sample in all_samples]))

    torch.distributed.barrier()
    if rank == 0:
        with open(get_filter_args_filename(args.output_jsonl_dir), "w") as f:
            args_dict = vars(args)
            json.dump(args_dict, f, ensure_ascii=False, indent=4)

    torch.distributed.barrier()
    toc = time.time()
    cos_data_duration = toc - tic
    print_with_rank_and_datetime(f"Finish! covert time cost {cos_data_duration:2f}sec", 0)
