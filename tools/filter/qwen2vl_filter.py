import argparse
import json
import os
import time
from datetime import timedelta

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

from gpatch.core.utils import print_with_rank_and_datetime


def get_filter_args_filename(output_jsonl_dir: str):
    return os.path.join(output_jsonl_dir, "qwen2vl_filter_args.json")


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


def create_filter(args):
    from tasks.qwen2vl.qwen2vl_dataset_map import (
        Qwen2VlDatasetMap,
        Qwen2VLTokenizer,
        get_processor,
    )

    # Currently, only HuggingFace tokenizers are supported.
    tokenizer = _HuggingFaceTokenizer(args.tokenizer_model)
    tokenizer.__class__ = Qwen2VLTokenizer
    tokenizer.qwen2vl_init()

    processor = get_processor(args)
    filter = Qwen2VlDatasetMap(
        args.min_pixels_num,
        args.max_pixels_num,
        False,
        args.use_grpo,
        tokenizer,
        args.seq_length,
        processor=processor,
        image_token_id=None,
        mask_history=True,
    )

    return filter


def create_new_filter(args):
    from megatron_datasets.qwenvl_dataset_map import QwenVlDatasetMap, get_processor

    # Currently, only HuggingFace tokenizers are supported.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    hf_config = AutoConfig.from_pretrained(args.tokenizer_model)

    processor = get_processor(args)
    filter = QwenVlDatasetMap(
        hf_config,
        args.min_pixels_num,
        args.max_pixels_num,
        args.use_grpo,
        tokenizer,
        args.seq_length,
        processor=processor,
        mask_history=True,
    )

    return filter


def filter_sample(sample, filter_inst, print_filter_msg=False):
    sample_json = json.loads(sample)
    imgs = []
    if "images" in sample_json:
        for image in sample_json["images"]:
            if 'width' not in image or 'height' not in image:
                try:
                    imgs.append(Image.open(image['image_path']).convert("RGB"))
                except Exception as e:
                    print(f"error: {e}", flush=True)
                    return ""
            else:
                sizes = (image['width'], image['height'])
                imgs.append(Image.new('RGB', sizes, (255, 255, 255)))
    sample_json["__images_feat__"] = imgs
    res = filter_inst.process(sample_json)
    if not isinstance(res, dict):
        if print_filter_msg:
            print(f"filter: {res} {sample}", flush=True)
        return ""
    tokenizer_len = res['tokenizer_len'].item()
    json_data = json.loads(sample)
    if "images" in json_data:
        for image in json_data["images"]:
            if 'width' not in image or 'height' not in image:
                img = Image.open(image['image_path']).convert("RGB")
                width, height = img.size
                image['width'] = width
                image['height'] = height
    assert "tokenizer_len" not in json_data
    json_data["tokenizer_len"] = tokenizer_len
    return json.dumps(json_data, ensure_ascii=False) + "\n"


def get_filter_args():
    parser = argparse.ArgumentParser(
        description="qwen2vl/qwen2.5vl sampler filter",
        allow_abbrev=False,
        conflict_handler='resolve'
    )
    parser.add_argument(
        '--model-arch',
        type=str,
        default="qwen2.5vl",
        choices=[
            "qwen2vl", "qwen2.5vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe",
            "wemm3_embedding"
        ],
        help='model arch type'
    )
    parser.add_argument(
        '--tokenizer-model', type=str, required=True, help="The tokenizer model path"
    )
    parser.add_argument("--processor-path", type=str, default=None, help="")
    parser.add_argument("--min-pixels-num", type=int, default=None, help="min image width * height")
    parser.add_argument("--max-pixels-num", type=int, default=None, help="max image width * height")
    parser.add_argument("--video-min-frames", type=int, default=None, help="min video frames")
    parser.add_argument("--video-max-frames", type=int, default=None, help="max video frames")
    parser.add_argument(
        "--video-min-pixels", type=int, default=None, help="min video num_frame * width * height"
    )
    parser.add_argument(
        "--video-max-pixels", type=int, default=None, help="max video num_frame * width * height"
    )

    parser.add_argument('--use-grpo', action='store_true', help='use grpo')
    parser.add_argument('--use-new-filter', action='store_true', help='use grpo')
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
    if args.use_new_filter:
        filter = create_new_filter(args)
    else:
        filter = create_filter(args)
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

        for sample in tqdm(
            samples[begin_idx:end_idx],
            desc=f"filter sampler [rank: {rank:4d}]",
            # disable=(rank != 0),
        ):
            new_samle = filter_sample(sample, filter, False)
            if new_samle != "":
                filtered_samples.append(new_samle)

        torch.distributed.barrier()
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
