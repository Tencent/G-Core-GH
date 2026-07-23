# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com, chanchzhang@tencent.com

import inspect
import math
import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed
from PIL import Image
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info
from torch.utils.data.dataloader import default_collate
from transformers import (
    AutoConfig,
    Qwen2VLImageProcessorFast,
    Qwen3VLVideoProcessor,
    WhisperFeatureExtractor,
)
from transformers.image_utils import is_valid_image
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.utils.import_utils import is_torchcodec_available
from transformers.video_utils import VideoMetadata
from typing_extensions import override

try:
    from transformers import Qwen3VLProcessor
except:
    Qwen3VLProcessor = None

from gdataset import GDatasetV4
from gdataset.data_loader.rope_index import get_index_helper
from gdataset.feat import PilImageListFeat
from megatron_datasets.mega_indexed_jsonl_dataset_mm import MegaIndexedJsonlDatasetMM
from megatron_datasets.mm_dataset import (
    MultiModalDatasetMap,
    convert_conversations,
    refact_conversations,
)
from megatron_datasets.tools.lmdb_read_cli import fetch_images_from_lmdb
from megatron_datasets.utils import get_iterator, random_pad_list

from mbridge.core.util import expand_thw, qwen2vl_pad_and_split
from mbridge.models.qwen3_vl.utils import reorganize_inputs

# copy from: https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
# 目前只保存读image的
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    ele: dict[str, str],
    image: Image.Image,
    default_min_pixels: int,
    default_max_pixels: int,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        default_min_pixels = default_min_pixels or MIN_PIXELS
        default_max_pixels = default_max_pixels or MAX_PIXELS
        width, height = image.size
        min_pixels = ele.get("min_pixels", default_min_pixels)
        max_pixels = ele.get("max_pixels", default_max_pixels)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def supports_mm_token_type_ids(mrope_index) -> bool:
    hf_class = getattr(mrope_index, "hf_class", None)
    if hf_class is None or not hasattr(hf_class, "get_rope_index"):
        return False

    try:
        rope_index_signature = inspect.signature(hf_class.get_rope_index)
    except (TypeError, ValueError):
        return False

    return "mm_token_type_ids" in rope_index_signature.parameters


def build_mm_token_type_ids(mrope_index, input_ids: torch.Tensor) -> torch.Tensor:
    mm_token_type_ids = torch.zeros_like(input_ids)

    image_token_id = getattr(mrope_index.config, "image_token_id", None)
    if image_token_id is not None:
        mm_token_type_ids[input_ids == image_token_id] = 1

    video_token_id = getattr(mrope_index.config, "video_token_id", None)
    if video_token_id is not None:
        mm_token_type_ids[input_ids == video_token_id] = 2

    return mm_token_type_ids


def get_mrope_kwargs(mrope_index, input_ids: torch.Tensor) -> dict:
    if not supports_mm_token_type_ids(mrope_index):
        return {}

    return {"mm_token_type_ids": build_mm_token_type_ids(mrope_index, input_ids)}


class UserQwen3OmniFeatureExtractor(WhisperFeatureExtractor):
    tmp_audios = None

    def __call__(
        self,
        raw_speech: np.ndarray | list[np.ndarray],
        **kwargs,
    ):
        if isinstance(raw_speech, list):
            audio = [self.tmp_audios[x] for x in raw_speech]
        elif isinstance(raw_speech, np.ndarray):
            audio = self.tmp_audios[raw_speech]

        return super().__call__(audio, **kwargs)


class UserQwen3VLVideoProcessor(Qwen3VLVideoProcessor):
    tmp_videos = None
    fps = 1.0

    def set_data(
        self,
        videos: list,
        fps: Optional[Union[int, float]] = None,
    ):
        self.tmp_videos = videos
        if fps != None:
            self.fps = fps

    def patched_load_video(
        self,
        idx: int,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        backend: str = "torchvision",
        sample_indices_fn: Optional[Callable] = None,
        **kwargs,
    ):
        if fps is not None and num_frames is not None and sample_indices_fn is None:
            raise ValueError(
                "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
            )

        # shape（T, C, H, W)
        video = self.tmp_videos[idx]
        metadata = VideoMetadata(
            total_num_frames=len(video),
            fps=fps,
            width=video.shape[3],
            height=video.shape[2],
            duration=len(video) / fps,
            video_backend=backend
        )
        frames_indices = sample_indices_fn(metadata=metadata, **kwargs)
        video = video[frames_indices].contiguous()

        metadata.update(
            {
                "frames_indices": frames_indices,
                "height": video.shape[2],
                "width": video.shape[3],
            }
        )

        return video, metadata

    @override
    def fetch_videos(
        self, video_url_or_urls: Union[str, list[str], list[list[str]]], sample_indices_fn=None
    ):
        backend = "torchcodec"
        if not is_torchcodec_available():
            warnings.warn(
                "`torchcodec` is not installed and cannot be used to decode the video by default. "
                "Falling back to `torchvision`. Note that `torchvision` decoding is deprecated and will be removed in future versions. "
            )
            backend = "torchvision"

        if isinstance(video_url_or_urls, list):
            video_num = 0
            if isinstance(video_url_or_urls[0], list):
                for urls in video_url_or_urls:
                    video_num += len(urls)
            else:
                video_num = len(video_url_or_urls)
            assert video_num == len(self.tmp_videos)
            return list(
                zip(
                    *[
                        self.patched_load_video(
                            i, fps=self.fps, backend=backend, sample_indices_fn=sample_indices_fn
                        ) for i in range(video_num)
                    ]
                )
            )

        else:
            video_num = 1
            assert video_num == len(self.tmp_videos)

            return self.patched_load_video(
                i, fps=self.fps, backend=backend, sample_indices_fn=sample_indices_fn
            )


class UserQwen2VLImageProcessorFast(Qwen2VLImageProcessorFast):
    tmp_images = None

    @override
    def fetch_images(self, image_url_or_urls: Union[str, list[str], list[list[str]]]):
        # Newer transformers call fetch_images twice: once for index/URL strings,
        # then again inside Fast image processor __call__ with already-loaded PIL.
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        if isinstance(image_url_or_urls, str):
            assert self.tmp_images is not None, "tmp_images is not set before fetch_images"
            idx = int(image_url_or_urls)
            return self.tmp_images[idx]
        if is_valid_image(image_url_or_urls):
            return image_url_or_urls
        raise TypeError(
            "only a single or a list of entries is supported but got "
            f"type={type(image_url_or_urls)}"
        )


class QwenVlDatasetMap(MultiModalDatasetMap):
    def __init__(
        self,
        hf_config,
        min_pixels,
        max_pixels,
        use_grpo,
        tokenizer,
        max_seq_len,
        processor=None,
        mask_history=False,
        meta_info_key="meta_info",
        moe_pad_with_random_token=False,
        grpo_resp_length=None,
        no_shift_label=False,
        config=None,
    ):
        super().__init__(
            use_for_hf=False,
            use_grpo=use_grpo,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            processor=processor,
            mask_history=mask_history,
            meta_info_key=meta_info_key,
            moe_pad_with_random_token=moe_pad_with_random_token,
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.video_processor = self.processor.video_processor
        self.feature_extractor = self.processor.feature_extractor if hasattr(
            self.processor, "feature_extractor"
        ) else None
        self.hf_config = hf_config.thinker_config if hasattr(
            hf_config, "thinker_config"
        ) else hf_config
        self.grpo_resp_length = grpo_resp_length
        self.no_shift_label = no_shift_label
        self.config = config

        self.pad_token_id = None
        if hasattr(self.tokenizer, "pad_token_id"):
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer._tokenizer.pad_token_id
        assert self.pad_token_id is not None

        if self.use_grpo:
            assert self.grpo_resp_length is not None

    @staticmethod
    def _align_to_final(partial_ids, final_ids):
        """Find the position in *final_ids* that corresponds to the end of
        *partial_ids*.

        Incremental ``apply_chat_template`` calls may inject extra tokens
        (e.g. ``<think>`` tags) that are absent in the full-conversation
        tokenization.  A two-pointer scan advances both pointers when tokens
        match; on mismatch only the *partial* pointer advances, effectively
        skipping any extra tokens.  Because the non-extra content is
        identical and in the same order in both sequences, the pointers
        always re-synchronise and *b* ends up at the correct position in
        *final_ids*.

        这里用贪心的方式匹配去掉 partial_token 里和 final_token 的 diff len
        这里要注意的是：@guanyouhe
        1. 如果 conversation 的对话中间的部分 assistant 有 <think> 标签包围了一些输入，在最后 apply_chat_teamplate
        之后，中间数据的 think 内容会被去掉；

        """
        a, b = 0, 0
        len_a, len_b = len(partial_ids), len(final_ids)
        while a < len_a and b < len_b:
            if partial_ids[a] == final_ids[b]:
                a += 1
                b += 1
            else:
                a += 1
        return b

    def get_image_token_cnt(self, image_grid_thw, video_grid_thw=None):
        merge_length = self.image_processor.merge_size**2
        total_cnt = torch.tensor(0, dtype=torch.long)
        if image_grid_thw is not None:
            for i in range(image_grid_thw.shape[0]):
                total_cnt += image_grid_thw[i].prod() // merge_length

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            for i in range(video_grid_thw.shape[0]):
                total_cnt += video_grid_thw.prod() // merge_length

        return total_cnt.item()

    def get_audio_token_cnt(self, input_features):
        if input_features is None:
            return 0
        assert input_features.ndim == 3
        last_dim = input_features.shape[-1]
        input_lengths_leave = last_dim % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (last_dim // 100) * 13
        return output_lengths

    def get_label_mask_and_media_info(
        self,
        conversations,
        tools,
        imgs=None,
        videos=None,
        audios=None,
        label_role=["assistant"],
        rm_bos=True,
    ):
        result = {}
        self.image_processor.tmp_images = imgs
        self.video_processor.set_data(videos)
        if self.feature_extractor is not None:
            self.feature_extractor.tmp_audios = audios

        partial_buffer = []
        final_input_ids = None

        for i in range(len(conversations)):
            if conversations[i]['role'] in ['system']:
                continue
            add_generation_prompt = False
            if i + 1 < len(conversations) and conversations[i]['role'] in [
                'user'
            ] and conversations[i + 1]['role'] in ["assistant"]:
                add_generation_prompt = True
            if self.use_grpo:
                assert conversations[-1]['role'] != "assistant"
                add_generation_prompt = True
            is_last = (i + 1 == len(conversations))
            chat_template_kwargs = {}
            if is_last:
                chat_template_kwargs["return_dict"] = True
                chat_template_kwargs["return_tensors"] = "pt"
            if self.config is not None and not getattr(self.config, 'ignore_thinking_flag', False):
                chat_template_kwargs["enable_thinking"] = self.config.training.enable_thinking

            inputs_dict = self.processor.apply_chat_template(
                conversations[:i + 1],
                tools=tools,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **chat_template_kwargs,
            )
            if is_last:
                input_ids = inputs_dict["input_ids"].tolist()[0]
            else:
                input_ids = inputs_dict[0]
            if rm_bos:
                input_ids = input_ids[1:]

            if is_last:
                final_input_ids = input_ids
                result.update(inputs_dict)

            partial_buffer.append((conversations[i]['role'], input_ids, is_last))

        mask_indexs = []
        pre_len = 0
        for role, partial_ids, was_last in partial_buffer:
            if was_last:
                cur_len = len(final_input_ids)
            else:
                cur_len = self._align_to_final(partial_ids, final_input_ids)
            if role not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len

        if self.mask_history:
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]

        result["mask_indexs"] = mask_indexs
        return result

    def truncate_by_max_seqlen(self, input_ids, labels, attention_mask):
        if len(input_ids) < self.max_seq_len + 1:
            if self.moe_pad_with_random_token:
                ban_token_ids = [
                    self.hf_config.image_token_id, self.hf_config.video_token_id,
                    self.hf_config.vision_start_token_id, self.hf_config.vision_end_token_id
                ]
                input_ids = random_pad_list(
                    input_ids, self.max_seq_len + 1 - len(input_ids), ban_token_ids
                )
            else:
                input_ids += [self.pad_token_id] * (self.max_seq_len + 1 - len(input_ids))
            labels += [-100] * (self.max_seq_len + 1 - len(labels))
            attention_mask += [0] * (self.max_seq_len + 1 - len(attention_mask))

        tmp_max_seq_len = self.max_seq_len
        if self.no_shift_label:
            tmp_max_seq_len = self.max_seq_len + 1
        else:
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
            labels = labels[1:]

        if len(input_ids) > tmp_max_seq_len:
            input_ids = input_ids[-tmp_max_seq_len:]
            labels = labels[-tmp_max_seq_len:]
            attention_mask = attention_mask[-tmp_max_seq_len:]
        return input_ids, labels, attention_mask

    def convert_example(
        self,
        conversations,
        imgs,
        videos,
        audios,
        tools=None,
        answer=None,
    ):
        inputs_dict = self.get_label_mask_and_media_info(
            conversations,
            tools,
            imgs,
            videos,
            audios,
            rm_bos=False,
        )
        input_ids = inputs_dict["input_ids"].tolist()[0]
        attention_mask = inputs_dict["attention_mask"].tolist()[0]
        labels = torch.tensor(input_ids, dtype=torch.int64)
        label_mask = inputs_dict["mask_indexs"]
        pixel_values = inputs_dict.pop("pixel_values", None)
        image_grid_thw = inputs_dict.pop("image_grid_thw", None)
        pixel_values_videos = inputs_dict.pop("pixel_values_videos", None)
        video_grid_thw = inputs_dict.pop("video_grid_thw", None)
        feature_attention_mask = inputs_dict.pop("feature_attention_mask", None)
        input_features = inputs_dict.pop("input_features", None)
        video_second_per_grid = inputs_dict.pop("video_second_per_grid", None)

        if self.use_grpo:
            assert len(label_mask) == 1 and label_mask[0][0] == 0
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100

        prompt_len = label_mask[-1][-1]
        tokenizer_len = len(input_ids)
        labels = labels.tolist()
        # grpo train need to complete sentences, so: prompt_len <= max_seq_len - resp_length
        if self.use_grpo and len(input_ids) > (self.max_seq_len - self.grpo_resp_length):
            return f"GRPO Invalid Sample: sample too long"

        input_ids, labels, attention_mask = self.truncate_by_max_seqlen(
            input_ids, labels, attention_mask
        )

        data_dict = {}
        data_dict["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        data_dict["labels"] = torch.tensor(labels, dtype=torch.int64)
        data_dict["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        data_dict["pixel_values"] = pixel_values
        data_dict["image_grid_thw"] = image_grid_thw
        data_dict["image_input_mask"] = data_dict["input_ids"] == self.hf_config.image_token_id
        data_dict["pixel_values_videos"] = pixel_values_videos
        data_dict["video_grid_thw"] = video_grid_thw
        data_dict["video_input_mask"] = data_dict["input_ids"] == self.hf_config.video_token_id
        data_dict["input_features"] = input_features.type(
            torch.bfloat16
        ) if input_features is not None else None
        data_dict["feature_attention_mask"] = feature_attention_mask
        data_dict["video_second_per_grid"] = video_second_per_grid.type(
            torch.int64
        ) if video_second_per_grid is not None else None

        sum_image_token = data_dict["image_input_mask"].sum().cpu().item()
        sum_image_token += data_dict["video_input_mask"].sum().cpu().item()
        total_image_token = self.get_image_token_cnt(image_grid_thw, video_grid_thw)
        if self.use_grpo:
            all_ignore = False
        else:
            if self.no_shift_label:
                all_ignore = torch.all(data_dict["labels"][1:] == -100).item()
            else:
                all_ignore = torch.all(data_dict["labels"] == -100).item()
        assert total_image_token >= sum_image_token
        # 跳过样本
        if total_image_token > sum_image_token or all_ignore:
            return f"Invalid Sample: image token-ids too long"

        if hasattr(self.hf_config, "audio_token_id"):
            total_audio_token = self.get_audio_token_cnt(input_features)
            sum_audio_token = (data_dict["input_ids"] == self.hf_config.audio_token_id
                              ).sum().cpu().item()
            # 跳过样本
            if total_audio_token > sum_audio_token:
                return f"Invalid Sample: audio token-ids too long"

        # prompt_len 是指原来没有截断过样本的 prompt_len，一般是给 grpo 使用的
        data_dict["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        # tokenizer_len 是指截断过样本的 tokenizer len
        data_dict["tokenizer_len"] = torch.tensor(
            min(tokenizer_len, self.max_seq_len), dtype=torch.int64
        )
        data_dict["sequence_lengths"] = torch.tensor(
            min(tokenizer_len, self.max_seq_len), dtype=torch.int64
        )
        return data_dict

    def process(self, example):
        imgs = example.pop("__images_feat__", [])  # read from feat
        videos = example.pop("__videos_feat__", None)  # read from feat
        audios = example.pop("__audios_feat__", None)  # read from feat
        audios_np_array = ([np.asarray(a, dtype=np.float32) for a in audios] if audios else None)

        imgs_valid = True
        for img in imgs:
            assert img is not None, f"the image is invalid"
            width, height = img.size
            if width < IMAGE_FACTOR or height < IMAGE_FACTOR:
                imgs_valid = False
                break
            if max(height, width) / min(height, width) > MAX_RATIO:
                imgs_valid = False
                break
        if not imgs_valid:
            return "image is to small"

        if len(imgs) == 0:
            imgs = None
        conversations = convert_conversations(example['conversations'])
        conversations = refact_conversations(conversations)
        tools = None
        if 'tools' in example:
            tools = example['tools']
        answer = None
        if 'label' in example:
            answer = example['label']
        assert len(conversations) >= 1

        # NOTE(guanyouhe): 这里 python/sglang/srt/multimodal/processors/qwen_vl.py 都做了 resize
        # qwen3vl processors 与 qwen2vl/qwen2.5vl 有所不同（应该是因为 qwen3vl 使用 Qwen2VLImageProcessorFast）
        # 所以这里得先做 resize
        if Qwen3VLProcessor is not None and isinstance(self.processor, Qwen3VLProcessor) \
          and imgs is not None:
            imgs = [
                resize_image(ele, img, self.min_pixels, self.max_pixels)
                for ele, img in zip(example['images'], imgs)
            ]

        data_dict = self.convert_example(conversations, imgs, videos, audios, tools, answer)
        if self.use_grpo and isinstance(data_dict, dict):
            data_dict["json_data"] = example
            imgs_np_array = None
            if imgs is not None:
                imgs_np_array = [
                    np.array(resize_image(ele, img, self.min_pixels, self.max_pixels))
                    for ele, img in zip(example['images'], imgs)
                ]
            data_dict["imgs_np_array"] = imgs_np_array
            data_dict["audios_np_array"] = audios_np_array
        if isinstance(data_dict, dict):
            if self.meta_info_key is not None:
                data_dict["meta_info"] = example.get(self.meta_info_key, None)
            else:
                data_dict["meta_info"] = None

        return data_dict

    def __call__(self, example):
        data_dict = self.process(example)
        assert isinstance(data_dict, dict), f"Should make sure the sample is valid: {data_dict}"
        return data_dict


class QwenVlDatasetV3(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        train_path_likes,
        domain_probabilities,
        domain_names,
        total_nums,
        global_batch_size,
        train_data_consuming_progresses,
        rank,
        dp_rank,
        dp_size,
        num_workers,
        shuffle_buffer_size,
        seed,
        use_grpo,
        grpo_resp_length,
        lmdb_port,
        hf_config,
        max_pixels,
        min_pixels,
        processor=None,
        mask_history=False,
        meta_info_key="meta_info",
        moe_pad_with_random_token=False,
    ):
        self.underlying = MegaIndexedJsonlDatasetMM(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            path_likes=train_path_likes,
            domain_probabilities=domain_probabilities,
            domain_names=domain_names,
            global_batch_size=global_batch_size,
            train_data_consuming_progresses=train_data_consuming_progresses,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=num_workers,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            # # NOTE(guanyouhe): 很多 datasetv3 参数因为 gpatch_v4 还没有，先不引入
            train=True,
            access_policy_interleave=False,
            retention_rates_per_domains=None,
            unsplit_eval_data=False,
            enable_pareto=[],
            pareto_alphas=[],
            pareto_scales=[],
            pareto_score_scales=[],
            top_domains_to_cut=1,
        )
        self.map_ds = QwenVlDatasetMap(
            hf_config=hf_config,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            use_grpo=use_grpo,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            processor=processor,
            mask_history=mask_history,
            meta_info_key=meta_info_key,
            moe_pad_with_random_token=moe_pad_with_random_token,
            grpo_resp_length=grpo_resp_length,
        )
        self.lmdb_port = lmdb_port
        self.domain_probabilities = domain_probabilities
        self.total_nums = total_nums
        self.guessed_total_num = 0
        assert len(self.total_nums) == len(self.domain_probabilities)
        for total_num, probability in zip(self.total_nums, self.domain_probabilities):
            self.guessed_total_num = max(int(total_num / probability), self.guessed_total_num)

    def __len__(self):
        return self.guessed_total_num

    def fetch_images(self, src_json_data: dict) -> list[Image.Image]:
        assert self.lmdb_port is not None, f"only support now: {self.lmdb_port=}"
        images = src_json_data['images']
        # 后面这里可以直接导入一个函数
        img_lists = fetch_images_from_lmdb(images, self.lmdb_port)
        return img_lists

    def __iter__(self):
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            domain_states.domain_lines += example["domain_line"]
            src_json_data = example["json_data"]
            del example["json_data"]

            src_json_data["__images_feat__"] = self.fetch_images(src_json_data)
            data_dict = self.map_ds.process(src_json_data)
            # 非法样本
            if isinstance(data_dict, str):
                continue
            assert "domain_line" not in data_dict
            data_dict["domain_line"] = torch.tensor(domain_states.domain_lines, dtype=torch.int64)
            example.update(data_dict)

            domain_states.domain_lines = 0
            yield example


class DataCollatorForQwenVl(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self,
        hw_factor: int = 1,
        model_arch="qwen2vl",
        tokenizer=None,
        is_dpo=False,
        use_grpo=False,
        cp_size=1,
        hf_config_path=None,
    ):
        super().__init__()
        # qwen2vl所有的模型merge_size都为2，因此它本来就是2*2的倍数
        self.hw_factor = hw_factor * 4
        self.model_arch = model_arch
        self.tokenizer = tokenizer
        self.is_dpo = is_dpo
        self.use_grpo = use_grpo
        self.cp_size = cp_size
        self.hf_config_path = hf_config_path
        self.mrope_index = get_index_helper(model_arch, self.hf_config_path)

        self.pad_token_id = None
        if hasattr(self.tokenizer, "pad_token_id"):
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer._tokenizer.pad_token_id
        assert self.pad_token_id is not None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.is_dpo:
            assert len(instances) % 2 == 0
            # 正负样本交错出现，要换好顺序
            instances = instances[::2] + instances[1::2]

        new_instances = []
        pixel_values = []
        image_grid_thws = []
        pixel_values_videos = []
        video_grid_thws = []
        json_data_list = []
        meta_info_list = []
        imgs_np_array_list = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
                image_grid_thws.append(instance["image_grid_thw"])
            if instance["pixel_values_videos"] is not None:
                pixel_values_videos.append(instance["pixel_values_videos"])
                video_grid_thws.append(instance["video_grid_thw"])
            del instance["pixel_values"]
            del instance["image_grid_thw"]
            del instance["pixel_values_videos"]
            del instance["video_grid_thw"]
            if self.use_grpo:
                json_data_list.append(instance["json_data"])
                del instance["json_data"]
                imgs_np_array_list.append(instance["imgs_np_array"])
                del instance["imgs_np_array"]
            meta_info_list.append(instance["meta_info"])
            del instance["meta_info"]
            for omni_key in ["input_features", "feature_attention_mask", "video_second_per_grid"]:
                if omni_key in instance:
                    del instance[omni_key]

            new_instances.append(instance)

        res = default_collate(new_instances)

        def cat_tensor(list_t: list[torch.Tensor], dim=0):
            if len(list_t) > 0:
                return torch.cat(list_t, dim=dim)
            return None

        pixel_values = cat_tensor(pixel_values, dim=0)
        image_grid_thws = cat_tensor(image_grid_thws, dim=0)
        pixel_values_videos = cat_tensor(pixel_values_videos, dim=0)
        video_grid_thws = cat_tensor(video_grid_thws, dim=0)
        mrope_kwargs = get_mrope_kwargs(self.mrope_index, res["input_ids"])
        position_ids, _ = self.mrope_index.get_rope_index(
            res["input_ids"],
            image_grid_thw=image_grid_thws,
            video_grid_thw=video_grid_thws,
            attention_mask=None,  # 这里写 None 就好，因为 grpo 要用到后面的编码
            **mrope_kwargs,
        )
        # Loss mask.
        loss_mask = torch.ones(res["labels"].size(), dtype=torch.float)
        loss_mask[res["labels"] == self.pad_token_id] = 0.0  # mask paddings
        loss_mask[res["labels"] == -100] = 0.0  # mask prompts
        # can reorganize_inputs at dataset
        vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
            input_ids=res["input_ids"],
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thws,
            video_grid_thw=video_grid_thws,
            image_input_mask=res["image_input_mask"],
            video_input_mask=res["video_input_mask"],
            image_token_id=self.mrope_index.config.image_token_id,
            video_token_id=self.mrope_index.config.video_token_id,
            square_merge_size=self.mrope_index.config.vision_config.spatial_merge_size**2,
        )
        del res["image_input_mask"]
        del res["video_input_mask"]

        if vision_data is not None:
            vision_grid_thw = expand_thw(vision_grid_thw)
            # pad empty and split image for tp/sp/cp
            cp_size = 1 if self.use_grpo else self.cp_size
            pixel_values, image_grid_thws, cp_img_num, images_padded = qwen2vl_pad_and_split(
                cp_size,
                self.hw_factor,
                [vision_data],
                [vision_grid_thw],
            )
            if self.model_arch in [
                "qwen3_vl_moe", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3_omni_moe",
                "wemm3_embedding"
            ]:
                for image_padded in images_padded:
                    assert not image_padded, "not support image padded now"

            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["image_grid_thw"] = torch.cat(image_grid_thws, dim=0)
            res["image_input_mask"] = vision_mask
            res["has_image"] = torch.tensor([True], dtype=torch.bool)
            res["images_padded"] = torch.tensor(images_padded, dtype=torch.int64)
            res["cp_img_num"] = torch.tensor(cp_img_num, dtype=torch.int64)
        else:
            res["pixel_values"] = None
            res["image_grid_thw"] = None
            res["image_input_mask"] = torch.zeros_like(res['labels'], dtype=torch.bool)
            res["has_image"] = torch.tensor([False], dtype=torch.bool)
            res["images_padded"] = torch.tensor([False], dtype=torch.int64)

        res["loss_mask"] = loss_mask
        # don't del clone
        res["position_ids"] = position_ids.clone()
        if self.use_grpo:
            res["json_data_list"] = json_data_list
            res["imgs_np_array_list"] = imgs_np_array_list
        res["meta_info"] = meta_info_list
        return res


class DataCollatorForQwenVlGRPO(DataCollatorForQwenVl):
    def __init__(
        self,
        hw_factor: int = 1,
        model_arch="qwen2vl",
        tokenizer=None,
        is_dpo=False,
        use_grpo=False,
        cp_size=1,
        hf_config_path=None,
        dp_rank=0,
    ):
        super().__init__(
            hw_factor=hw_factor,
            model_arch=model_arch,
            tokenizer=tokenizer,
            is_dpo=is_dpo,
            use_grpo=use_grpo,
            cp_size=cp_size,
            hf_config_path=hf_config_path,
        )
        self.dp_rank = dp_rank
        self.worker_id = None
        self.uniq_id = 0

    def gen_unique_id(self):
        if self.worker_id is None:
            self.worker_id = get_worker_info().id
        self.uniq_id += 1
        return f"dp_rank_{self.dp_rank}_worke_id_{self.worker_id}_{self.uniq_id}"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) == 1, "--ppo-rollout-micro-batch-size must be 1"
        data = super().__call__(instances)
        # TODO(guanyouhe): 这里暂时不回传与进度相关的东西
        image_padded = data["images_padded"].bool()[0].item()
        assert not image_padded, f"image padded 必须为 False，因为会被重新组合"

        pixel_values = None
        vision_grid_thw = None
        image_input_mask = data["image_input_mask"]
        if "pixel_values" in data:
            pixel_values = data["pixel_values"].type(torch.bfloat16)
            vision_grid_thw = data["image_grid_thw"]

        batch_data = dict(
            # type is list
            unique_id=[self.gen_unique_id()],
            json_data_list=data['json_data_list'],
            tokens=[data["input_ids"].squeeze(0)],
            prompt_len=[data["prompt_len"]],
            imgs_np_array_list=data["imgs_np_array_list"],
            # save at mm_data_cache, type is tensor
            position_ids=data["position_ids"],
            vision_grid_thw=vision_grid_thw,
            image_input_mask=image_input_mask,
            vision_data=pixel_values,
            cache_keys=[
                "position_ids",
                "vision_grid_thw",
                "image_input_mask",
                "vision_data",
            ],
        )
        return batch_data


class TrainerV4DataCollatorForQwenVl(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self,
        hw_factor: int = 1,
        model_arch="qwen2vl",
        tokenizer=None,
        is_dpo=False,
        use_grpo=False,
        hf_config_path=None,
        only_return_last_hidden_state=False,
    ):
        super().__init__()
        # qwen2vl所有的模型merge_size都为2，因此它本来就是2*2的倍数
        self.hw_factor = hw_factor * 4
        self.model_arch = model_arch
        self.tokenizer = tokenizer
        self.is_dpo = is_dpo
        self.use_grpo = use_grpo
        self.hf_config_path = hf_config_path
        self.only_return_last_hidden_state = only_return_last_hidden_state
        self.mrope_index = get_index_helper(model_arch, self.hf_config_path)

        self.pad_token_id = None
        if hasattr(self.tokenizer, "pad_token_id"):
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer._tokenizer.pad_token_id
        assert self.pad_token_id is not None

    def __call__(
        self, instances: Sequence[Dict] | Sequence[Tuple[Dict, Dict]]
    ) -> Dict[str, torch.Tensor]:
        if self.is_dpo:
            # dpo 的输入是 Sequence[Tuple[Dict, Dict]]
            assert isinstance(instances[0], tuple)
            new_instances = []
            for instance in instances:
                assert len(instance) == 2
                new_instances.append(instance[0])
                new_instances.append(instance[1])

            instances = new_instances
            assert len(instances) % 2 == 0
            # 正负样本交错出现，要换好顺序
            instances = instances[::2] + instances[1::2]

        batch_size = len(instances)
        extra_keys = [
            'input_ids', 'labels', 'attention_mask', 'prompt_len', 'tokenizer_len',
            'sequence_lengths', 'meta_info'
        ]
        if self.use_grpo:
            extra_keys.extend(["json_data", "imgs_np_array"])
        if self.model_arch == "qwen3_omni_moe":
            extra_keys.extend(["input_features", "feature_attention_mask", "video_second_per_grid"])
            if self.use_grpo:
                extra_keys.append("audios_np_array")
        k_map = {
            "input_ids": "tokens",
            "prompt_len": "prompt_lengths",
        }

        longest_len = torch.stack([instance["sequence_lengths"]
                                   for instance in instances]).view(-1).max().item()
        ret_res = defaultdict(list)
        for instance in instances:
            attention_mask = None
            kwargs = {}
            if self.model_arch == "qwen3_omni_moe":
                if instance["feature_attention_mask"] is not None:
                    kwargs["audio_seqlens"] = torch.sum(instance["feature_attention_mask"], dim=1)
                else:
                    kwargs["audio_seqlens"] = None
                kwargs["second_per_grids"] = instance["video_second_per_grid"]
            input_ids = instance["input_ids"].unsqueeze(0)
            if self.model_arch == "qwen3_omni_moe":
                # GRPO 训练时 input_ids 长度为 seq_length（含 padding），
                # 真实 attention_mask 只覆盖 prompt，会导致 prompt 之外的
                # position_ids 全部为 0，影响后续 rollout 推理时的外推。
                # 这里传 全 1 的 attention_mask 让所有 token 都能拿到正确的
                # 顺序 position_ids，与 qwen3_vl 的处理保持一致
                # (qwen3_vl 的 get_rope_index 在 attention_mask=None 时会
                # 内部用 ones_like 兜底，qwen3_omni 没有兜底所以这里显式传)。
                attention_mask = torch.ones_like(input_ids)
            mrope_kwargs = get_mrope_kwargs(self.mrope_index, input_ids)
            position_ids, _ = self.mrope_index.get_rope_index(
                input_ids,
                image_grid_thw=instance["image_grid_thw"],
                video_grid_thw=instance["video_grid_thw"],
                attention_mask=attention_mask,
                **mrope_kwargs,
                **kwargs,
            )
            # can reorganize_inputs at dataset
            vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
                input_ids=instance["input_ids"].unsqueeze(0),
                pixel_values=instance["pixel_values"],
                pixel_values_videos=instance["pixel_values_videos"],
                image_grid_thw=instance["image_grid_thw"],
                video_grid_thw=instance["video_grid_thw"],
                image_input_mask=instance["image_input_mask"].unsqueeze(0),
                video_input_mask=instance["video_input_mask"].unsqueeze(0),
                image_token_id=self.mrope_index.config.image_token_id,
                video_token_id=self.mrope_index.config.video_token_id,
                square_merge_size=self.mrope_index.config.vision_config.spatial_merge_size**2,
            )

            if vision_data is not None:
                vision_grid_thw = expand_thw(vision_grid_thw)
                ret_res["vision_data"].append(vision_data)
                ret_res["vision_grid_thw"].append(vision_grid_thw)

            ret_res["image_input_mask"].append(vision_mask)
            ret_res["position_ids"].append(position_ids.clone())

            label = instance['labels'].unsqueeze(0)
            loss_mask = torch.ones(label.size(), dtype=torch.float)
            loss_mask[label == self.pad_token_id] = 0.0  # mask paddings
            loss_mask[label == -100] = 0.0  # mask prompts
            ret_res["loss_mask"].append(loss_mask)
            ret_res["only_return_last_hidden_state"].append(self.only_return_last_hidden_state)

            for k in extra_keys:
                new_k = k
                if new_k in k_map:
                    new_k = k_map[k]
                # NOTE: 这里需要截断，因为框架都是按照最长的长度来处理的, 截断方便后面的 longest train
                if k in ["input_ids", "labels"]:
                    ret_res[new_k].append(instance[k][..., :longest_len])
                else:
                    ret_res[new_k].append(instance[k])

        ret_res = dict(ret_res)
        for k in ret_res.keys():
            assert len(ret_res[k]) == batch_size, f"error: {k=} {batch_size=} {len(ret_res[k])=}"

        return ret_res


class TrainerV4DataCollatorForQwenVlGRPO(TrainerV4DataCollatorForQwenVl):
    def __init__(
        self,
        hw_factor: int = 1,
        model_arch="qwen2vl",
        tokenizer=None,
        hf_config_path=None,
        dp_rank=0,
    ):
        super().__init__(
            hw_factor=hw_factor,
            model_arch=model_arch,
            tokenizer=tokenizer,
            is_dpo=False,
            use_grpo=True,
            hf_config_path=hf_config_path,
        )
        self.dp_rank = dp_rank
        self.worker_id = None
        self.uniq_id = 0

    def gen_unique_id(self):
        if self.worker_id is None:
            self.worker_id = get_worker_info().id
        self.uniq_id += 1
        return f"dp_rank_{self.dp_rank}_worke_id_{self.worker_id}_{self.uniq_id}"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) == 1, "--ppo-rollout-micro-batch-size must be 1"
        data = super().__call__(instances)
        assert len(data["position_ids"]) == 1
        assert len(data["image_input_mask"]) == 1

        vision_data = None
        vision_grid_thw = None
        if "vision_data" in data:
            assert len(data["vision_data"]) == 1
            assert len(data["vision_grid_thw"]) == 1
            vision_data = data["vision_data"][0].type(torch.bfloat16)
            vision_grid_thw = data["vision_grid_thw"][0]

        input_features = None
        feature_attention_mask = None
        if "input_features" in data:
            assert len(data["input_features"]) == 1
            assert len(data["feature_attention_mask"]) == 1
            input_features = data["input_features"][0]
            feature_attention_mask = data["feature_attention_mask"][0]

        cache_keys = [
            "position_ids",
            "vision_grid_thw",
            "image_input_mask",
            "vision_data",
        ]
        if input_features is not None:
            cache_keys.extend(["input_features", "feature_attention_mask"])

        batch_data = dict(
            # type is list
            unique_id=[self.gen_unique_id()],
            json_data_list=data['json_data'],
            tokens=data["tokens"],
            prompt_len=data["prompt_lengths"],
            imgs_np_array_list=data["imgs_np_array"],
            # save at mm_data_cache, type is tensor
            position_ids=data["position_ids"][0],
            vision_grid_thw=vision_grid_thw,
            image_input_mask=data["image_input_mask"][0],
            vision_data=vision_data,
            cache_keys=cache_keys,
        )
        if "audios_np_array" in data:
            batch_data["audios_np_array_list"] = data["audios_np_array"]
        if input_features is not None:
            batch_data["input_features"] = input_features
            batch_data["feature_attention_mask"] = feature_attention_mask
        return batch_data


def get_processor(args, model_arch=None):
    processor_path = args.processor_path
    image_process = UserQwen2VLImageProcessorFast.from_pretrained(processor_path)
    video_process = UserQwen3VLVideoProcessor.from_pretrained(processor_path)
    processor = AutoProcessor.from_pretrained(
        processor_path, image_processor=image_process, video_processor=video_process
    )
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if hasattr(args, "model_arch"):
        model_arch = args.model_arch
    if model_arch in [
        "qwen3_vl_moe", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3_omni_moe", "wemm3_embedding"
    ]:
        # set the user param
        if args.min_pixels_num is not None:
            processor.image_processor.min_pixels = args.min_pixels_num
            processor.image_processor.size["shortest_edge"] = args.min_pixels_num
        if args.max_pixels_num is not None:
            processor.image_processor.max_pixels = args.max_pixels_num
            processor.image_processor.size["longest_edge"] = args.max_pixels_num

        if args.video_min_frames is not None:
            processor.video_processor.min_frames = args.video_min_frames
        if args.video_max_frames is not None:
            processor.video_processor.max_frames = args.video_max_frames

        if args.video_min_pixels is not None:
            processor.video_processor.size["shortest_edge"] = args.video_min_pixels
        if args.video_max_pixels is not None:
            processor.video_processor.size["longest_edge"] = args.video_max_pixels
    return processor


def sort_by_prompt_len(sample):
    return sample["tokenizer_len"]


def build_train_valid_test_datasets(
    args,
    hf_config,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    is_dpo=False,
    feats=None,
):
    assert not is_dpo, "not support now"
    train_path_like = args.gdatasetv4_train_metadata_file
    eval_path_like = args.gdatasetv4_eval_metadata_file
    processor = get_processor(args)
    mask_history = args.mask_history
    use_grpo = args.use_grpo
    if use_grpo:
        assert mask_history, f"mask_history must be True when use grpo"

    gbs = args.global_batch_size
    consumed = args.iteration * gbs
    if args.use_grpo:
        gbs = args.ppo_rollout_global_batch_size
        assert args.iteration % args.train_iters_each_rollout == 0
        ppo_step = args.iteration // args.train_iters_each_rollout
        consumed = ppo_step * gbs

    smart_padding_compare_func = None
    smart_padding_buffer_size = 0
    if args.px_inputs_pad_to_longest:
        smart_padding_compare_func = sort_by_prompt_len
        smart_padding_buffer_size = args.px_smart_padding_buffer_size

    train_ds = GDatasetV4(
        train_path_like,
        dp_rank=dp_rank,
        dp_size=dp_size,
        gbs=gbs,
        shuffling_buffer_size=args.px_shuffle_buffer_size,
        consumed=consumed,
        feats=feats,
        seed=42,
        smart_padding_compare_func=smart_padding_compare_func,
        smart_padding_buffer_size=smart_padding_buffer_size,
        mbs=args.micro_batch_size,
    )
    train_map_fn = QwenVlDatasetMap(
        hf_config,
        args.min_pixels_num,
        args.max_pixels_num,
        use_grpo,
        tokenizer,
        args.seq_length,
        processor=processor,
        mask_history=mask_history,
        moe_pad_with_random_token=args.moe_pad_with_random_token,
    )
    train_ds.map(train_map_fn)
    train_ds.set_epoch(0)

    eval_ds = None
    if eval_path_like is not None:
        eval_gbs = gbs
        if args.use_grpo:
            eval_gbs = args.ppo_eval_rollout_global_batch_size
        eval_ds = GDatasetV4(
            eval_path_like,
            dp_rank=dp_rank,
            dp_size=dp_size,
            gbs=eval_gbs,
            shuffling_buffer_size=args.px_shuffle_buffer_size,
            consumed=0,
            feats=feats,
            seed=42,
        )
        eval_map_fn = QwenVlDatasetMap(
            hf_config,
            args.min_pixels_num,
            args.max_pixels_num,
            use_grpo,
            tokenizer,
            args.seq_length,
            processor=processor,
            mask_history=mask_history,
            moe_pad_with_random_token=args.moe_pad_with_random_token,
        )
        eval_ds.map(eval_map_fn)
        eval_ds.set_epoch(0)
    test_ds = None

    return train_ds, eval_ds, test_ds


def build_train_valid_test_data_iter(
    args, tokenizer, rank=0, dp_rank=0, dp_size=1, use_for_hf=False, is_dpo=False, feats=None
):
    if feats is None:
        feats = {
            'images':
                PilImageListFeat(
                    lmdb=True,
                    return_src_data=True,
                    convert_to_rgb=True,
                    new_name="__images_feat__",
                ),
        }
    hf_config = AutoConfig.from_pretrained(args.processor_path)
    train_ds, eval_ds, test_ds = build_train_valid_test_datasets(
        args,
        hf_config,
        tokenizer,
        rank,
        dp_rank,
        dp_size,
        is_dpo=is_dpo,
        feats=feats,
    )

    hw_factor = args.context_parallel_size
    if args.sequence_parallel:
        hw_factor *= args.tensor_model_parallel_size
    # grpo数据先不pad
    if args.use_grpo or args.model_arch in [
        "qwen3_vl_moe", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3_omni_moe", "wemm3_embedding"
    ]:
        hw_factor = 1

    collate_func = DataCollatorForQwenVl(
        hw_factor=hw_factor,
        model_arch=args.model_arch,
        tokenizer=tokenizer,
        is_dpo=is_dpo,
        use_grpo=args.use_grpo,
        cp_size=args.context_parallel_size,
        hf_config_path=args.processor_path,
    )

    batch_size = args.micro_batch_size
    if args.use_grpo:
        batch_size = args.ppo_rollout_micro_batch_size
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
        eval_batch_size = batch_size
        if args.use_grpo:
            eval_batch_size = args.ppo_eval_rollout_micro_batch_size
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
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
    return get_iterator(train_dataloader), get_iterator(eval_dataloader
                                                       ), get_iterator(test_dataloader)
