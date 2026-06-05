"""Gemma4 multimodal SFT dataset for llava-en-zh-300k.

Data format (parquet):
    - messages: list of {"role": str, "content": str}
    - images: list of {"bytes": bytes, "path": str}

Converts llava-style messages to Gemma4's multimodal content format,
then tokenises via the Gemma4 processor.
"""

import io
from typing import Any, Dict

import torch
from datasets import load_dataset
from PIL import Image, PngImagePlugin
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.utils import log
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


class Gemma4SftDataset(Dataset):
    def __init__(
        self,
        config: FinetuneConfig,
        tokenizer=None,
        processor=None,
        split="train",
    ):
        if PngImagePlugin.MAX_TEXT_CHUNK < 100 * 1024 * 1024:
            PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.seq_len = config.training.seq_length

        raw_ds = load_dataset("parquet", data_files=config.data.data_pathes, split=split)

        def is_valid(example):
            try:
                msgs = example.get("messages", [])
                if not msgs:
                    return False
                has_user = any(m["role"] == "user" for m in msgs)
                has_assistant = any(m["role"] == "assistant" for m in msgs)
                if not has_user or not has_assistant:
                    return False
                imgs = example.get("images", [])
                for img_data in imgs:
                    if img_data is None:
                        return False
                    if isinstance(img_data, dict) and not img_data.get("bytes"):
                        return False
                return True
            except Exception:
                return False

        len_src = len(raw_ds)
        self.dataset = raw_ds.filter(is_valid, num_proc=1)
        log(f"Gemma4SftDataset: filtered {len_src - len(self.dataset)} / {len_src} samples")

    def _decode_images(self, raw_images):
        images = []
        for img_data in raw_images:
            if isinstance(img_data, Image.Image):
                images.append(img_data.convert("RGB"))
            elif isinstance(img_data, dict) and img_data.get("bytes"):
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                images.append(img)
        return images

    def _convert_messages(self, messages, num_images):
        """Convert llava-style messages to Gemma4 multimodal content format.

        In llava data the user text contains ``<image>`` markers. This method
        replaces each marker with a ``{"type": "image"}`` content element so
        the Gemma4 processor can expand it into the correct number of soft
        tokens.
        """
        gemma4_msgs = []
        img_counter = 0
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if "<image>" in text and role == "user" and img_counter < num_images:
                parts = text.split("<image>")
                content = []
                for i, part in enumerate(parts):
                    if i > 0 and img_counter < num_images:
                        content.append({"type": "image"})
                        img_counter += 1
                    if part.strip():
                        content.append({"type": "text", "text": part.strip()})
                gemma4_msgs.append({"role": role, "content": content})
            else:
                gemma4_msgs.append({"role": role, "content": text})
        return gemma4_msgs

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        messages = example["messages"]
        raw_images = example.get("images", [])

        images = self._decode_images(raw_images)
        gemma4_msgs = self._convert_messages(messages, len(images))

        # --- Split into prompt (non-assistant) and full conversation ----------
        last_asst_idx = None
        for i in range(len(gemma4_msgs) - 1, -1, -1):
            if gemma4_msgs[i]["role"] == "assistant":
                last_asst_idx = i
                break
        assert last_asst_idx is not None

        prompt_msgs = gemma4_msgs[:last_asst_idx]

        # Tokenise the prompt (with images) to get the boundary
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=prompt_text,
            images=images if images else None,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        # Tokenise the full conversation
        full_text = self.processor.apply_chat_template(
            gemma4_msgs, tokenize=False, add_generation_prompt=False
        )
        full_inputs = self.processor(
            text=full_text,
            images=images if images else None,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        seq_length = len(input_ids)

        # Labels: mask everything before last assistant response
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Truncate if too long
        if seq_length > self.seq_len:
            input_ids = input_ids[-self.seq_len:]
            labels = labels[-self.seq_len:]
            seq_length = self.seq_len

        result = {
            "tokens": input_ids,
            "labels": labels,
            "sequence_lengths": torch.tensor(seq_length, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_len, dtype=torch.long),
        }

        pv = full_inputs.get("pixel_values", None)
        if pv is not None:
            result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv

        ip = full_inputs.get("image_position_ids", None)
        if ip is not None:
            result["image_position_ids"] = ip.squeeze(0) if ip.dim() > 3 else ip

        mm = full_inputs.get("mm_token_type_ids", None)
        if mm is not None:
            result["mm_token_type_ids"] = mm.squeeze(0)

        attn = full_inputs.get("attention_mask", None)
        if attn is not None:
            result["attention_mask"] = attn.squeeze(0)

        return result


def collate_fn(examples):
    """Collate into a dict of lists, compatible with ``expand_rollout_batches``."""
    batch = {}
    for key in examples[0].keys():
        batch[key] = [ex[key] for ex in examples]
    return batch


def get_dataset_and_dataloader(
    config: FinetuneConfig = None,
    tokenizer=None,
    dp_rank=0,
    dp_size=1,
    meta_info=None,
):
    processor = AutoProcessor.from_pretrained(
        config.policy.hf_tokenizer_path,
        trust_remote_code=True,
    )

    dataset = Gemma4SftDataset(config, tokenizer, processor)
    sampler = ResumableDistributedSampler(
        dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed,
        drop_last=True,
    )
    if meta_info is not None and "resume_step" in meta_info:
        resume_step = meta_info["resume_step"]
        gas = config.training.train_gbs // (dp_size * config.training.train_mbs)
        consumed_batches = resume_step * gas
        sampler.set_start_index(consumed_batches, config.training.train_mbs)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
    )
    return {
        "train_dataset": dataset,
        "train_sampler": sampler,
        "train_dataloader": dataloader,
    }
