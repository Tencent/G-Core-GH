"""
other module should import llamafactory related func from llamafactory_bridge, instead of importing llamafactory directlly
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Optional, Union

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.converter import (
    DatasetConverter,
    get_dataset_converter,
    register_dataset_converter,
)
from llamafactory.data.data_utils import Role
from llamafactory.data.parser import DatasetAttr
from llamafactory.extras import logging
from llamafactory.extras.constants import DATA_CONFIG, IGNORE_INDEX
from llamafactory.hparams import DataArguments
from transformers import AutoProcessor, AutoTokenizer

logger = logging.get_logger(__name__)
from types import MethodType


@dataclass
class LlamaFactoryConfig:
    model_args: Any = None
    training_args: Any = None
    data_args: Any = None
    template: Any = None
    tokenizer_module: Any = None


# do not resize in llamafactor logic
def _preprocess_image(self, image, **kwargs):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def init_llamafactory(args):
    """"
    init some args and a template to reuse llama factory code
    maping megatron args to model_args、model_args and traing_args
    """
    model_name_or_path = getattr(args, "processor_path", None) or args.model_path
    # mock model_args、data_args、training_args
    model_args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        use_fast_tokenizer=True,
        cache_dir=None,
        hf_hub_token=None,
        trust_remote_code=True
    )
    # arguments that should be add to args
    dataset_dir = getattr(args, "dataset_dir", None)
    dataset = getattr(args, "dataset", "demo")
    media_dir = getattr(args, "media_dir", None)
    template = getattr(args, "template", None) or getattr(args, "model_arch", None)
    max_samples = getattr(args, "max_samples", None)
    assert template is not None
    seq_len = args.seq_length
    num_workers = args.num_workers
    assert num_workers is not None
    cutoff_len = seq_len + 128

    # cutoff more and filter sample if len(input_ids) > seq_len

    data_args = DataArguments(
        dataset_dir=dataset_dir,
        dataset=dataset,
        media_dir=media_dir,
        template=template,
        ignore_pad_token_for_loss=True,
        overwrite_cache=True,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
    )

    training_args = SimpleNamespace(
        seed=0,
        predict_with_generate=False,
        should_log=True,
        do_train=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=1,
        dataloader_drop_last=True,
        train_dataloader_shuffle=True,
        data_seed=0,
        local_process_index=0,
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # image resize should be done by processor, rather than llamafactory
    template.mm_plugin._preprocess_image = MethodType(_preprocess_image, template.mm_plugin)

    llamafactory_config = LlamaFactoryConfig(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        template=template,
        tokenizer_module=tokenizer_module
    )
    return llamafactory_config


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = {"trust_remote_code": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    # patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        pass
        # patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


@dataclass
class GDatasetV4Adaptor(DatasetConverter):
    user_tag = "user"
    assistant_tag = "assistant"
    observation_tag = "observation"
    function_tag = "func"
    system_tag = "system"
    messages = "conversations"
    content_tag = "content"
    role_tag = "role"
    tools = "tools"
    images = "images"
    videos = "videos"
    audios = "audios"

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.user_tag: Role.USER.value,
            self.assistant_tag: Role.ASSISTANT.value,
            self.observation_tag: Role.OBSERVATION.value,
            self.function_tag: Role.FUNCTION.value,
            self.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.user_tag, self.observation_tag)
        even_tags = (self.assistant_tag, self.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.messages]
        if (
            self.system_tag and len(messages) != 0 and messages[0][self.role_tag] == self.system_tag
        ):
            system = messages[0][self.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(
                    f"Invalid role tag {message[self.role_tag]} { accept_tags[turn_idx % 2]} in {messages}."
                )
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.role_tag]],
                    "content": message[self.content_tag],
                }
            )

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        else:  # normal example
            if len(aligned_messages) == 1:
                prompt = aligned_messages
                response = []
            else:
                prompt = aligned_messages[:-1]
                response = aligned_messages[-1:]

        # get images from feat
        example[self.images] = example.pop("__images_feat__", [])

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.tools] if self.tools in example else "",
            "_images": example[self.images] if self.images in example else None,
            "_videos": example[self.videos] if self.videos in example else None,
            "_audios": example[self.audios] if self.audios in example else None,
        }
        return output
