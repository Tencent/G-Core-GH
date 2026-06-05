# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com
from copy import deepcopy
from dataclasses import dataclass
from email.mime import image
from tkinter import N
from typing import Dict, Sequence, List, Tuple, Union
from types import SimpleNamespace
import dataclasses
from enum import IntEnum, auto

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate

from megatron_datasets.mm_dataset import (
    MultiModalDataset,
    fetch_images,
    convert_conversations,
    remove_bos,
)
from megatron_datasets.utils import print_rank_0, get_iterator

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

# from internvl github
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_img_end(token):
    global IMG_END_TOKEN
    IMG_END_TOKEN = token


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
        for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def transformer_imgs(imgs, input_size=448, max_num=12):
    res = []
    for img in imgs:
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        res.append(pixel_values)
    return res


# copy from: InternVL3-2B/conversation.py
class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = '\n'
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (role + ': ' + message.replace('\r\n', '\n').replace('\n\n', '\n'))
                    ret += '\n\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = '[INST] '
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + ' '
                    else:
                        ret += tag + ' ' + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if system_prompt == '' else system_prompt + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (template.name not in conv_templates), f'{template.name} has been registered.'

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# Both Hermes-2 and internlm2-chat are chatml-format conversation templates. The difference
# is that during training, the preprocessing function for the Hermes-2 template doesn't add
# <s> at the beginning of the tokenized sequence, while the internlm2-chat template does.
# Therefore, they are completely equivalent during inference.
register_conv_template(
    Conversation(
        name='Hermes-2',
        system_template='<|im_start|>system\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>',
        stop_str='<|endoftext|>',
    )
)

register_conv_template(
    Conversation(
        name='internlm2-chat',
        system_template='<|im_start|>system\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>',
    )
)

register_conv_template(
    Conversation(
        name='phi3-chat',
        system_template='<|system|>\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|user|>\n', '<|assistant|>\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|end|>',
    )
)

register_conv_template(
    Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)


class InternVLDataset(MultiModalDataset):
    def __init__(
        self,
        internvl_template,
        use_for_hf,
        mask_history,
        use_grpo,
        image_size,
        patch_size,
        max_num,
        downsample_ratio,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.internvl_template = internvl_template
        self.use_for_hf = use_for_hf
        self.mask_history = mask_history
        self.use_grpo = use_grpo
        self.num_image_token = int((image_size // patch_size)**2 * (downsample_ratio**2))
        self.image_token_index = self.tokenizer._tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.max_num = max_num
        self.image_size = image_size

    def convert_img2tensor(self, imgs):
        res = []
        for img in imgs:
            np_array = np.array(img)
            res.append(torch.from_numpy(np_array))
        return res

    def get_tokenizer_dict(self, conversations, imgs, add_generation_prompt):
        conversations = deepcopy(conversations)
        template = get_conv_template(self.internvl_template)
        if add_generation_prompt:
            conversations.append(dict(role="assistant", content=None))
        for conversation in conversations:
            if conversation['role'] == "user":
                template.append_message(template.roles[0], conversation['content'])
            elif conversation['role'] == "assistant":
                template.append_message(template.roles[1], conversation['content'])
            elif conversation['role'] == "system":
                # ignore system prompt
                continue
            else:
                assert False, f"not support this role: {conversation['role']}"

        text = template.get_prompt()

        num_patches_list = [pixel_values.shape[0] for pixel_values in imgs]
        for num_patches in num_patches_list:
            if '<image>' not in text:
                break
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            text = text.replace('<image>', image_tokens, 1)

        return self.tokenizer._tokenizer(text=text, return_tensors="pt")

    # tools and rm_bos not use
    def gen_label_mask(self, conversations, imgs, tools, label_role=["assistant"], rm_bos=True):
        pre_len = 0
        mask_indexs = []
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

            text_dict = self.get_tokenizer_dict(conversations[:i + 1], imgs, add_generation_prompt)

            cur_len = len(text_dict['input_ids'].squeeze(0).tolist())
            if conversations[i]['role'] not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len
        if self.mask_history:
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]
        return mask_indexs

    def convert_example(
        self,
        example,
        conversations,
        imgs,
        domain_states,
        tools=None,
        answer=None,
    ):
        imgs = transformer_imgs(imgs, input_size=self.image_size, max_num=self.max_num)
        add_generation_prompt = False
        if self.use_grpo and conversations[-1]['role'] == "assistant":
            conversations = conversations[:-1]
        if self.use_grpo:
            assert conversations[-1]['role'] != "assistant"
            add_generation_prompt = True

        all_text_dict = self.get_tokenizer_dict(conversations, imgs, add_generation_prompt)
        input_ids = all_text_dict['input_ids'].squeeze(0)
        attention_mask = all_text_dict['attention_mask'].squeeze(0).tolist()
        if self.image_token_id is not None:
            assert self.image_token_id == self.image_token_index
        total_image_token = (input_ids == self.image_token_index).sum().cpu().item()

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
        if len(input_ids) < self.max_seq_len + 1:
            input_ids += [self.tokenizer._tokenizer.pad_token_id
                         ] * (self.max_seq_len + 1 - len(input_ids))
            labels += [-100] * (self.max_seq_len + 1 - len(labels))
            attention_mask += [0] * (self.max_seq_len + 1 - len(attention_mask))

        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        if self.use_for_hf:
            labels = labels[:-1]
        else:
            labels = labels[1:]
        if len(input_ids) > self.max_seq_len:
            if self.use_grpo:
                print(f"GRPO Abort Sample at dp-rank:{self.underlying.dp_rank}[too long]")
                return None
            input_ids = input_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]
            attention_mask = attention_mask[-self.max_seq_len:]

        example["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        example["labels"] = torch.tensor(labels, dtype=torch.int64)
        example["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        if len(imgs) == 0:
            example["pixel_values"] = None
        else:
            example["pixel_values"] = torch.cat(imgs, dim=0).to(torch.bfloat16)
        image_input_mask = (example["input_ids"] == self.image_token_index)

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

        return example

    def __iter__(self):
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            # json_data 就是从jsonl读出的数据
            json_data = example["json_data"]
            imgs = None
            if 'images' in json_data and len(json_data['images']) > 0:
                imgs = fetch_images(json_data['images'], self.tar_dir, self.lmdb_port)
            conversations = json_data['conversations']
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
            yield example_copy


@dataclass
class DataCollatorForInternVL(object):
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
class DataCollatorForInternVLGRPO(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        pixel_values = []
        json_data_list = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
            del instance["pixel_values"]

            json_data_list.append(instance["json_data"])
            del instance["json_data"]

        res = default_collate(instances)
        if len(pixel_values) > 0:
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
        res["json_data_list"] = json_data_list
        return res


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
    processor = None
    internvl_template = args.internvl_template
    mask_history = args.mask_history
    use_grpo = args.use_grpo
    assert not is_dpo, f"not support dpo now"
    if use_grpo:
        assert mask_history, f"mask_history must be True when use grpo"

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = InternVLDataset(
        internvl_template,
        use_for_hf,
        mask_history,
        use_grpo,
        args.img_h,
        args.patch_dim,
        args.max_num,
        args.downsample_ratio,
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
        eval_ds = InternVLDataset(
            internvl_template,
            use_for_hf,
            mask_history,
            use_grpo,
            args.img_h,
            args.patch_dim,
            args.max_num,
            args.downsample_ratio,
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
    assert not is_dpo, f"not support dpo now"

    if args.use_grpo:
        collate_func = DataCollatorForInternVLGRPO()
        batch_size = args.ppo_rollout_micro_batch_size
    else:
        collate_func = DataCollatorForInternVL()
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
