from functools import lru_cache
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer

#from roll.datasets.collator import DataCollatorWithPaddingForMM


@lru_cache(maxsize=10)
def compute_conversation_end_token_id(tokenizer: PreTrainedTokenizer) -> List[int]:
    """Find ``<|im_end|>`` token id."""
    assistant_mock = [{"role": "user", "content": ""}]
    assistant_token_ids_mock: List[int] = tokenizer.apply_chat_template(
        assistant_mock, tokenize=True, return_dict=False
    )
    for token_id in reversed(assistant_token_ids_mock):
        if token_id in tokenizer.all_special_ids:
            return [token_id]
    return []


def custom_apply_chat_template(
    messages: List[Dict],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt=True,
    tools=None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
) -> List:
    if len(messages) == 0:
        return []
    if tools is not None:
        assert messages[0]["role"] == "system", "tools is only supported for system role message"
    extra_kwargs: Dict = {}
    if enable_thinking is not None:
        extra_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        # Qwen3.8 resolves low / medium / xhigh in its tokenizer chat
        # template. This is prompt construction, not an SGLang sampling
        # parameter, so it must be forwarded to apply_chat_template here.
        extra_kwargs["reasoning_effort"] = reasoning_effort
    if messages[0]["role"] == "system":
        # NOTE 非常的离谱，transformers 新版本把 return_dict 默认设置改成 Ture 了
        # 导致 apply_chat_template 返回的是一个 dict，而不是一个 list，手动处理一下
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=False,
            **extra_kwargs,
        )
        return token_ids
    else:
        # NOTE qwen3.5 的 jinja 要求必须要有 user role，否则报错... 所以这里加一个 user role 的 mock
        system_mock = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        system_token_ids_mock = tokenizer.apply_chat_template(
            system_mock, tokenize=True, return_dict=False
        )
        token_ids = tokenizer.apply_chat_template(
            system_mock + messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=False,
            **extra_kwargs,
        )
        return token_ids[len(system_token_ids_mock):]


def custom_vl_apply_chat_template(
    messages: List[Dict],
    collator: "DataCollatorWithPaddingForMM",
    add_generation_prompt=True
) -> Dict:
    if len(messages) == 0:
        return {}

    images = []
    for message in messages:
        if message["role"] == "user":
            content: List[Dict] = message["content"]
            images.extend(
                [
                    content[i].pop("image_PIL")
                    for i in range(len(content)) if content[i]["type"] == "image"
                ]
            )

    if messages[0]["role"] == "system":
        messages_text = collator.processor.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, return_dict=False
        )
        features = [
            {
                collator.prompt_key: messages_text,
                collator.image_key: images,
                collator.image_flag_key: True
            }
        ]
        inputs = collator(features)
        inputs.pop("position_ids", None)
        return inputs
    else:
        system_mock = [{"role": "system", "content": ""}]
        system_token_ids_mock = collator.processor.apply_chat_template(
            system_mock, tokenize=True, return_dict=False
        )
        messages_text = collator.processor.apply_chat_template(
            system_mock + messages, return_dict=False
        )
        features = [
            {
                collator.prompt_key: messages_text,
                collator.image_key: images,
                collator.image_flag_key: True
            }
        ]
        inputs = collator(features)
        inputs.pop("position_ids", None)
        inputs["input_ids"] = inputs["input_ids"][:, len(system_token_ids_mock):]
        inputs["attention_mask"] = inputs["attention_mask"][:, len(system_token_ids_mock):]
        return inputs


def messages_to_tokens_and_masks(
    messages: List[Dict], tokenizer: PreTrainedTokenizer, add_generation_prompt=False
):
    """
    Convert a list of chat messages into token IDs and response masks.

    Args:
        messages: Chat messages, each with 'role' and 'content' keys.
        tokenizer: PreTrainedTokenizer.
        add_generation_prompt: Add a generation prompt for the last message.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            - Token ID lists per message.
            - Response masks (1 for tokens to be predicted, 0 otherwise).

    case:
        messages = [
            {"role": "system",
             "content": "You're a helpful assistant. "},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
            {"role": "assistant", "content": "Landon"}
        ]
        <|im_start|>system
        You're a helpful assistant. <|im_end|>
        <|im_start|>user
        What is the capital of France?<|im_end|>
        <|im_start|>assistant
        Paris<|im_end|>
        <|im_start|>assistant
        Landon<|im_end|>

        [[151644, 8948, 198, 2610, 2299, 264, 10950, 17847, 13, 220, 151645, 198], [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198], [151644, 77091, 198, 59604, 151645, 198], [151644, 77091, 198, 43, 11037, 151645]]
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
        [('<|im_start|>', 0, 151644), ('system', 0, 8948), ('\n', 0, 198), ('You', 0, 2610), ("'re", 0, 2299), (' a', 0, 264), (' helpful', 0, 10950), (' assistant', 0, 17847), ('.', 0, 13), (' ', 0, 220), ('<|im_end|>', 0, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('user', 0, 872), ('\n', 0, 198), ('What', 0, 3838), (' is', 0, 374), (' the', 0, 279), (' capital', 0, 6722), (' of', 0, 315), (' France', 0, 9625), ('?', 0, 30), ('<|im_end|>', 0, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('assistant', 0, 77091), ('\n', 0, 198), ('Paris', 1, 59604), ('<|im_end|>', 1, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('assistant', 0, 77091), ('\n', 0, 198), ('L', 1, 43), ('andon', 1, 11037), ('<|im_end|>', 1, 151645)]

    """
    token_ids_list = []
    response_masks_list = []
    system_mock = {"role": "system", "content": ""}
    system_token_ids_mock = tokenizer.apply_chat_template(
        [system_mock], tokenize=True, return_dict=False
    )
    for i, message in enumerate(messages):
        if message["role"].lower() == "system":
            token_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)
            token_ids_list.append(token_ids)
            response_masks_list.append([0] * len(token_ids))
        if message["role"].lower() in ["user", "assistant"]:
            token_ids = tokenizer.apply_chat_template(
                [system_mock, message],
                tokenize=True,
                add_generation_prompt=add_generation_prompt and i == len(messages) - 1,
                return_dict=False
            )
            token_ids = token_ids[len(system_token_ids_mock):]
            if message["role"].lower() == "user":
                token_ids_list.append(token_ids)
                response_masks_list.append([0] * len(token_ids))
            elif message["role"].lower() == "assistant":
                assistant_token_ids = []
                response_masks = []
                token_id_without_format = tokenizer.encode(message["content"])
                first_assistant_idx = token_ids.index(token_id_without_format[0])
                assistant_token_ids.extend(token_ids[:first_assistant_idx])
                response_masks.extend([0] * first_assistant_idx)
                after_eos_token_id = False
                for idx in range(first_assistant_idx, len(token_ids)):
                    if i == len(messages) - 1 and after_eos_token_id and not add_generation_prompt:
                        break
                    assistant_token_ids.append(token_ids[idx])
                    if after_eos_token_id:
                        response_masks.append(0)
                    else:
                        response_masks.append(1)
                    if token_ids[idx] == tokenizer.eos_token_id:
                        after_eos_token_id = True
                token_ids_list.append(assistant_token_ids)
                response_masks_list.append(response_masks)

    return token_ids_list, response_masks_list


def token_ids_to_assistant_mask(
    messages: List[Dict], input_ids_list: List[List], tokenizer: PreTrainedTokenizer
):
    """
    Reconstruct per-message assistant masks from already-tokenized inputs.

    Args:
        messages: Original chat messages; only ``"assistant"`` ones are inspected.
        input_ids_list: Already-tokenized version of ``messages``, aligned index-by-index.
        tokenizer: HF tokenizer whose ``.eos_token_id`` marks the end of an assistant turn.

    Returns:
        List[List[int]] with 1 for tokens belonging to the assistant's actual
        response (excluding format / control tokens) and 0 elsewhere.
    """
    token_ids_list = []
    assistant_mask_list = []
    for i, (message, token_ids) in enumerate(zip(messages, input_ids_list)):
        if message["role"].lower() in ["system", "user"]:
            token_ids_list.append(token_ids)
            assistant_mask_list.append([0] * len(token_ids))
        if message["role"].lower() in ["assistant"]:
            assistant_token_ids = []
            assistant_mask = []
            token_id_without_format = tokenizer.encode(message["content"])
            first_assistant_idx = len(token_ids)
            if len(token_id_without_format) > 0:
                first_assistant_idx = token_ids.index(token_id_without_format[0])
            assistant_token_ids.extend(token_ids[:first_assistant_idx])
            assistant_mask.extend([0] * first_assistant_idx)
            after_eos_token_id = False
            for idx in range(first_assistant_idx, len(token_ids)):
                assistant_token_ids.append(token_ids[idx])
                if after_eos_token_id:
                    assistant_mask.append(0)
                else:
                    assistant_mask.append(1)
                if token_ids[idx] == tokenizer.eos_token_id:
                    after_eos_token_id = True
            token_ids_list.append(assistant_token_ids)
            assistant_mask_list.append(assistant_mask)

    return assistant_mask_list


def split_by_token(input_ids: list, token: int) -> list[list]:
    """
    Split ``input_ids`` by the given token; each sub-list starts with ``token``.

    Args:
        input_ids: list[int].
        token: Splitting token id.

    Returns:
        List[List[int]].

    Example:
      >>> split_by_token(
      ...     input_ids=[151644, 8948, 198, 151644, 872, 198],
      ...     token=151644
      ... )
      [[151644, 8948, 198], [151644, 872, 198]]
    """

    if not input_ids:
        return []

    result = []
    current_segment = []

    for item in input_ids:
        if item == token:
            if current_segment:
                result.append(current_segment)
            current_segment = [item]
        else:
            current_segment.append(item)

    if current_segment:
        result.append(current_segment)

    return result
