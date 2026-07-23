from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from gpatch_v4.utils.training_utils import list_of_tensor_to_list


def _last_boxed_only_string(response: str) -> Optional[str]:
    start = response.rfind("\\boxed{")
    if start < 0:
        return None

    open_braces = 0
    for index in range(start, len(response)):
        if response[index] == "{":
            open_braces += 1
        elif response[index] == "}":
            open_braces -= 1
            if open_braces == 0:
                return response[start:index + 1]
    return None


def _extract_boxed_content(response: str) -> str:
    boxed = _last_boxed_only_string(response)
    if boxed is None:
        return "[INVALID]"
    return boxed[len("\\boxed{"):-1]


def _ground_truth_to_string(label: Any) -> str:
    if isinstance(label, torch.Tensor):
        label = label.item()
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label)


def dapo_math_strict_box_reward(
    batched_data: list[dict[str, list[Any]]],
    tokenizer: AutoTokenizer = None,
    actor_tokenizer: AutoTokenizer = None,
) -> tuple[torch.Tensor, None, dict[str, torch.Tensor]]:
    """Score decoded responses using VERL's strict boxed-answer reward.

    Parameters
    ----------
    batched_data : list[dict[str, list[Any]]]
        Each batch must provide response token boundaries and ground-truth labels.
    tokenizer : AutoTokenizer, optional
    actor_tokenizer : AutoTokenizer
        Decodes only tokens after each prompt boundary.

    Returns
    -------
    torch.Tensor
        Shape ``(N, 1)`` with ``+1`` for exact final-box matches and ``-1`` otherwise.
    None
    dict[str, torch.Tensor]
        Contains ``acc_reward`` with shape ``(N, 1)``.
    """
    del tokenizer
    assert actor_tokenizer is not None

    tokens = []
    sequence_lengths = []
    prompt_lengths = []
    labels = []
    for batch in batched_data:
        tokens.extend(batch["tokens"])
        sequence_lengths.extend(batch["sequence_lengths"])
        prompt_lengths.extend(batch["prompt_lengths"])
        if "gt_label" in batch:
            labels.extend(batch["gt_label"])
        elif "labels" in batch:
            labels.extend(batch["labels"])
        else:
            raise ValueError("neither 'gt_label' nor 'labels' found in batch")

    tokens_cpu = list_of_tensor_to_list(tokens, False)
    sequence_lengths = [int(length.item()) for length in sequence_lengths]
    prompt_lengths = [int(length.item()) for length in prompt_lengths]
    assert len(tokens_cpu) == len(sequence_lengths) == len(prompt_lengths) == len(labels)

    response_token_ids = [
        token_ids[prompt_length:sequence_length]
        for token_ids, prompt_length, sequence_length in zip(
            tokens_cpu, prompt_lengths, sequence_lengths
        )
    ]
    responses = actor_tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)

    rewards = []
    accuracies = []
    for response, label in zip(responses, labels):
        prediction = _extract_boxed_content(response[-300:])
        correct = prediction == _ground_truth_to_string(label)
        rewards.append(1.0 if correct else -1.0)
        accuracies.append(float(correct))

    rule_reward = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
    acc_reward = torch.tensor(accuracies, dtype=torch.float32).view(-1, 1)
    return rule_reward, None, {"acc_reward": acc_reward}
