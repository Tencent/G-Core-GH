import unittest

import torch

from tasks.math_rl_v4.bt_reward_algin_steer import dapo_math_strict_box_reward


class _Tokenizer:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def batch_decode(
        self,
        token_ids: list[list[int]],
        skip_special_tokens: bool,
    ) -> list[str]:
        assert skip_special_tokens
        assert len(token_ids) == len(self.responses)
        return self.responses


def _make_batch(labels: list[object]) -> dict[str, list[object]]:
    batch_size = len(labels)
    return {
        "tokens": [torch.tensor([1, 2], dtype=torch.long) for _ in range(batch_size)],
        "sequence_lengths": [torch.tensor(2) for _ in range(batch_size)],
        "prompt_lengths": [torch.tensor(0) for _ in range(batch_size)],
        "gt_label": labels,
    }


class MathDapoRewardTest(unittest.TestCase):
    def test_strict_box_rewards(self) -> None:
        responses = [
            "first \\boxed{0}, final \\boxed{42}",
            "\\boxed{7}",
            "there is no boxed answer",
        ]
        rewards, per_token_rewards, metrics = dapo_math_strict_box_reward(
            [_make_batch(["42", "42", "42"])],
            actor_tokenizer=_Tokenizer(responses),
        )

        self.assertIsNone(per_token_rewards)
        torch.testing.assert_close(rewards, torch.tensor([[1.0], [-1.0], [-1.0]]))
        torch.testing.assert_close(metrics["acc_reward"], torch.tensor([[1.0], [0.0], [0.0]]))

    def test_nested_box_and_last_300_characters(self) -> None:
        responses = [
            "\\boxed{\\frac{1}{2}}",
            "\\boxed{42}" + "x" * 300,
        ]
        rewards, _, metrics = dapo_math_strict_box_reward(
            [_make_batch([r"\frac{1}{2}", "42"])],
            actor_tokenizer=_Tokenizer(responses),
        )

        torch.testing.assert_close(rewards, torch.tensor([[1.0], [-1.0]]))
        torch.testing.assert_close(metrics["acc_reward"], torch.tensor([[1.0], [0.0]]))
