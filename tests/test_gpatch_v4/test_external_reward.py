import json
import unittest
from unittest.mock import MagicMock

import torch

from tasks.math_rl_v4.math_external_reward import MathRuleExternalReward

from gpatch_v4.configs.config import RlConfig
from gpatch_v4_test_helper import load_config

_YAML_NAME = "test_math_rl_async_rollout_2turns_rollout"


def _make_mock_tokenizer(decoded_strs):
    tok = MagicMock()
    tok.batch_decode = MagicMock(return_value=list(decoded_strs))
    return tok


def _make_batch(tokens_list, seq_lens, gt_labels, prev_rewards=None):
    batch = {
        "tokens": [torch.tensor(t, dtype=torch.long) for t in tokens_list],
        "sequence_lengths":
            [
                torch.tensor(s, dtype=torch.long) if not isinstance(s, torch.Tensor) else s
                for s in seq_lens
            ],
        "gt_label": gt_labels,
    }
    if prev_rewards is not None:
        batch["rewards"] = prev_rewards
    return batch


class MathRuleExternalRewardTest(unittest.IsolatedAsyncioTestCase):
    async def test_calc_external_reward_output_schema(self):
        tok = _make_mock_tokenizer(
            ["The final answer is \\boxed{42}", "The final answer is \\boxed{7}"]
        )
        reward = MathRuleExternalReward(tokenizer=tok)

        batch = _make_batch(
            tokens_list=[[1, 2, 3, 4], [5, 6, 7, 8]],
            seq_lens=[4, 4],
            gt_labels=[json.dumps({"answer": 42}),
                       json.dumps({"answer": 42})],
        )

        updates = await reward.calc_external_reward([batch], ppo_step=0)

        self.assertEqual(len(updates), 1)
        upd = updates[0]

        for key in ("external_reward", "acc_reward", "fmt_reward", "rewards"):
            self.assertIn(key, upd, f"missing key {key}")
            self.assertIsInstance(upd[key], list)
            self.assertEqual(len(upd[key]), 2)
            for t in upd[key]:
                self.assertIsInstance(t, torch.Tensor)
                self.assertEqual(t.dtype, torch.float32)

    async def test_rewards_merge_without_prior(self):
        tok = _make_mock_tokenizer(["\\boxed{42}", "nope"])
        reward = MathRuleExternalReward(tokenizer=tok)

        batch = _make_batch(
            tokens_list=[[1], [2]],
            seq_lens=[1, 1],
            gt_labels=[json.dumps({"answer": 42}),
                       json.dumps({"answer": 42})],
        )
        upd = (await reward.calc_external_reward([batch], ppo_step=0))[0]

        self.assertEqual(len(upd["rewards"]), len(upd["external_reward"]))
        for r, e in zip(upd["rewards"], upd["external_reward"]):
            self.assertTrue(torch.equal(r, e))

    async def test_rewards_merge_with_prior(self):
        tok = _make_mock_tokenizer(["\\boxed{42}", "\\boxed{42}"])
        reward = MathRuleExternalReward(tokenizer=tok)

        prior = [torch.tensor([0.5]), torch.tensor([1.5])]
        batch = _make_batch(
            tokens_list=[[1], [2]],
            seq_lens=[1, 1],
            gt_labels=[json.dumps({"answer": 42}),
                       json.dumps({"answer": 42})],
            prev_rewards=prior,
        )
        upd = (await reward.calc_external_reward([batch], ppo_step=0))[0]

        for i, (r, e) in enumerate(zip(upd["rewards"], upd["external_reward"])):
            expected = prior[i] + e
            self.assertTrue(
                torch.allclose(r, expected),
                f"idx={i}: got {r.tolist()}, expected {expected.tolist()}",
            )

    async def test_numerical_known_case(self):
        resp_strs = [
            "The final answer is \\boxed{42}",  # acc=1, fmt=1
            "The final answer is \\boxed{7}",  # acc=0, fmt=1
            "answer: \\boxed{abc}",  # acc=0, fmt=0 (non-numeric box)
            "no box at all, 42",  # acc=0, fmt=0
        ]
        tok = _make_mock_tokenizer(resp_strs)
        reward = MathRuleExternalReward(tokenizer=tok)

        gt = 42
        batch = _make_batch(
            tokens_list=[[1], [2], [3], [4]],
            seq_lens=[1, 1, 1, 1],
            gt_labels=[json.dumps({"answer": gt})] * 4,
        )
        upd = (await reward.calc_external_reward([batch], ppo_step=0))[0]

        expected_acc = [1.0, 0.0, 0.0, 0.0]
        expected_fmt = [1.0, 1.0, 0.0, 0.0]
        expected_ext = [a + f for a, f in zip(expected_acc, expected_fmt)]

        for i in range(4):
            self.assertAlmostEqual(upd["acc_reward"][i].item(), expected_acc[i])
            self.assertAlmostEqual(upd["fmt_reward"][i].item(), expected_fmt[i])
            self.assertAlmostEqual(upd["external_reward"][i].item(), expected_ext[i])
            self.assertAlmostEqual(upd["rewards"][i].item(), expected_ext[i])


class AgentLoopActorSetupExternalRewardTest(unittest.TestCase):
    def test_happy_path_instantiates_reward(self):
        from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import (
            AgentLoopActor,
        )

        config = load_config(_YAML_NAME, RlConfig)

        actor = object.__new__(AgentLoopActor)
        actor.config = config
        actor.training_config = config.training
        actor.worker_id = 0
        actor.tokenizer = _make_mock_tokenizer([])
        actor.external_reward = None

        actor._setup_external_reward()

        self.assertIsNotNone(actor.external_reward)
        self.assertEqual(type(actor.external_reward).__name__, "MathRuleExternalReward")
        self.assertTrue(callable(getattr(actor.external_reward, "calc_external_reward", None)))
        self.assertIs(actor.external_reward.tokenizer, actor.tokenizer)


if __name__ == "__main__":
    unittest.main()
