import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

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
    async def test_eval_delay_is_awaited_when_configured(self):
        tok = _make_mock_tokenizer(["\\boxed{42}"])
        config = SimpleNamespace(
            external_reward=SimpleNamespace(eval_delay_s=60),
        )
        reward = MathRuleExternalReward(config=config, tokenizer=tok)
        batch = _make_batch(
            tokens_list=[[1, 2, 3, 4]],
            seq_lens=[4],
            gt_labels=[json.dumps({"answer": 42})],
        )

        with patch(
            "tasks.math_rl_v4.math_external_reward.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            await reward.calc_external_reward([batch], ppo_step=0, is_eval=True)

        sleep.assert_awaited_once_with(60)

    async def test_eval_delay_minus_one_uses_random_uniform(self):
        tok = _make_mock_tokenizer(["\\boxed{42}"])
        config = SimpleNamespace(
            external_reward=SimpleNamespace(eval_delay_s=-1),
        )
        reward = MathRuleExternalReward(config=config, tokenizer=tok)
        batch = _make_batch(
            tokens_list=[[1, 2, 3, 4]],
            seq_lens=[4],
            gt_labels=[json.dumps({"answer": 42})],
        )

        with (
            patch(
                "tasks.math_rl_v4.math_external_reward.random.uniform",
                return_value=12.5,
            ) as uniform,
            patch(
                "tasks.math_rl_v4.math_external_reward.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            await reward.calc_external_reward([batch], ppo_step=0, is_eval=True)

        uniform.assert_called_once_with(50.0, 200.0)
        sleep.assert_awaited_once_with(12.5)

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


class AsyncEvalExternalRewardTest(unittest.IsolatedAsyncioTestCase):
    class _EvalRolloutGenerator:
        def __init__(self):
            self.clear_data_cache = MagicMock()

        async def __call__(self, data_iter, num_microbatches, ppo_step):
            return [{"ppo_step": ppo_step}]

        def add_back_rollout_attr_after_sampling(self, rollout_batches):
            return rollout_batches

    class _DelayedExternalReward:
        def __init__(self):
            self.release = asyncio.Event()
            self.calls = []

        async def calc_external_reward(
            self,
            rollout_batches,
            ppo_step,
            is_eval=False,
            _started_event=None,
        ):
            self.calls.append((ppo_step, is_eval))
            if _started_event is not None:
                _started_event.set()
            await self.release.wait()
            return [{} for _ in rollout_batches]

    def _make_actor(self):
        from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor

        actor = object.__new__(GrpoTrainActor)
        actor.config = SimpleNamespace(
            training=SimpleNamespace(
                use_external_reward=True,
                total_eval_step=1,
            ),
            external_reward=SimpleNamespace(async_eval=True),
        )
        actor.external_reward = self._DelayedExternalReward()
        actor._pending_eval_external_rewards = None
        actor.eval_rollout_generator = self._EvalRolloutGenerator()
        actor.eval_dataloader = [object()]
        actor.policy_engine = SimpleNamespace(
            set_model_eval=MagicMock(),
            set_model_train=MagicMock(),
        )
        actor.get_num_eval_rollout_micro_batches = MagicMock(return_value=1)
        actor.compute_rollout_metrics = MagicMock(return_value={"rewards": 1.0})
        actor.eval_logging = MagicMock()
        return actor

    async def test_async_rewards_report_on_next_eval_and_final_flush(self):
        from gpatch_v4.actor import grpo_train_actor as actor_module

        actor = self._make_actor()
        timers = MagicMock()

        with (
            patch(
                "gpatch_v4.actor.grpo_train_actor.TimerSingleton.get_timer",
                return_value=timers,
            ),
            patch("gpatch_v4.actor.grpo_train_actor.cpu_barrier"),
            patch("gpatch_v4.actor.grpo_train_actor.clear_memory"),
            patch("gpatch_v4.actor.grpo_train_actor.check_rollout_batches", return_value=True),
            patch.object(
                actor_module.BroadcastUtils,
                "broadcast_rollout_batch",
                side_effect=lambda batches: batches,
            ),
        ):
            await actor._eval_loop(10)

            self.assertEqual(actor.external_reward.calls, [(10, True)])
            actor.eval_logging.assert_not_called()
            self.assertIsNotNone(actor._pending_eval_external_rewards)

            second_eval = asyncio.create_task(actor._eval_loop(20))
            await asyncio.sleep(0)
            self.assertFalse(second_eval.done())

            actor.external_reward.release.set()
            await second_eval

            actor.eval_logging.assert_called_once_with(
                {"eval-rewards": 1.0},
                10,
                report_step=20,
                commit=False,
            )
            self.assertEqual(actor.external_reward.calls, [(10, True), (20, True)])

            await actor._maybe_report_pending_eval_external_rewards(commit=True, force=True)
            self.assertEqual(
                actor.eval_logging.call_args_list,
                [
                    call({"eval-rewards": 1.0}, 10, report_step=20, commit=False),
                    call({"eval-rewards": 1.0}, 20, commit=True),
                ],
            )
            self.assertIsNone(actor._pending_eval_external_rewards)
            self.assertEqual(actor.eval_rollout_generator.clear_data_cache.call_count, 2)

    async def test_async_rewards_report_on_train_step_poll(self):
        from gpatch_v4.actor import grpo_train_actor as actor_module

        actor = self._make_actor()
        timers = MagicMock()

        with (
            patch(
                "gpatch_v4.actor.grpo_train_actor.TimerSingleton.get_timer",
                return_value=timers,
            ),
            patch("gpatch_v4.actor.grpo_train_actor.cpu_barrier"),
            patch("gpatch_v4.actor.grpo_train_actor.clear_memory"),
            patch("gpatch_v4.actor.grpo_train_actor.check_rollout_batches", return_value=True),
            patch.object(
                actor_module.BroadcastUtils,
                "broadcast_rollout_batch",
                side_effect=lambda batches: batches,
            ),
        ):
            await actor._eval_loop(10)
            self.assertIsNotNone(actor._pending_eval_external_rewards)

            await actor._maybe_report_pending_eval_external_rewards(report_step=11)
            actor.eval_logging.assert_not_called()
            self.assertIsNotNone(actor._pending_eval_external_rewards)

            actor.external_reward.release.set()
            await asyncio.sleep(0)

            await actor._maybe_report_pending_eval_external_rewards(report_step=12)
            actor.eval_logging.assert_called_once_with(
                {"eval-rewards": 1.0},
                10,
                report_step=12,
                commit=False,
            )
            self.assertIsNone(actor._pending_eval_external_rewards)

            await actor._eval_loop(20)
            self.assertEqual(actor.external_reward.calls, [(10, True), (20, True)])
            # Previous pending already reclaimed; new eval is pending again.
            actor.eval_logging.assert_called_once()
            self.assertIsNotNone(actor._pending_eval_external_rewards)


if __name__ == "__main__":
    unittest.main()
