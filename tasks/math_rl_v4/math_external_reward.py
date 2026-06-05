"""Math rule-based external reward for AgentLoopActor.

Reuses ``cal_accuracy_reward`` and ``cal_format_reward`` from
``bt_reward.py`` but packaged as a ``BaseExternalReward`` subclass
so it can be plugged into the async rollout pipeline.

Unlike the GPU-actor ``DemoExternalReward``, this implementation
runs in a single-process AgentLoopActor (0 GPU, no torch.distributed),
so it does **not** use ``BroadcastUtils`` or ``is_mp_and_cp_head``.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import torch

from tasks.math_rl_v4.bt_reward import cal_accuracy_reward, cal_format_reward

from gpatch_v4.reward.base_external_reward import BaseExternalReward
from gpatch_v4.utils.training_utils import list_of_tensor_to_list


class MathRuleExternalReward(BaseExternalReward):
    """Rule-based math reward (accuracy + format).

    Designed to run inside ``AgentLoopActor`` / ``TwoTurnReflectAgentLoopActor``
    where there is no distributed context.  Computation is purely local
    (no network I/O), so it runs synchronously inside the coroutine.

    Parameters
    ----------
    config : RlConfig or None
        Top-level RL config (unused here, kept for interface compat).
    tokenizer : PreTrainedTokenizer or None
        Actor tokenizer used to decode generated tokens.
    """
    def __init__(self, config=None, tokenizer=None):
        super().__init__(config=config, tokenizer=tokenizer)
        assert tokenizer is not None, "MathRuleExternalReward requires a tokenizer"

    async def calc_external_reward(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
        is_eval: bool = False,
        _started_event: Optional[asyncio.Event] = None,
    ) -> List[Dict[str, Any]]:
        """Compute per-batch reward updates.

        Returns one update dict per input batch; the caller is
        responsible for merging each update back into its rollout batch
        (e.g. ``rb.update(updates[i])``).
        """
        # No async I/O here; signal readiness right away for interface compat.
        if _started_event is not None:
            _started_event.set()

        return [self._compute_update(batch) for batch in rollout_batches]

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _compute_update(self, batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Return a dict of fields to merge into ``batch``."""
        tokens_list = batch["tokens"]
        seq_lens = batch["sequence_lengths"]
        gt_labels = batch.get("gt_label", batch.get("labels"))
        assert gt_labels is not None, "gt_label or labels must be present"

        # Convert tensors to plain lists
        tokens_cpu = list_of_tensor_to_list(tokens_list, False)
        seq_len_cpu = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in seq_lens]

        # Parse gt_label (may be str-encoded JSON or tensor)
        if isinstance(gt_labels[0], str):
            gt_values = []
            for label in gt_labels:
                parsed = json.loads(label)
                gt_values.append(float(parsed["answer"]))
        else:
            gt_values = list_of_tensor_to_list(gt_labels, True)

        # Truncate tokens to actual sequence length
        for i in range(len(tokens_cpu)):
            tokens_cpu[i] = tokens_cpu[i][:seq_len_cpu[i]]

        # Decode and score
        resp_strs = self.tokenizer.batch_decode(tokens_cpu, skip_special_tokens=False)
        acc_reward, _, _ = cal_accuracy_reward(resp_strs, gt_values)
        fmt_reward = cal_format_reward(resp_strs)

        # Combine into per-sample reward tensors
        acc_t = torch.tensor(acc_reward, dtype=torch.float32).view(-1, 1)
        fmt_t = torch.tensor(fmt_reward, dtype=torch.float32).view(-1, 1)
        combined = acc_t + fmt_t
        n = combined.size(0)

        # Store as list of per-sample tensors (same format as DemoExternalReward)
        external_reward = [combined[i] for i in range(n)]
        acc_reward_list = [acc_t[i] for i in range(n)]
        fmt_reward_list = [fmt_t[i] for i in range(n)]

        upd: Dict[str, Any] = {
            "external_reward": external_reward,
            "acc_reward": acc_reward_list,
            "fmt_reward": fmt_reward_list,
        }
        # Also write "rewards" so that advantage computation (GRPO etc.)
        # can use it when gen_rm / bt_rm are disabled.
        prev_rewards = batch.get("rewards")
        if prev_rewards is None:
            upd["rewards"] = external_reward
        else:
            upd["rewards"] = [prev_rewards[i] + external_reward[i] for i in range(n)]
        return upd
