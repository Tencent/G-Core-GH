"""PPO policy/critic padding checks, run at actor init before the critic engine.

What each check covers, and the routes it skips: ``gpatch_v4/gdebug/README.md``.
"""
from typing import List

from gpatch_v4.utils.logging_utils import logging_rank0

VALID_MODES = ("off", "warn", "abort")


def _logprob_lens_follow_engine_padding(config) -> bool:
    """Whether the policy engine's padding is what sets the log-prob lengths.

    ``skip_prev_logps`` and ``dynamic_context_parallel`` size them per sample
    instead. Both are read for truthiness, the way the host reads them.
    """
    if config.ppo.skip_prev_logps:
        return False
    return not config.policy.dist_config.dynamic_context_parallel


def _find_risky_padding_settings(config) -> List[str]:
    """Return one diagnostic per risky setting, empty when none applies."""
    risks: List[str] = []
    if config.ppo.advantage_type != "ppo" or config.training.training_backend != "mcore":
        return risks

    policy, critic = config.policy, config.critic
    if _logprob_lens_follow_engine_padding(config):
        if critic.smart_pad_infer and not policy.smart_pad_infer:
            risks.append(
                "critic.smart_pad_infer=True with policy.smart_pad_infer=False: the critic "
                "values can come out shorter than the log-probs they are aligned against, "
                "which either stops the run in the aligner or left-pads them, shifting every "
                "value later by the number of zeros prepended. Set critic.smart_pad_infer=False."
            )
        elif (
            policy.smart_pad_infer and critic.smart_pad_infer and
            policy.forward_only_mbs != critic.forward_only_mbs
        ):
            risks.append(
                f"policy.forward_only_mbs={policy.forward_only_mbs} != "
                f"critic.forward_only_mbs={critic.forward_only_mbs} with smart_pad_infer=True "
                f"on both: the two engines group samples differently, so the same sample can "
                f"be padded to two different lengths. Set both fields to the same value."
            )
    return risks


def maybe_check_ppo_padding(config) -> None:
    mode = config.debug.ppo_padding_check_mode
    if mode not in VALID_MODES:
        raise ValueError(
            f"config.debug.ppo_padding_check_mode={mode!r}, expected one of {VALID_MODES}"
        )
    if mode == "off":
        return
    risks = _find_risky_padding_settings(config)
    if not risks:
        return
    body = "\n  ".join(risks)
    if mode == "abort":
        # every rank reaches this hook and reads the same static config, so the
        # whole group leaves together
        raise RuntimeError(f"[gdebug.ppo_padding] risky PPO padding configuration:\n  {body}")
    logging_rank0(
        f"WARN: [gdebug.ppo_padding] risky PPO padding configuration, the run continues "
        f"and PPO results may be wrong:\n  {body}"
    )
