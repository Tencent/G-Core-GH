import torch

from gpatch_v4.core import AdvantageContext, AdvantageResult
from gpatch_v4.core.advantage_impl import calculate_grpo_advantages
from gpatch_v4.core.ppo_feature_store import (
    feature_history_key,
    get_ppo_feature_store,
)
from gpatch_v4.utils import log

_ADV_MEAN = "adv_mean"
_ADV_MIN = "adv_min"
_ADV_MAX = "adv_max"


# 仅作为一个外部注入的 demo，这个函数本身相比 grpo 没做啥改动
def custom_grpo_advantage(ctx: AdvantageContext) -> AdvantageResult:
    assert ctx.rewards is not None
    advantages, returns = calculate_grpo_advantages(
        rewards=ctx.rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        sample_mask=ctx.sample_mask,
    )
    log(f"custom_grpo_advantage: {advantages=}, {returns=}", rank=0)

    return AdvantageResult(advantages=advantages, returns=returns)


# GRPO + 读 adv_* history，并把本 microbatch 的 mean/min/max record 进 feature_store
def custom_grpo_advantage_with_adv_stats(ctx: AdvantageContext) -> AdvantageResult:
    assert ctx.rewards is not None
    advantages, returns = calculate_grpo_advantages(
        rewards=ctx.rewards,
        mask=ctx.mask,
        grpo_sampling_times=ctx.config.training.sampling_keep_n,
        grpo_advantage_epsilon=ctx.config.ppo.grpo_advantage_epsilon,
        sample_mask=ctx.sample_mask,
    )

    store = get_ppo_feature_store()
    hist_mean = store.get(feature_history_key(_ADV_MEAN), [])
    hist_min = store.get(feature_history_key(_ADV_MIN), [])
    hist_max = store.get(feature_history_key(_ADV_MAX), [])
    log(
        f"custom_grpo_advantage_with_adv_stats history: "
        f"adv_mean={hist_mean}, adv_min={hist_min}, adv_max={hist_max}",
        rank=0,
    )

    valid = [a[m.bool()] for a, m in zip(advantages, ctx.mask) if m.bool().any()]
    if valid:
        vals = torch.cat(valid).float()
        store.record(_ADV_MEAN, float(vals.sum().item()), weight=float(vals.numel()), reduce="mean")
        store.record(_ADV_MIN, float(vals.min().item()), reduce="min")
        store.record(_ADV_MAX, float(vals.max().item()), reduce="max")

    return AdvantageResult(advantages=advantages, returns=returns)
