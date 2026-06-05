from gpatch_v4.core import AdvantageContext, AdvantageResult
from gpatch_v4.core.advantage_impl import calculate_grpo_advantages
from gpatch_v4.utils import log


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
