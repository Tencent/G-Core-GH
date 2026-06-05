from gpatch_v4.core.advantage_helper import (
    ADVANTAGE_DISPATCH,
    POST_ADVANTAGE_DISPATCH,
    AdvantageContext,
    AdvantageResult,
    PostAdvantageContext,
    PostAdvantageResult,
    get_advantage_fn,
    get_post_advantage_fn,
    register_custom_advantage,
    register_custom_post_advantage,
)

BUILDIN_ADVANTAGE_TYPE = list(ADVANTAGE_DISPATCH.keys())
BUILDIN_POST_ADVANTAGE_TYPE = list(POST_ADVANTAGE_DISPATCH.keys())
