import copy


def merge_core_transformer_config(bridge_config, training_config=None):
    """
    add attr to bridge_config
    """
    config = copy.deepcopy(bridge_config)
    if training_config is not None:
        setattr(config, "ppo_dump_moe_topk", training_config.training.ppo_dump_moe_topk)
    return config
