"""Placement-type helpers shared by orchestration, rollout, and clients."""

from __future__ import annotations

from typing import Any

COLOCATE = "colocate"
DISAGGREGATED = "disaggregated"
PARTIAL_COLOCATED = "partial_colocated"

RL_PLACEMENT_TYPES = (COLOCATE, DISAGGREGATED, PARTIAL_COLOCATED)
LEGACY_PLACEMENT_TYPES = (COLOCATE, DISAGGREGATED)


def placement_type(config_or_value: Any) -> str:
    """Return a placement string from either a config object or the value itself."""
    return getattr(config_or_value, "placement_type", config_or_value)


def is_colocate(config_or_value: Any) -> bool:
    return placement_type(config_or_value) == COLOCATE


def is_disaggregated(config_or_value: Any) -> bool:
    return placement_type(config_or_value) == DISAGGREGATED


def is_partial_colocated(config_or_value: Any) -> bool:
    return placement_type(config_or_value) == PARTIAL_COLOCATED


def use_ipc_weight_update(config_or_value: Any) -> bool:
    """Whether policy-to-sampler weights use IPC rather than NCCL broadcast."""
    return placement_type(config_or_value) in (COLOCATE, PARTIAL_COLOCATED)


def _num_gpus(dist_config: Any) -> int:
    return dist_config.nnodes * dist_config.num_gpus_per_node


def partial_colocated_role_num_gpus(config: Any) -> tuple[int, int, int]:
    """Return ``(policy, sampler, gen_rm)`` GPU counts for partial colocate."""
    policy_gpus = _num_gpus(config.policy.dist_config)
    sampler_gpus = _num_gpus(config.sampler.dist_config)
    gen_rm_gpus = _num_gpus(config.gen_rm.dist_config)
    return policy_gpus, sampler_gpus, gen_rm_gpus


def validate_partial_colocated_config(config: Any) -> None:
    """Fail fast for the first supported ``partial_colocated`` shape."""
    if not is_partial_colocated(config):
        return

    assert type(config).__name__ in ("RlConfig", "AISearchRlConfig"), (
        "partial_colocated currently supports only RlConfig/AISearchRlConfig"
    )
    training = config.training
    assert training.single_controller, "partial_colocated requires training.single_controller=True"
    assert training.async_rollout, "partial_colocated requires training.async_rollout=True"
    assert training.rollout_max_staleness == 0, (
        "partial_colocated requires training.rollout_max_staleness=0"
    )
    assert training.use_gen_rm_reward, "partial_colocated requires training.use_gen_rm_reward=True"
    assert not training.use_bt_rm_reward, "partial_colocated currently does not support bt_rm"

    assert getattr(
        config.sampler, "backend", "sglang"
    ) == "sglang", ("partial_colocated currently supports only sampler.backend='sglang'")
    assert getattr(
        config.gen_rm, "backend", "sglang"
    ) == "sglang", ("partial_colocated currently supports only gen_rm.backend='sglang'")
    assert len(config.sampler.model_info
              ) == 1, ("partial_colocated currently supports exactly one sampler model")
    assert len(
        config.sampler.infer_engine_configs
    ) == 1, ("partial_colocated currently supports exactly one sampler infer engine config")

    policy_gpus, sampler_gpus, gen_rm_gpus = partial_colocated_role_num_gpus(config)
    assert sampler_gpus > 0, "partial_colocated requires sampler GPUs"
    assert gen_rm_gpus > 0, "partial_colocated requires gen_rm GPUs"
    assert policy_gpus == sampler_gpus + gen_rm_gpus, (
        "partial_colocated requires policy GPUs == sampler GPUs + gen_rm GPUs, "
        f"got policy={policy_gpus}, sampler={sampler_gpus}, gen_rm={gen_rm_gpus}"
    )

    sampler_engine_dc = config.sampler.infer_engine_configs[0].dist_config
    sampler_engine_gpus = _num_gpus(sampler_engine_dc)
    assert sampler_engine_gpus == sampler_gpus, (
        "partial_colocated requires sampler.dist_config and sampler infer engine GPUs to match, "
        f"got sampler={sampler_gpus}, infer_engine={sampler_engine_gpus}"
    )
    sampler_mp = (
        sampler_engine_dc.tensor_model_parallel_size *
        sampler_engine_dc.pipeline_model_parallel_size
    )
    assert sampler_gpus % sampler_mp == 0, (
        f"sampler GPUs ({sampler_gpus}) must be divisible by sampler MP size ({sampler_mp})"
    )

    assert len(config.gen_rm.reward_model_info) == len(
        config.gen_rm.infer_engine_configs
    ), ("gen_rm.reward_model_info and gen_rm.infer_engine_configs must have the same length")
