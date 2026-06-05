from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator
from gpatch_v4.rollout_generator.ds_generator import DynamicSamplingRolloutGenerator
from gpatch_v4.rollout_generator.off_policy_distill_generator import (
    OffPolicyDistillRolloutGenerator,
)
from gpatch_v4.rollout_generator.on_policy_distill_generator import OnPolicyDistillRolloutGenerator
from gpatch_v4.rollout_generator.replay_generator import ReplayRolloutGenerator


class RolloutGeneratorFactory:
    """Factory that returns the appropriate rollout generator."""
    @staticmethod
    def get_rollout_generator(
        config,
        sampler_client,
        gen_rm_client,
        bt_rm_client,
        run_eval=False,
        **kwargs
    ) -> BaseRolloutGenerator:
        """Instantiate a rollout generator based on ``config.policy.rollout_gen_type``.

        Parameters
        ----------
        config : object
        sampler_client : object
        gen_rm_client : object
        bt_rm_client : object
        run_eval : bool, optional
        **kwargs : dict

        Returns
        -------
        BaseRolloutGenerator
        """
        rollout_generator_cls = None
        if config.policy.rollout_gen_type == "base":
            rollout_generator_cls = BaseRolloutGenerator
        elif config.policy.rollout_gen_type == "replay":
            rollout_generator_cls = ReplayRolloutGenerator
        elif config.policy.rollout_gen_type == "dynamic_sampling":
            rollout_generator_cls = DynamicSamplingRolloutGenerator
        elif config.policy.rollout_gen_type == "on_policy_distill":
            rollout_generator_cls = OnPolicyDistillRolloutGenerator
        elif config.policy.rollout_gen_type == "off_policy_distill":
            rollout_generator_cls = OffPolicyDistillRolloutGenerator
        elif config.policy.rollout_gen_type == "partial_rollout":
            raise NotImplementedError("partial_rollout is not implemented")
        elif config.policy.rollout_gen_type == "external":
            from gpatch_v4.utils.common_utils import import_fn_from_path
            rollout_generator_cls = import_fn_from_path(
                config.policy.rollout_gen_py_path,
                config.policy.rollout_gen_cls_name,
            )

        return rollout_generator_cls(
            config, sampler_client, gen_rm_client, bt_rm_client, run_eval, **kwargs
        )
