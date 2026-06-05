"""
Bagel Trainer — inherits shared train_loop from BaseTrainer.
"""

from typing_extensions import override

from gpatch_v4.configs.bagel_configs import BagelConfig
from gpatch_v4.extended_pipeline.pipeline_bagel import FSDP2EngineBagel
from gpatch_v4.trainer.omni_base_trainer import OmniBaseTrainer
from gpatch_v4.trainer.validation_mixin import InTrainingValidationMixin
from gpatch_v4.utils import dataclass_from_args


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


class BagelTrainer(InTrainingValidationMixin, OmniBaseTrainer):
    arg_cls = BagelConfig
    log_prefix = "bagel"

    def __init__(self, args):
        args = dataclass_from_args(args, self.arg_cls)
        super().__init__(args)
        self.engine = self.build_engine()

    @override
    def build_engine(self):
        return FSDP2EngineBagel(self.args)

    @override
    def get_flop_coefficients(self):
        return qwen2_flop_coefficients(self.engine.llm_config)

    @override
    def build_train_valid_test_data_iter(self):
        # Bagel uses CustomBagelTrainer override in tasks/omni/bagel/
        raise NotImplementedError(
            "BagelTrainer.build_train_valid_test_data_iter must be overridden "
            "by CustomBagelTrainer in tasks/omni/bagel/."
        )
