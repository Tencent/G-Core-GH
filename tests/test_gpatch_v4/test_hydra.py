import unittest

from gpatch_v4.configs.config import T2iRlConfig
from gpatch_v4_test_helper import load_config


class HydraTest(unittest.TestCase):
    def test_1(self):
        config = load_config('test_t2i_grpo_gen_rm', T2iRlConfig)
        assert config.gen_rm.reward_model_info[0].model_arch == 'qwen2_5_vl'
