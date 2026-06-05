import sys
import os

from gpatch.training.arguments import gpatch_extra_args


def get_tasks_args(parser):
    parser = gpatch_extra_args(parser)

    group = parser.add_argument_group(title='dqa-ppo-actor')
    group.add_argument(
        '--ppo-sampling-keeping-strategy',
        type=str,
        default='best-and-worst',
        choices=['best-and-worst', 'test', 'all'],
        help='''
ppo 多重采样保留策略。需要自己确保能 match `--ppo-sampling-keep` 。
                       '''
    )
    group.add_argument('--use-tool-calling', action='store_true', help='use tool calling of sglang')
    group.add_argument(
        '--max-tool-calling-rounds', type=int, default=1, help='max rounds of tool calling'
    )
    return parser
