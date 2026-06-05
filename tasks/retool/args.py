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
    group.add_argument(
        '--ppo-debug-simulate-webapi',
        action='store_true',
        help='use asyncio.sleep to simulate calling webapi'
    )
    group.add_argument(
        '--max-concurrency-all-dp',
        type=int,
        default=256,
        help='max concurrency limit on calling webapi for all DP ranks'
    )
    group.add_argument(
        '--no-hook-webapi', action='store_true', help='use hook to simulate calling webapi'
    )

    # ReTool-specific arguments for multi-turn tool-calling GRPO
    group.add_argument(
        '--retool-max-turns',
        type=int,
        default=8,
        help='Maximum number of assistant turns in ReTool multi-turn generation'
    )
    group.add_argument(
        '--retool-sandbox-timeout',
        type=int,
        default=20,
        help='Timeout in seconds for code execution in sandbox'
    )
    group.add_argument(
        '--retool-max-response-per-turn',
        type=int,
        default=4096,
        help='Maximum tokens per turn in ReTool multi-turn generation'
    )

    # ReTool/VERL-style: answer format and tools for Qwen tool-use template
    group.add_argument(
        '--px-answer-format',
        type=str,
        default=None,
        help='Answer format string to append to user messages (VERL-style)'
    )
    group.add_argument(
        '--px-tools-json',
        type=str,
        default=None,
        help='JSON string of tools schema for Qwen tool-use template'
    )
    group.add_argument(
        "--disable-shuffle-data-files", action='store_true', help='disable shuffle data files'
    )
    return parser
