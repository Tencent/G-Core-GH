def ppo_mm_extra_args(extra_args_provider, parser):
    """Extra arguments."""
    parser = extra_args_provider(parser)

    group = parser.add_argument_group(title='ppo mm extra arguments')
    group.add_argument(
        '--ppo-mm-rule-type',
        choices=[
            'captcha',
            'nq_hotpotq',
            'geometry3k',
            'agent',
            "import_file",
            "pmc_vqa",
            "image_pair",
            'geometry3k-v2',
        ],
        type=str,
        default='captcha',
        help='多模态中不同的 rule 计算规则'
    )
    return parser
