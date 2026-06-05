import json
import sys
import os

from megatron_datasets.utils import print_rank_0, _print_args


def _add_dataset_extra_args(parser):
    # NOTE(parkeychen):
    # megatron_datasets/args.py中前向兼容，避免影响除tasks/math_rl_v3以外的代码
    # tasks/math_rl_v3相关代码切至GDatasetV4，不再支持旧数据集
    # 指定--px-data-config-path走老分支
    # 指定--gdatasetv4-train-metadata-file走GDatasetV4分支
    # 两者互斥
    data_config_exclusive_group = parser.add_mutually_exclusive_group()
    data_config_exclusive_group.add_argument(
        "--px-data-config-path", type=str, default=None, help="The path of data config"
    )
    data_config_exclusive_group.add_argument(
        "--gdatasetv4-train-metadata-file",
        type=str,
        default=None,
        help="The path of the train metadata.json of GDatasetV4",
    )

    group = parser.add_argument_group(title='dataset extra args')
    group.add_argument(
        "--px-vision-encoder-config-path",
        type=str,
        default=None,
        help="The path of yaml config, using for multimodal"
    )
    group.add_argument(
        "--px-domain-probabilities",
        nargs="*",
        type=float,
        default=None,
        help="The propabilities of domains"
    )
    group.add_argument(
        "--px-retention-rates-per-domain",
        type=float,
        nargs='*',
        default=None,
        help="The drop rates per domain"
    )
    group.add_argument(
        "--px-data-file-format",
        type=str,
        default="jsonl",
        choices=['jsonl', 'pkl'],
        help="select the data file format"
    )
    group.add_argument(
        "--px-eval-data-path", nargs='*', default=None, type=str, help="the path of eval data"
    )
    group.add_argument(
        "--px-eval-data-file-format",
        type=str,
        default="jsonl",
        choices=['jsonl', 'pkl'],
        help="select the data version"
    )
    group.add_argument(
        "--px-total-samples-of-dataset",
        type=int,
        default=0,
        help="Required when using sft iterable dataset, "
    )
    group.add_argument(
        "--px-eval-samples-per-domain",
        type=int,
        nargs='*',
        default=None,
        help="The propabilities of multi files"
    )
    group.add_argument(
        "--px-eval-data-domain-names",
        type=str,
        nargs='*',
        default=None,
        help="all domain names of eval data"
    )
    group.add_argument(
        '--px-finetune-eval-num',
        type=int,
        default=0,
        help="Required when using llama sft eval iterable dataset"
    )
    group.add_argument(
        "--px-eval-iters-per-domain",
        type=int,
        nargs='*',
        default=None,
        help="The propabilities of multi files"
    )
    group.add_argument(
        '--px-shuffle-data', action='store_true', help="whether to shuffle data or not"
    )
    group.add_argument(
        "--px-shuffle-strategy",
        type=str,
        choices=['no_shuffle', 'interleave', 'intra_domain', 'inter_domain'],
        default='interleave',
        help="shuffle strategy, currently only support v1. Choose from\n" \
        "no_shuffle: do not shuffle data\n" \
        "interleave: retrieve sample from multi domains interleavely\n" \
        "intra_domain: shuffle data within each domain and retrieve sample sequentially\n" \
        "inter_domain: shuffle data across all domains and retrieve sample sequentially\n" \
    )
    group.add_argument(
        "--px-shuffle-buffer-size",
        type=int,
        default=1000000,
        help="the buffer size of local shuffle"
    )
    group.add_argument(
        "--no-px-pad-to-max-len",
        action='store_false',
        dest="px_pad_to_max_len",
        help='whether to pad_to_max_len or not'
    )
    group.add_argument(
        "--px-train-data-domain-names", type=str, nargs='*', default=None, help="all domain names"
    )
    group.add_argument(
        '--px-use-indexed-jsonl-dataset',
        action='store_true',
        help='use indexed jsonl dataset or not'
    )
    group.add_argument(
        '--px-indexed-jsonl-dataset-version',
        type=str,
        default='v1',
        choices=['v1', 'v2', 'v3'],
        help='indexed jsonl dataset type'
    )
    group.add_argument('--px-top-domains-to-cut', type=int, default=1, help='px_top_domains_to_cut')
    group.add_argument(
        '--px-indexed-jsonl-dataset-access-policy-interleave',
        action='store_true',
        help='indexed jsonl dataset access policy = interleave or not'
    )
    group.add_argument('--px-eval-real-len',
                       type=int,
                       default=None,
                       help=f'When training using 12/16k, if you want eval to be comparable to 4k training, ' \
                            f'you need to set --px-eval-real-len to generate the same eval data as 4k training.')
    group.add_argument(
        "--px-auto-cal-eval-iters",
        action='store_true',
        help='whether to auto cal for eval iters, default 1 epoch'
    )
    group.add_argument(
        "--apply-pareto-sampling", action='store_true', help='whether to enalbe pareto sampling'
    )
    group.add_argument(
        "--px-train-apply-pareto", type=int, nargs='*', default=None, help="px_train_apply_pareto"
    )
    group.add_argument(
        "--px-train-pareto-alpha",
        type=float,
        nargs='*',
        default=None,
        help="px_train_pareto_alpha"
    )
    group.add_argument(
        "--px-train-pareto-scale",
        type=float,
        nargs='*',
        default=None,
        help="px_train_pareto_scale"
    )
    group.add_argument(
        "--train-pareto-score-scale",
        type=float,
        nargs='*',
        default=None,
        help="train_pareto_score_scale"
    )
    group.add_argument(
        '--assert-too-long',
        action='store_true',
        help='If the dpo data is too long, it raises an assertion error'
    )
    group.add_argument(
        "--gdatasetv4-eval-metadata-file",
        type=str,
        default=None,
        help="The path of the eval metadata.json of GDatasetV4",
    )
    group.add_argument(
        '--px-apply-chat-template', action='store_true', help='whether apply chat template or not'
    )
    group.add_argument("--px-system-prompt", type=str, default=None, help='the assistant prompt')
    return parser


def ppo_new_data_loader_calc_args(args):
    guessed_total_num = args.max_samples or 1024
    args.ppo_step_per_epoch = guessed_total_num // args.ppo_rollout_global_batch_size
    kept_gbs = args.ppo_rollout_global_batch_size * args.ppo_sampling_keep
    assert kept_gbs % args.global_batch_size == 0
    args.train_iters_each_rollout = kept_gbs // args.global_batch_size * args.ppo_max_epochs_2
    args.train_iters = args.ppo_max_epochs * args.ppo_step_per_epoch * args.train_iters_each_rollout
    assert args.ppo_step_save_interval > 0
    args.save_interval = args.ppo_step_save_interval * args.train_iters_each_rollout
    if not hasattr(args, "train_data_consuming_progresses"):
        args.train_data_consuming_progresses = {}


# automatically calculates:
#  1. train-iters
#  2. ppo-step-per-epoch
def ppo_auto_calc_args(args):
    use_dataset_version = "v3"
    if args.gdatasetv4_train_metadata_file:
        with open(args.gdatasetv4_train_metadata_file, 'r') as f:
            metadata = json.load(f)
        rollout_gbs = args.ppo_rollout_global_batch_size
        data_file_num_lines = metadata['data_file_num_lines']
        data_file_num_lines = [x // rollout_gbs * rollout_gbs for x in data_file_num_lines]
        guessed_total_num = sum(data_file_num_lines)
        use_dataset_version = "v4"
    elif args.px_data_config_path:
        with open(args.px_data_config_path, 'r') as f:
            data_config = json.load(f)

        guessed_total_num = 0
        if "train_data_infos" in data_config.keys():
            for key, values in data_config["train_data_infos"].items():
                train_data_path = values["path"]
                sample_rate = float(values["sample_rate"])
                assert sample_rate == 1.0
                probability = float(values["probability"])
                with open(os.path.join(train_data_path, 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                total_num = metadata['total_num']
                guessed_total_num = max(
                    int(total_num / probability), guessed_total_num
                )  # FIXME interleaving all exhausted assumed
    else:
        raise ValueError(
            "--gdatasetv4-train-metadata-file or --px-data-config-path must be specified"
        )

    eval_guessed_total_num = 0
    # if eval enabled, calculate len of eval dataset after drop last
    if args.ppo_step_eval_interval > 0:
        if use_dataset_version == "v4":
            if not os.path.exists(args.gdatasetv4_eval_metadata_file):
                raise ValueError(
                    f"eval metadata file {args.gdatasetv4_eval_metadata_file} not exists"
                )

            with open(args.gdatasetv4_eval_metadata_file, 'r') as f:
                eval_metadata = json.load(f)
            eval_rollout_gbs = args.ppo_eval_rollout_global_batch_size
            eval_data_file_num_lines = eval_metadata['data_file_num_lines']
            eval_data_file_num_lines = [
                x // eval_rollout_gbs * eval_rollout_gbs for x in eval_data_file_num_lines
            ]
            eval_guessed_total_num = sum(eval_data_file_num_lines)
        elif use_dataset_version == "v3":
            with open(args.px_data_config_path, 'r') as f:
                data_config = json.load(f)
            if "eval_data_infos" not in data_config.keys():
                raise ValueError(f"eval_data_infos not in the file {args.px_data_config_path}")

            eval_guessed_total_num = 0
            eval_rollout_gbs = args.ppo_eval_rollout_global_batch_size
            for key, values in data_config["eval_data_infos"].items():
                train_data_path = values["path"]
                sample_rate = values.get("sample_rate", 1.0)
                assert sample_rate == 1.0
                probability = values.get("probability", 1.0)
                with open(os.path.join(train_data_path, 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                total_num = metadata['total_num']
                eval_guessed_total_num = max(int(total_num / probability), eval_guessed_total_num)
        else:
            raise ValueError(f"{use_dataset_version=}")

    ppo_auto_calc_args = getattr(args, "ppo_auto_calc_args", False)
    if not ppo_auto_calc_args:
        return guessed_total_num, eval_guessed_total_num
    assert args.train_iters == -1
    assert args.save_interval == -1
    assert args.ppo_step_per_epoch == -1
    assert args.train_iters_each_rollout == -1

    # PPO 的 epoch / step 概念不同于普通 train，epoch-1 来自 nemo aligner，ppo step 来自
    # nemo aligner，epoch-2 来自 trl。
    #
    # ```python
    # for epoch in range(ppo_max_epochs):
    #   for ppo_step in range(ppo_step_per_epoch):
    #       for epoch_2 in range(ppo_max_epochs_2):
    #           for rb in rollout_batches:
    #               train_step() # actual iteration
    # ```

    # TODO @astrachang maybe critic ?
    if args.rl_role == 'actor':
        assert args.ppo_rollout_global_batch_size % (
            args.data_parallel_size * args.ppo_rollout_micro_batch_size
        ) == 0
        args.ppo_step_per_epoch = guessed_total_num // args.ppo_rollout_global_batch_size

        kept_gbs = args.ppo_rollout_global_batch_size * args.ppo_sampling_keep
        assert kept_gbs % args.global_batch_size == 0
        args.train_iters_each_rollout = kept_gbs // args.global_batch_size * args.ppo_max_epochs_2
        args.train_iters = args.ppo_max_epochs * args.ppo_step_per_epoch * args.train_iters_each_rollout

        assert args.ppo_step_save_interval > 0
        args.save_interval = args.ppo_step_save_interval * args.train_iters_each_rollout

        # ppo_actor_freeze_ppo_steps 是 ppo step 的层次
        # freeze_iters 则是程序内部定义的 iteration 层次
        args.freeze_iters = 0
        if args.ppo_actor_freeze_ppo_steps > 0:
            assert args.train_samples is None
            args.freeze_iters = args.ppo_actor_freeze_ppo_steps * args.train_iters_each_rollout

        # if eval enabled, auto cal and check
        if args.ppo_step_eval_interval > 0:
            assert args.ppo_eval_rollout_global_batch_size % (
                args.data_parallel_size * args.ppo_eval_rollout_micro_batch_size
            ) == 0

            # auto cal ppo_eval_steps if set to -1
            if args.ppo_eval_steps == -1:
                assert (
                    eval_guessed_total_num >= args.ppo_eval_rollout_global_batch_size
                ), f"number of eval samples {eval_guessed_total_num} (after drop last) < eval gbs {args.ppo_eval_rollout_global_batch_size}"

                args.ppo_eval_steps = eval_guessed_total_num // args.ppo_eval_rollout_global_batch_size

            assert args.ppo_eval_steps > 0, f"ppo_eval_steps should be > 0, but got {args.ppo_eval_steps=}"

    _print_args("PPO auto calc args", args)
    return guessed_total_num, eval_guessed_total_num


def parse_dataset_config(args):

    if args.use_new_dataloader:
        ppo_new_data_loader_calc_args(args)
        return

    if args.gdatasetv4_train_metadata_file:
        # TODO(parkeychen): interleave datasets not implemented
        # temporary
        with open(args.gdatasetv4_train_metadata_file, 'r') as f:
            metadata = json.load(f)
        args.px_domain_probabilities = [1.0]
        args.px_retention_rates_per_domain = [1.0]
        args.px_train_data_domain_names = [metadata['name']]
    elif args.px_data_config_path:
        px_data_config_path = args.px_data_config_path
        with open(px_data_config_path, 'r') as f:
            data_config = json.load(f)

            train_data_path = []
            train_probability = []
            train_sample_rate = []
            train_data_domain_names = []
            train_apply_pareto = []
            train_pareto_alpha = []
            train_pareto_scale = []
            train_pareto_score_scale = []
            eval_data_path = []
            eval_sample_nums_per_domain = []
            eval_data_domain_names = []
            if "train_data_infos" in data_config.keys():
                for key, values in data_config["train_data_infos"].items():
                    train_data_path.append(values["path"])
                    train_probability.append(float(values["probability"]))
                    train_sample_rate.append(float(values["sample_rate"]))
                    train_data_domain_names.append(key)
                    if args.apply_pareto_sampling:
                        train_apply_pareto.append(int(values.get("apply_pareto", 0)))
                        train_pareto_alpha.append(float(values.get("pareto_alpha", 9.0)))
                        train_pareto_scale.append(float(values.get("pareto_scale", 1.0)))
                        train_pareto_score_scale.append(
                            float(values.get("pareto_score_scale", 1.0))
                        )
                args.data_path = train_data_path
                args.px_domain_probabilities = train_probability
                # TODO @xiaotaoliu 修一下？drop = sample 略显离谱，建议依旧保持 sample rate 的 naming。
                args.px_retention_rates_per_domain = train_sample_rate
                args.px_train_data_domain_names = train_data_domain_names
                args.px_train_apply_pareto = train_apply_pareto
                args.px_train_pareto_alpha = train_pareto_alpha
                args.px_train_pareto_scale = train_pareto_scale
                args.train_pareto_score_scale = train_pareto_score_scale

            # mega eval_iters <= 0 有特殊含义，即不做 eval；不要加入。
            if "eval_data_infos" in data_config.keys(
            ) and (args.eval_iters > 0 or (args.use_grpo and args.ppo_step_eval_interval > 0)):
                for key, values in data_config["eval_data_infos"].items():
                    eval_data_path.append(values["path"])
                    # 兼容性考虑做保留，默认为 -1。
                    eval_sample_nums_per_domain.append(values.get("eval_samples_num", -1))
                    eval_data_domain_names.append(key)
                args.px_eval_data_path = eval_data_path
                args.px_eval_samples_per_domain = eval_sample_nums_per_domain
                args.px_eval_data_domain_names = eval_data_domain_names

        # 排序 domains（一开始为了 doremi 做的，后来发现排序起来有些代码处理简单）
        # TODO(@xiaotaoliu)：这 6 个 args naming 有点谜，没什么规则。不过有空再说吧，先做高优先的...
        # train
        if args.px_domain_probabilities is not None:
            args.px_domain_probabilities = [
                x for _, x in sorted(zip(args.data_path, args.px_domain_probabilities))
            ]
            args.px_retention_rates_per_domain = [
                x for _, x in sorted(zip(args.data_path, args.px_retention_rates_per_domain))
            ]
            args.px_train_data_domain_names = [
                x for _, x in sorted(zip(args.data_path, args.px_train_data_domain_names))
            ]
            if args.apply_pareto_sampling:
                # 必须做排序，否则对应会错乱
                args.px_train_apply_pareto = [
                    x for _, x in sorted(zip(args.data_path, args.px_train_apply_pareto))
                ]
                args.px_train_pareto_alpha = [
                    x for _, x in sorted(zip(args.data_path, args.px_train_pareto_alpha))
                ]
                args.px_train_pareto_scale = [
                    x for _, x in sorted(zip(args.data_path, args.px_train_pareto_scale))
                ]
                args.train_pareto_score_scale = [
                    x for _, x in sorted(zip(args.data_path, args.train_pareto_score_scale))
                ]
        args.data_path = sorted(args.data_path)

        # eval
        if args.px_eval_data_path:
            args.px_eval_samples_per_domain = [
                x for _, x in sorted(zip(args.px_eval_data_path, args.px_eval_samples_per_domain))
            ]
            args.px_eval_data_domain_names = [
                x for _, x in sorted(zip(args.px_eval_data_path, args.px_eval_data_domain_names))
            ]
            args.px_eval_data_path = sorted(args.px_eval_data_path)

        assert args.px_domain_probabilities is None or len(args.data_path
                                                          ) == len(args.px_domain_probabilities)
        if args.px_data_file_format == "pkl":
            assert args.px_retention_rates_per_domain is not None
            assert len(args.data_path) == len(args.px_retention_rates_per_domain)

        if args.px_eval_data_path:
            assert args.px_eval_data_domain_names is not None
            assert args.px_eval_samples_per_domain is not None
            assert len(args.px_eval_data_path) == len(args.px_eval_samples_per_domain)
            assert len(args.px_eval_data_path) == len(args.px_eval_data_domain_names)
            '''
            实际上对于 on-the-fly 处理 + 攒 seq 的情况（例如 pretrain，sft 不 pad），无法提前确定 train-iter 和 eval-iter，
            只能是近似。对于 train 而言，最多是多点数据少点数据的区别，问题不大。

            这里回归 megatron-lm 原版的逻辑：
            1. 用户给出 eval-iter，指定错了就错了。
            2. 如果 eval-iter 太小了，没有消费完，那么会留到下一次 eval，导致 eval 数据略有不同。有 log 可以观察。
            3. 如果 `args.eval_iters` 太大了，会重放数据，不会挂掉。有 log 可以观察。

            说实话这里 eval-iters 没处理好，但 nvidia 原版也是这样的 bug，先不处理了。如果你需要定制化的 eval-iter
            逻辑再联系 nrwu and xiaotaoliu。

            下面是之前 xiaotaoliu 做 pretrain 留下的逻辑，也是正确的。每个 domain 的 eval sample 个数。这里之前没考虑好，
            如果是 on-the-fly 处理，map 后的 sample 个数无法确定，而且绑定了 tokenizer。
            '''
            if len(args.px_eval_samples_per_domain
                  ) > 0 and all([c >= 0 for c in args.px_eval_samples_per_domain]):
                eval_iters_per_domain = []
                for eval_sample_nums in args.px_eval_samples_per_domain:
                    eval_iters_per_domain.append(eval_sample_nums // args.global_batch_size)
                args.px_eval_iters_per_domain = eval_iters_per_domain
                assert args.px_auto_cal_eval_iters or args.eval_iters == sum(args.px_eval_iters_per_domain), \
                    f"The sum of px_eval_iters_per_domain {sum(args.px_eval_iters_per_domain)} must " \
                    + f"be equal to eval_iters {args.eval_iters}"
            else:
                assert args.px_do_eval_per_domain is False, \
                    f"setting px_do_eval_per_domain must pass eval_samples_num for each domain data"
    else:
        raise ValueError(
            "--gdatasetv4-train-metadata-file or --px-data-config-path must be specified"
        )

    # 记录数据消费进度，在load_user_train_data_consuming_progresses时可能被创建了
    if not hasattr(args, "train_data_consuming_progresses"):
        args.train_data_consuming_progresses = {}

    guessed_total_num, eval_guessed_total_num = ppo_auto_calc_args(args)
    if args.px_inputs_pad_to_longest:
        if args.use_grpo:
            total_data_num_per_dp = guessed_total_num * args.ppo_sampling_keep // args.data_parallel_size
        else:
            total_data_num_per_dp = guessed_total_num // args.data_parallel_size
        assert args.px_smart_padding_buffer_size <= total_data_num_per_dp, \
            f"px_smart_padding_buffer_size {args.px_smart_padding_buffer_size} should be <= len(dataset) // dp_size {total_data_num_per_dp}"

    print_rank_0(
        f"parse_dataset_config gdatasetv4_train_metadata_file {args.gdatasetv4_train_metadata_file}"
        f"parse_dataset_config gdatasetv4_eval_metadata_file {args.gdatasetv4_eval_metadata_file}"
        f"parse_dataset_config px_data_config_path {args.px_data_config_path}"
        f" data_path {args.data_path}"
        f" px_domain_probabilities {args.px_domain_probabilities}"
        f" px_retention_rates_per_domain {args.px_retention_rates_per_domain}"
        f" px_train_apply_pareto {args.px_train_apply_pareto}"
        f" px_train_pareto_alpha {args.px_train_pareto_alpha}"
        f" px_train_pareto_scale {args.px_train_pareto_scale}"
        f" train_pareto_score_scale {args.train_pareto_score_scale}"
        f" train_data_domain_names {args.px_train_data_domain_names}"
        f" px_eval_data_path {args.px_eval_data_path}"
        f" llama_eval_samples_per_domain {args.px_eval_samples_per_domain}"
        f" eval_iters {args.eval_iters}"
    )
