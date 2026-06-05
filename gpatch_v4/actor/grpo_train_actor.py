import asyncio
import inspect
import os
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler

from megatron.core import mpu
from megatron.core.utils import divide

from gpatch_v4.actor.mixin import (
    CheckpointConverterMixin,
    MetricsMixin,
    OnloadManager,
    RetryActorMixin,
    RlTrainerMixin,
    TokenizerMixin,
)
from gpatch_v4.client import BtRmClient, GenRmClient, SamplerClient
from gpatch_v4.core import (
    BUILDIN_ADVANTAGE_TYPE,
    BUILDIN_POST_ADVANTAGE_TYPE,
    register_custom_advantage,
    register_custom_post_advantage,
)
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.core.smart_pad_helper import (
    DPBalanceHelper,
    smart_pad_train_get_reorder_rollout_batches,
)
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.rollout_generator import RolloutGeneratorFactory
from gpatch_v4.training_backend import (
    BUILDIN_LOSS_FUNC,
    TrainingEngineFactory,
    register_custom_loss_fn,
)
from gpatch_v4.utils import (
    BUILDIN_FILTER_SAMPLING_STRATEGIES,
    BroadcastUtils,
    FilterSamplingRegistry,
    TimerSingleton,
    TrainReporterSingleton,
    catch_exception_ctx_async,
    check_rollout_batches,
    clear_memory,
    display_rollout_generation,
    expand_rollout_batch,
    expand_rollout_batches,
    extend_value_to_dict,
    format_config,
    get_iterator_k_split_list,
    import_fn_from_path,
    init_timer_singleton,
    init_train_reporter_singleton,
    log,
    logging_meminfo_str,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
    record_time_to_metrics,
    reduce_metrics,
    register_custom_filter_sampling,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.common_utils import compress_ppo_save_train_data
from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler
from gpatch_v4.utils.test_utils import save_data


class GrpoTrainActor(
    BaseActor, TokenizerMixin, RlTrainerMixin, MetricsMixin, CheckpointConverterMixin,
    RetryActorMixin
):
    """Ray actor for LM GRPO training.

    Orchestrates dataset loading, model setup, rollout generation,
    advantage computation, and PPO training loops.
    """
    @catch_exception_ctx_async("GrpoTrainActor.init")
    async def init(self, config):
        """Initialize the actor: parallel state, tokenizer, dataset, and models.

        Parameters
        ----------
        config : RlConfig
        """
        super().init(config)
        self.retry_actor_init()

        # init parallel group
        initlize_parallel_state(config, config.policy.dist_config)
        init_pg(config.policy.dist_config)

        self.build_tokenizer()
        self.tokenizer = self.actor_tokenizer
        self.disp_rng = random.Random(self.config.training.seed)

        self.build_dataset_and_dataloader()
        self.prev_ppo_step = 0
        self.validated_config()
        logging_rank0(f"{self.__class__.__name__} config {format_config(self.config)}")

        if self.require_critic_model():
            extra_args = {
                "policy_config": config.critic,
                "tokenizer": self.tokenizer,
                "is_critic_model": True
            }
            self.critic_engine = TrainingEngineFactory.get_training_engine(config, **extra_args)

        extra_args = {"policy_config": config.policy, "tokenizer": self.tokenizer}
        self.policy_engine = TrainingEngineFactory.get_training_engine(config, **extra_args)
        # TODO: 这里最好不要用户填，麻烦得很
        self.metrics_report = self.config.training.metrics_report if self.config.training.metrics_report else []
        self.compact_thread = None
        if (not config.checkpoint.convert_mcore_to_hf_offline) and is_last_rank():
            init_train_reporter_singleton(self.config.report, self.config)
        init_timer_singleton(self.config.report)
        self.load_hf_config()

    def _save_dumped_metrics(self, _metrics, expanded_rbs, ppo_step_i, epoch_idx):
        """Collect base, loss_fn and MoE-topk data, and save to disk."""
        training_config = self.config.training
        dumped_loss_and_moe_metrics = _metrics.pop("dumped_metrics_per_ppo_step", None)

        if is_mp_and_cp_head():
            # merge base, loss_fn and moe metrics
            assert dumped_loss_and_moe_metrics is not None, f"dumped_loss_and_moe_metrics is None"
            all_dumped_metrics = []
            for rb in expanded_rbs:
                all_dumped_metrics.append(
                    {
                        "ppo_step":
                            ppo_step_i,
                        "tokens":
                            rb["tokens"].detach().to(torch.int32),
                        "gt_label":
                            rb.get("gt_label"),
                        "rewards":
                            rb["rewards"],
                        "rewards_details":
                            rb.get("rewards_details"),
                        "rollout_logprobs":
                            rb["rollout_log_probs"].clone().detach().to(torch.bfloat16),
                        "pre_logprobs":
                            rb["logprobs"].clone().detach().to(torch.bfloat16),
                        "advantages":
                            rb["advantages"].detach().to(torch.bfloat16),
                    }
                )
            assert len(all_dumped_metrics) == len(dumped_loss_and_moe_metrics), \
                f"all_dumped_metrics length {len(all_dumped_metrics)} != dumped_loss_and_moe_metrics length {len(dumped_loss_and_moe_metrics)}"
            for s1, s2 in zip(all_dumped_metrics, dumped_loss_and_moe_metrics):
                s1.update(s2)

            # save to .pt file
            save_path = os.path.join(
                training_config.ppo_dump_metrics_dir,
                f'tmp/PpoStep{ppo_step_i}_SubEpoch{epoch_idx}_{datetime.now().strftime("%Y%m%d_%H%M")}'
            )
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                all_dumped_metrics,
                os.path.join(
                    save_path,
                    f'dp{mpu.get_data_parallel_rank()}_rank{torch.distributed.get_rank()}.pt'
                )
            )

    def validated_config(self):
        """Validate training configuration constraints and auto-load settings."""
        training_config = self.config.training
        if training_config.sampling_keeping_strategy == "all":
            assert training_config.sampling_repeat_n == training_config.sampling_keep_n, \
                f"sampling_repeat_n {training_config.sampling_repeat_n} != sampling_keep_n {training_config.sampling_keep_n}"
        else:
            assert training_config.sampling_repeat_n >= training_config.sampling_keep_n

        assert training_config.rollout_gbs * training_config.sampling_keep_n >= training_config.train_gbs
        assert training_config.rollout_mbs * mpu.get_data_parallel_world_size(
        ) <= training_config.rollout_gbs
        assert training_config.train_mbs * mpu.get_data_parallel_world_size(
        ) <= training_config.train_gbs

        if self.config.placement_type == "colocate":
            policy_nnodes = self.config.policy.dist_config.nnodes
            sampler_nnodes = self.config.sampler.infer_engine_configs[0].dist_config.nnodes
            assert policy_nnodes == sampler_nnodes, (
                f"colocate mode requires policy nnodes ({policy_nnodes}) "
                f"== sampler nnodes ({sampler_nnodes})"
            )
        if self.config.training.auto_load_from_save_ckpt:
            if os.path.exists(
                os.path.join(
                    self.config.checkpoint.save_ckpt_path, 'latest_checkpointed_iteration.txt'
                )
            ):
                self.config.checkpoint.load_ckpt_path = self.config.checkpoint.save_ckpt_path
            self.config.checkpoint.no_load_optim = False
        if self.config.checkpoint.convert_mcore_to_hf_offline:
            self.config.checkpoint.no_load_optim = True

        if training_config.sampling_keeping_strategy not in BUILDIN_FILTER_SAMPLING_STRATEGIES:
            assert training_config.ppo_filter_samplings_path is not None
            register_custom_filter_sampling(
                training_config.sampling_keeping_strategy,
                training_config.ppo_filter_samplings_path,
                training_config.ppo_filter_samplings_name,
            )
        if self.config.ppo.loss_func not in BUILDIN_LOSS_FUNC:
            assert self.config.ppo.loss_func_py_path is not None and self.config.ppo.loss_func_py_name is not None, "Custom loss function must be provided"
            register_custom_loss_fn(
                self.config.ppo.loss_func,
                self.config.ppo.loss_func_py_path,
                self.config.ppo.loss_func_py_name,
            )
        if self.config.ppo.advantage_type not in BUILDIN_ADVANTAGE_TYPE:
            assert self.config.ppo.custom_advantage_py_path is not None and self.config.ppo.custom_advantage_py_name is not None, (
                f"Non-builtin advantage_type '{self.config.ppo.advantage_type}' requires "
                f"custom_advantage_py_path and custom_advantage_py_name to be set."
            )
            register_custom_advantage(
                self.config.ppo.advantage_type, self.config.ppo.custom_advantage_py_path,
                self.config.ppo.custom_advantage_py_name
            )
        if self.config.ppo.advantage_type not in BUILDIN_POST_ADVANTAGE_TYPE:
            custom_post_adv_path = self.config.ppo.custom_advantage_py_path
            custom_post_adv_name = self.config.ppo.custom_post_advantage_py_name
            if custom_post_adv_path is not None and custom_post_adv_name is not None:
                register_custom_post_advantage(
                    self.config.ppo.advantage_type, custom_post_adv_path, custom_post_adv_name
                )

        if self.config.ppo.skip_prev_logps:
            self._validate_skip_prev_logps()

    def _validate_skip_prev_logps(self):
        """Assert all preconditions for skipping the prev_logps forward pass."""
        training_config = self.config.training
        ppo_config = self.config.ppo

        assert training_config.train_gbs == training_config.rollout_gbs * training_config.sampling_keep_n, (
            f"skip_prev_logps requires strict on-policy: "
            f"train_gbs ({training_config.train_gbs}) == "
            f"rollout_gbs ({training_config.rollout_gbs}) * "
            f"sampling_keep_n ({training_config.sampling_keep_n})"
        )
        assert training_config.ppo_max_epochs_2 == 1, (
            f"skip_prev_logps requires ppo_max_epochs_2 == 1, "
            f"got {training_config.ppo_max_epochs_2}"
        )
        _supported_loss_funcs = {"grpo", "gspo"}
        assert ppo_config.loss_func in _supported_loss_funcs, (
            f"skip_prev_logps only supports loss_func in {_supported_loss_funcs}, "
            f"got '{ppo_config.loss_func}'"
        )
        _incompatible_adv_types = {"on_policy_distill", "g_opd"}
        assert ppo_config.advantage_type not in _incompatible_adv_types, (
            f"skip_prev_logps is incompatible with advantage_type='{ppo_config.advantage_type}'"
        )
        assert ppo_config.ppo_initial_policy_kl_penalty == 0, (
            f"skip_prev_logps requires ppo_initial_policy_kl_penalty == 0, "
            f"got {ppo_config.ppo_initial_policy_kl_penalty}"
        )

    @catch_exception_ctx_async("GrpoTrainActor.setup_client")
    async def setup_client(self):
        """Set up RPC clients for sampler, gen-RM, and BT-RM."""
        self.gen_rm_client = None
        self.bt_rm_client = None

        self.sampler_client = SamplerClient(self.config)
        await self.sampler_client.maybe_init_distributed_weight_group_for_disagg()
        if self.config.training.use_gen_rm_reward:
            self.gen_rm_client = GenRmClient(self.config)
        if self.config.training.use_bt_rm_reward:
            self.bt_rm_client = BtRmClient(self.config)

        self.external_reward = None
        if self.config.training.use_external_reward:
            self._setup_external_reward_class()

    def _setup_external_reward_class(self):
        """Set up the external reward from config."""
        external_reward_config = self.config.external_reward
        assert external_reward_config.reward_info is not None, \
            "external_reward.reward_info must be set when use_external_reward=True"
        assert len(external_reward_config.reward_info) == 1, \
            "Currently only one external reward is supported"
        rm_info = external_reward_config.reward_info[0]
        assert rm_info.reward_py_path is not None, "reward_py_path must be set"
        assert rm_info.reward_cls_name is not None, "reward_cls_name must be set"

        reward_cls = import_fn_from_path(rm_info.reward_py_path, rm_info.reward_cls_name)

        # Validate the class interface
        assert hasattr(reward_cls, 'calc_external_reward'), \
            f"{rm_info.reward_cls_name} must implement calc_external_reward"

        self.external_reward = reward_cls(config=self.config, tokenizer=self.tokenizer)
        logging_rank0(f"External reward cls initialized: {rm_info.reward_cls_name}")

    @catch_exception_ctx_async("GrpoTrainActor.setup_rollout_generator")
    async def setup_rollout_generator(self):
        """Instantiate the rollout generator using the factory."""
        self.train_rollout_generator = RolloutGeneratorFactory.get_rollout_generator(
            self.config,
            self.sampler_client,
            self.gen_rm_client,
            self.bt_rm_client,
        )

    @catch_exception_ctx_async("GrpoTrainActor.setup_model_and_optimizer")
    async def setup_model_and_optimizer(self):
        """Build training engines for policy and (optional) critic models."""
        logging_rank0(f"begin setup_model_and_optimizer...")
        # setup_model_and_get_optimizer for value model and offload
        if self.require_critic_model():
            self.critic_engine.setup_model_and_get_optimizer()
            # offload value model, grad buffer, optimizer state upon building engine
            self.critic_engine.offload_model()
            self.critic_engine.offload_optimizer()

        self.prev_ppo_step = self.policy_engine.setup_model_and_get_optimizer()
        logging_rank0(f"finished setup_model_and_optimizer at {self.prev_ppo_step} ...")
        if self.prev_ppo_step > 0:
            self.build_dataset_and_dataloader(self.prev_ppo_step)
            logging_rank0(f"dataloader rebuilt from {self.prev_ppo_step}.")

    async def save_checkpoint(self, step):
        """Save a policy checkpoint.

        Parameters
        ----------
        step : int
        """
        self.policy_engine.save_checkpoint(step)

    async def rollout_eval(self, ppo_step_i, num_rollout_micro_batches):
        timers = TimerSingleton.get_timer()
        rollout_batches = []

        rollout_batches = await self.train_rollout_generator(
            self.eval_iter,
            num_rollout_micro_batches,
            ppo_step_i,
        )
        cpu_barrier()
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)
        rollout_batches = self.train_rollout_generator.add_back_rollout_attr_after_sampling(
            rollout_batches
        )
        assert check_rollout_batches(rollout_batches)
        rollout_metrics = self.compute_rollout_metrics(rollout_batches)
        return rollout_batches, rollout_metrics

    async def _eval_loop(self, ppo_step):
        timers = TimerSingleton.get_timer()
        timers("eval_loop", log_level=0).start(barrier=True)

        training_config = self.config.training
        self.policy_engine.set_model_eval()
        self.eval_iter = iter(self.eval_dataloader)
        global_metrics = defaultdict(float)
        rollout_nb = self.get_num_rollout_micro_batches()

        for step in range(training_config.total_eval_step):
            clear_memory()
            rollout_batches, rollout_metrics = await self.rollout_eval(
                ppo_step_i=step,
                num_rollout_micro_batches=rollout_nb,
            )

            for k, v in rollout_metrics.items():
                global_metrics[f"eval-{k}"] += v

        for k in global_metrics:
            global_metrics[k] /= training_config.total_eval_step

        self.policy_engine.set_model_train()
        timers("eval_loop").stop()

        self.eval_logging(global_metrics, ppo_step)
        self.train_rollout_generator.clear_data_cache()
        cpu_barrier()

    def eval_logging(self, metrics, iteration):
        if is_last_rank():
            log_prefix = f"[GRPO EVAL] step {iteration}"
            TrainReporterSingleton.log_and_report(metrics, iteration, log_prefix=log_prefix)

    def build_dataset_and_dataloader(self, resume_step=None):
        """Build training dataset and dataloader from a user-provided factory function.

        Parameters
        ----------
        resume_step : int, optional
            If set, skip consumed samples for resuming.
        """
        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) >= 4,
                'config' in fn_kwargs,
                'tokenizer' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            extra_args = {}
            if resume_step is not None and "meta_info" in fn_kwargs:
                extra_args['meta_info'] = {'resume_step': resume_step}
            fn_ret = fn(
                config=self.config,
                tokenizer=self.tokenizer,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
                **extra_args,
            )
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

        self.eval_dataset = fn_ret.get('eval_dataset', None)
        self.eval_dataloader = fn_ret.get('eval_dataloader', None)
        self.eval_sampler = fn_ret.get('eval_sampler', None)

        self.train_iter = None
        if resume_step is not None:
            # 此时说明是第二次调用了，就不重复算了
            return
        self.auto_calc_ppo_step()

    def auto_calc_ppo_step(self):
        """Automatically compute total PPO steps, gradient accumulation steps, etc."""
        training_config = self.config.training
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()

        assert training_config.rollout_gbs * training_config.sampling_repeat_n >= training_config.train_gbs
        gas = training_config.train_gbs // (dp_size * training_config.train_mbs)
        rollout_gas = training_config.rollout_gbs // (dp_size * training_config.rollout_mbs)
        # TODO: train_dataloader 允许用户自定义的话，len() 在 dp rank 之间会不会不 match，导致程序
        # hang，有待处理。
        ppo_step_per_epoch = (len(self.train_dataloader) // rollout_gas)
        total_ppo_step = ppo_step_per_epoch * training_config.num_train_epoches

        training_config.total_ppo_step = total_ppo_step
        training_config.ppo_step_per_epoch = ppo_step_per_epoch
        training_config.gradient_accumulation_steps = gas
        log(
            f"train_dataset length: {len(self.train_dataloader)=} {len(self.train_dataset)=} "
            f"{dp_size=} {training_config.total_ppo_step=}"
        )

        if training_config.eval_interval > 0:
            assert self.eval_dataset is not None, f"enable eval:{training_config.eval_interval} should have eval_dataset"
            assert self.eval_dataloader is not None, f"enable eval:{training_config.eval_interval} should have eval_dataloader"

            use_eval_rollout = training_config.eval_rollout_gbs is not None

            eval_rollout_gas = training_config.eval_rollout_gbs // (
                dp_size * training_config.eval_rollout_mbs
            ) if use_eval_rollout else rollout_gas

            eval_step = (len(self.eval_dataloader) // eval_rollout_gas)
            training_config.total_eval_step = eval_step

            assert eval_step > 0, f"{len(self.eval_dataloader)=} > {eval_rollout_gas=}"

    def calculate_advantage(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
    ):
        """Compute advantages from rollout data (not implemented).

        Parameters
        ----------
        rollout_batches : list of dict
        """
        raise NotImplementedError()
        return rollout_batches

    def maybe_calculate_values(self, rollout_batches: List[Dict[str, List[Any]]]):
        if self.require_critic_model():
            self.policy_engine.offload_model()
            # value model is onloaded here
            old_values = self.critic_engine.compute_values(rollout_batches)
            for rb, v in zip(rollout_batches, old_values, strict=True):
                assert ("values" not in rb) or rb["values"][
                    0] is None, f"features key 'values' is reserved for internal usage"
                rb["values"] = v

    async def rollout(
        self, epoch_i, ppo_step_i, num_rollout_micro_batches, debug_disable_advantage=False
    ):
        """Generate rollouts, compute log probs, advantages, and metrics.

        Parameters
        ----------
        epoch_i : int
        ppo_step_i : int
        num_rollout_micro_batches : int
        debug_disable_advantage : bool, optional
            Skip advantage computation, by default *False*.

        Returns
        -------
        tuple[list[dict], dict]
            ``(rollout_batches, metrics)``.
        """
        timers = TimerSingleton.get_timer()
        rollout_batches = []

        timers("rollout", log_level=0).start(barrier=True)
        rollout_batches = await self.train_rollout_generator(
            self.train_iter,
            num_rollout_micro_batches,
            ppo_step_i,
        )

        cpu_barrier()
        timers("rollout").stop()
        logging_memory_usage_details("memory tracking before bcast data", rank=0)
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)
        clear_memory()
        logging_memory_usage_details("memory tracking after bcast data", rank=0)

        rollout_batches = self.train_rollout_generator.add_back_rollout_attr_after_sampling(
            rollout_batches
        )

        timers("external_reward", log_level=0).start(barrier=True)
        external_reward_task = None
        if self.config.training.use_external_reward and self.external_reward is not None:
            external_reward_started = asyncio.Event()
            external_reward_task = asyncio.create_task(
                self.external_reward.calc_external_reward(
                    rollout_batches,
                    ppo_step_i,
                    is_eval=False,
                    _started_event=external_reward_started,
                )
            )
            await external_reward_started.wait()

        assert check_rollout_batches(
            rollout_batches
        ), f"rollout_batches fmt error: {rollout_batches=}"
        logging_meminfo_str("CPU Memory: after get_rollout_batches: ")

        # filter sampling (pre-filter: before logprobs/advantage computation)
        training_config = self.config.training
        need_filter = training_config.sampling_keep_n != training_config.sampling_repeat_n
        filter_stage = training_config.filter_sampling_stage if need_filter else None

        if filter_stage == "pre":
            strategy = training_config.sampling_keeping_strategy
            samples_before = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            filter_sampling_fn = FilterSamplingRegistry.get(strategy)
            rollout_batches = filter_sampling_fn(
                self.config,
                rollout_batches,
                training_config.sampling_repeat_n,
                training_config.sampling_keep_n,
            )
            samples_after = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            logging_rank0(
                f"[filter] stage=pre, strategy={strategy}, "
                f"repeat_n={training_config.sampling_repeat_n}, "
                f"keep_n={training_config.sampling_keep_n}, "
                f"samples_per_batch: {samples_before} -> {samples_after}"
            )
        elif filter_stage == "post":
            logging_rank0(
                f"[filter] stage=post (deferred), strategy={training_config.sampling_keeping_strategy}, "
                f"repeat_n={training_config.sampling_repeat_n}, "
                f"keep_n={training_config.sampling_keep_n}, "
                f"processing all {training_config.sampling_repeat_n} samples through logprobs/advantage"
            )

        # post-filter defers filtering to after logprobs/advantage, so samples
        # per batch still uses repeat_n at this point.
        effective_keep_n = (
            training_config.sampling_repeat_n
            if filter_stage == "post" else training_config.sampling_keep_n
        )

        # compute logps
        samples_per_batch = training_config.rollout_mbs * effective_keep_n
        restore_info = None
        for data in rollout_batches:
            ll = len(data["tokens"])
            src_dp = [torch.tensor(mpu.get_data_parallel_rank())] * ll
            data["src_dp"] = src_dp

        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rebalanced_batches, restore_info = DPBalanceHelper.rebalance_for_compute_log_probs(
                rollout_batches,
                samples_per_batch,
                add_custom_keys=getattr(self.config.task, "add_custom_keys", None),
            )
            origin_rollout_batches = rollout_batches
            rollout_batches = rebalanced_batches

        skip_prev = self.config.ppo.skip_prev_logps
        timers("compute_logps", log_level=0).start(barrier=True)
        ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs(
            rollout_batches,
            compute_pre_logps=not skip_prev,
        )
        cpu_barrier()
        timers("compute_logps").stop()

        if external_reward_task is not None:
            reward_updates = await external_reward_task
            reward_updates = BroadcastUtils.broadcast_rollout_batch(reward_updates)
            for rb, updates in zip(rollout_batches, reward_updates):
                rb.update(updates)
        cpu_barrier()
        timers("external_reward").stop()

        if not self.config.policy.without_ref:
            for rb, ref_logps in zip(rollout_batches, ref_logprobs):
                rb["ref_logprobs"] = ref_logps
        if skip_prev:
            for rb in rollout_batches:
                rb["logprobs"] = [
                    torch.zeros(len(t) - 1, dtype=torch.float32) for t in rb["tokens"]
                ]
        else:
            for rb, prev_logps in zip(rollout_batches, prev_logprobs, strict=True):
                rb["logprobs"] = prev_logps

        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            rollout_batches = DPBalanceHelper.restore_log_probs_to_original_batches(
                origin_rollout_batches,
                rollout_batches,
                restore_info,
                without_ref=self.config.policy.without_ref,
            )

        clear_memory()
        logging_memory_usage_details("memory tracking after compute_log_probs", rank=0)
        logging_meminfo_str("CPU Memory: after get_rollout_batches: ")

        expected_mbs = training_config.rollout_mbs * effective_keep_n
        for rb in rollout_batches:
            assert len(
                rb['tokens']
            ) == expected_mbs, f"len(rb['tokens']) {len(rb['tokens'])} != {expected_mbs}"
        assert check_rollout_batches(rollout_batches), f"rbs fmt error: {rollout_batches=}"

        rollout_metrics = self.compute_rollout_metrics(rollout_batches)
        cpu_barrier()
        self.maybe_calculate_values(rollout_batches)
        timers("generate_ppo_data", log_level=0).start(barrier=True)
        rollout_batches, ppo_metrics = self.generate_ppo_data(rollout_batches)
        cpu_barrier()
        timers("generate_ppo_data").stop()
        metrics = rollout_metrics | ppo_metrics
        display_rollout_generation(self.tokenizer, self.disp_rng, rollout_batches)
        self.train_rollout_generator.clear_data_cache()

        return rollout_batches, metrics

    def permute_and_expand_rollout_batches(self, rollout_batches: List[Dict[str, List[Any]]]):
        """Permute timesteps and expand batched rollout dicts into single-sample dicts.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        list of dict
            Expanded per-sample dicts.
        """
        rollout_gbs = self.config.training.rollout_gbs
        rollout_mbs = self.config.training.rollout_mbs
        repeat_n = self.config.training.sampling_repeat_n
        expected_len = rollout_mbs * repeat_n

        extend_rollout_batches = []
        for rollout_batch in rollout_batches:
            self.extended_pipeline.permute_timesteps(rollout_batch)
            extend_rollout_batches.extend(expand_rollout_batch(rollout_batch))
        return extend_rollout_batches

    def get_num_rollout_micro_batches(self):
        """Compute the number of rollout micro-batches per DP rank.

        Returns
        -------
        int
        """
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        rollout_nb = training_config.rollout_gbs // (dp_size * training_config.rollout_mbs)
        assert rollout_nb > 0, f"rollout_nb {rollout_nb}"
        return rollout_nb

    def get_critic_model_warmup_step(self):
        if not self.require_critic_model():
            return 0
        return self.config.ppo.critic_model_warmup_steps

    async def train_one_ppo_step(
        self,
        epoch_i,
        ppo_step_i,
        cur_epoch_ppo_step,
    ):
        """Execute one full PPO step: rollout, train critic, train policy.

        Parameters
        ----------
        epoch_i : int
        ppo_step_i : int
            Global PPO step.
        cur_epoch_ppo_step : int
            PPO step within the current epoch.
        """
        timers = TimerSingleton.get_timer()
        begint_time = sync_cuda_and_get_time()
        training_config = self.config.training
        rollout_nb = self.get_num_rollout_micro_batches()
        assert rollout_nb > 0, f"rollout_nb {rollout_nb}"
        rollout_gbs = training_config.rollout_gbs
        rollout_mbs = training_config.rollout_mbs
        repeat_n = training_config.sampling_repeat_n
        keep_n = training_config.sampling_keep_n
        assert keep_n <= repeat_n, f"keep_n {keep_n} > repeat_n {repeat_n}"

        # rollout_NB = rollout_GBS / rollout_MBS / DP_size
        # len(rollout_batches) == rollout_NB
        # v = rollout_batches[0][k]
        # len(v) == rollout_MBS * keep_n
        rollout_batches, metrics = await self.rollout(epoch_i, ppo_step_i, rollout_nb)

        # post-filter: filter after logprobs/advantage computation
        filter_stage = (training_config.filter_sampling_stage if keep_n != repeat_n else None)
        if filter_stage == "post":
            samples_before = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            original_metrics = {k + "_original": v for k, v in metrics.items()}
            strategy = training_config.sampling_keeping_strategy
            filter_sampling_fn = FilterSamplingRegistry.get(strategy)
            rollout_batches = filter_sampling_fn(self.config, rollout_batches, repeat_n, keep_n)
            samples_after = len(rollout_batches[0][next(iter(rollout_batches[0]))])
            logging_rank0(
                f"[filter] stage=post applied, strategy={strategy}, "
                f"samples_per_batch: {samples_before} -> {samples_after}, "
                f"recomputing metrics on filtered set"
            )
            rollout_metrics = self.compute_rollout_metrics(rollout_batches)
            ppo_data_metrics = self.compute_ppo_global_statistics(rollout_batches)
            metrics = rollout_metrics | ppo_data_metrics | original_metrics

        if self.config.debug.save_every_rollout_data or (
            self.config.debug.save_first_rollout_data and ppo_step_i == 0
        ):
            save_data(
                rollout_batches, "debug-tmp",
                f"rollout_batches_{ppo_step_i}_{torch.distributed.get_rank()}.pt"
            )
            save_data(
                metrics, "debug-tmp",
                f"rollout_metrics_{ppo_step_i}_{torch.distributed.get_rank()}.pt"
            )

        assert len(rollout_batches) == rollout_nb, f'{len(rollout_batches)=} {rollout_nb=}'
        assert len(next(iter(rollout_batches[0].values()))) == rollout_mbs * keep_n

        expanded_rbs = expand_rollout_batches(rollout_batches)
        total_samples = training_config.rollout_gbs * keep_n
        assert len(expanded_rbs) * mpu.get_data_parallel_world_size(
        ) == total_samples, f"{len(expanded_rbs)} != {total_samples}"

        # ---- dp_balance for train: rebalance expanded samples across DP ranks ----
        if getattr(self.config.policy, 'balance_dp_seqlen', False):
            expanded_rbs = DPBalanceHelper.rebalance_row_batches_for_train(
                expanded_rbs,
                add_custom_keys=getattr(self.config.task, "add_custom_keys", None),
            )

        # ---- smart_pad_train: sort samples by seqlen for efficient padding ----
        if self.config.policy.smart_pad_train:
            num_global_batch = rollout_gbs * keep_n // training_config.train_gbs
            reorder_expanded_rbs = smart_pad_train_get_reorder_rollout_batches(
                expanded_rbs,
                num_global_batch,
                training_config.train_gbs // mpu.get_data_parallel_world_size(),
                training_config.pad_to_mulitiple_of,
                reorder_seed=ppo_step_i,
            )
            expanded_rbs = reorder_expanded_rbs

        timers("train_step", log_level=0).start(barrier=True)
        if self.require_critic_model():
            logging_rank0("train value model")
            with OnloadManager(self.critic_engine, True):
                for i in range(training_config.ppo_max_epochs_2):
                    num_train_global_steps = rollout_gbs * keep_n // training_config.train_gbs
                    ppo_step_iters = get_iterator_k_split_list(expanded_rbs, num_train_global_steps)
                    _metrics = self.critic_engine.rl_train_value(ppo_step_iters)
                    extend_value_to_dict(metrics, _metrics)
                lr = self.critic_engine.step_and_get_lr()
                metrics["value/lr"] = lr
            cpu_barrier()
            logging_rank0("train value model done")
            # onload policy model weight, policy model is offload before calculate values
            self.policy_engine.onload_model()

        self.policy_engine.onload_optimizer()
        logging_rank0("train policy model")

        should_dump = training_config.ppo_dump_metrics_interval > 0 and (
            ppo_step_i + 1
        ) % training_config.ppo_dump_metrics_interval == 0

        for epoch_idx in range(training_config.ppo_max_epochs_2):
            # only train if is not in critic warmup step
            if ppo_step_i >= self.get_critic_model_warmup_step():
                num_train_global_steps = rollout_gbs * keep_n // training_config.train_gbs
                ppo_step_iters = get_iterator_k_split_list(expanded_rbs, num_train_global_steps)
                self.policy_engine.should_dump_metrics = should_dump
                _metrics = self.policy_engine.rl_train_actor(ppo_step_iters)

                if should_dump:
                    self._save_dumped_metrics(_metrics, expanded_rbs, ppo_step_i, epoch_idx)

                extend_value_to_dict(metrics, _metrics)
        cpu_barrier()
        logging_rank0("train policy model done")
        timers("train_step").stop()

        if should_dump and torch.distributed.get_rank() == 0:
            self.compact_thread = compress_ppo_save_train_data(
                self.compact_thread, training_config.ppo_dump_metrics_dir
            )

        lr = self.policy_engine.step_and_get_lr()
        metrics["policy/lr"] = lr
        end_time = sync_cuda_and_get_time()
        metrics["time_perf/total_time"] = end_time - begint_time
        output_metrics = reduce_metrics(metrics)

        time_log_keys = [
            "rollout", "compute_logps", "generate_ppo_data", "train_step", "sampler_generate",
            "gen_rm_generate", "bt_rm_generate", "external_reward"
        ]
        output_metrics = record_time_to_metrics(timers, time_log_keys, output_metrics, reset=True)

        if is_last_rank():
            log_prefix = f"training ppo_step {ppo_step_i}/{training_config.total_ppo_step} epoch {epoch_i}"
            TrainReporterSingleton.log_and_report(output_metrics, ppo_step_i, log_prefix=log_prefix)
        del expanded_rbs, rollout_batches, metrics
        clear_memory()
        cpu_barrier()
        return output_metrics

    def maybe_set_epoch(self, epoch, reset_start_index=True):
        """Set epoch on the distributed sampler for proper shuffling.

        Parameters
        ----------
        epoch : int
        reset_start_index : bool
            Whether to reset the start_index of ResumableDistributedSampler.
            Set to False when resuming mid-epoch to preserve the skip offset.
        """
        if self.train_sampler:
            if isinstance(self.train_sampler, ResumableDistributedSampler):
                self.train_sampler.set_epoch(epoch)
                if reset_start_index:
                    self.train_sampler.start_index = 0
            elif isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
            else:
                raise ValueError(f"train_sampler {type(self.train_sampler)} is not supported")
        elif hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    async def _train_loop(self):
        """Main training loop: iterate over epochs and PPO steps."""
        training_config = self.config.training
        ppo_step = self.prev_ppo_step
        init_ppo_step = self.prev_ppo_step
        init_epoch = init_ppo_step // training_config.ppo_step_per_epoch
        init_ppo_step = init_ppo_step % training_config.ppo_step_per_epoch
        eval_before_train_flag = self.config.training.eval_before_train

        await self.update_weights()
        cpu_barrier()

        ret_metrics = []

        for epoch in range(init_epoch, training_config.num_train_epoches):
            if epoch == init_epoch and init_ppo_step > 0:
                reset_start_index = False
            else:
                reset_start_index = True
            self.maybe_set_epoch(epoch, reset_start_index)
            self.train_iter = iter(self.train_dataloader)

            if epoch == init_epoch:
                start_steps_per_epoch = init_ppo_step
            else:
                start_steps_per_epoch = 0
            for cur_epoch_ppo_step in range(
                start_steps_per_epoch, training_config.ppo_step_per_epoch
            ):
                self.last_progress_time = time.time()
                if eval_before_train_flag and self.config.training.total_eval_step > 0:
                    await self._eval_loop(ppo_step)
                    eval_before_train_flag = False

                ppo_step_metrics = await self.train_one_ppo_step(
                    epoch, ppo_step, cur_epoch_ppo_step
                )
                if self.config.debug.trainer_return_ppo_step_metrics:
                    ret_metrics.append(ppo_step_metrics)

                ppo_step += 1
                if ppo_step % training_config.save_interval == 0:
                    await self.save_checkpoint(ppo_step)

                await self.update_weights()

                if (
                    self.config.training.total_eval_step > 0 and
                    ppo_step % training_config.eval_interval == 0
                ):
                    await self._eval_loop(ppo_step)

                cpu_barrier()

        if self.compact_thread is not None:
            self.compact_thread.join()

        self.train_step_finished = True

        # save final ckpt
        cpu_barrier()
        if ppo_step % training_config.save_interval != 0:
            self.policy_engine.onload_model()
            self.policy_engine.onload_optimizer()
            await self.save_checkpoint(ppo_step)

        if is_last_rank():
            TrainReporterSingleton.finish()
        return ret_metrics

    async def train_loop(self):
        try:
            return await self._train_loop()
        except Exception as e:
            log(f"train_loop error: {e}")
            traceback.print_exc()
            raise e
