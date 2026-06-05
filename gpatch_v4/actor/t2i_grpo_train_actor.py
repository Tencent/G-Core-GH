import asyncio
import inspect
import os
import sys
import time
from typing import Any, Dict, List

import torch
import torch.distributed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from megatron.core import mpu

from gpatch_v4.actor.mixin import MetricsMixin, ProfileMixin, RetryActorMixin
from gpatch_v4.client.bt_rm_client_t2i import T2iBtRmClient
from gpatch_v4.client.gen_rm_client_t2i import T2iGenRmClient
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.extended_pipeline import ExtendPipelineFactory
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.utils import (
    BroadcastUtils,
    TrainReporterSingleton,
    check_rollout_batches,
    clear_memory,
    expand_rollout_batch,
    extend_value_to_dict,
    get_iterator_k_split_list,
    import_fn_from_path,
    init_train_reporter_singleton,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
    profile_memory_and_time,
    reduce_metrics,
    repeat_interleave_tensor_or_list,
    save_images,
    sync_cuda_and_get_time,
    unbind_tensor_to_list,
)


class T2iGrpoTrainActor(BaseActor, MetricsMixin, RetryActorMixin, ProfileMixin):
    """Ray actor implementing the T2I GRPO training workflow.

    Handles dataset construction, rollout generation, reward
    computation, advantage calculation, and PPO-style training
    for text-to-image diffusion models.
    """
    async def init(self, config):
        """Initialize the actor: parallel state, dataset, pipeline.

        Parameters
        ----------
        config : T2iRlConfig
        """
        super().init(config)
        self.retry_actor_init()

        if self.config.training.allow_tf32:
            # 强行对齐
            torch.backends.cuda.matmul.allow_tf32 = True

        # init parallel group
        initlize_parallel_state(config, config.policy.dist_config)
        init_pg(config.policy.dist_config)

        self.build_dataset_and_dataloader()

        self.validated_config()
        log(f"TrainActor config {config}", rank=torch.distributed.get_world_size() - 1)

        self.extended_pipeline = ExtendPipelineFactory.get_pipeline(config)
        self.prev_ppo_step = self.extended_pipeline.setup_pipeline()
        self.sample_idx = 0

        if is_last_rank():
            init_train_reporter_singleton(self.config.report, self.config)

    def validated_config(self):
        """Validate training configuration invariants.

        Raises
        ------
        AssertionError
            If any configuration constraint is violated.
        """
        training_config = self.config.training
        assert training_config.train_gbs is None
        assert training_config.gradient_accumulation_steps is None
        assert (training_config.sampling_steps - 1) * (training_config.timestep_fraction) >= 1
        assert training_config.rollout_gbs * training_config.sampling_repeat_n >= training_config.train_gbs_wo_timestep
        assert training_config.rollout_mbs * mpu.get_data_parallel_world_size(
        ) <= training_config.rollout_gbs
        assert training_config.train_mbs * mpu.get_data_parallel_world_size(
        ) <= training_config.train_gbs_wo_timestep
        training_config.disable_cfg_uncond_grad = training_config.enable_cfg and training_config.disable_cfg_uncond_grad

    async def setup_client(self):
        """Set up generative and BT reward model RPC clients."""
        self.gen_rm_client = None
        self.bt_rm_client = None
        if self.config.training.use_gen_rm_reward:
            self.gen_rm_client = T2iGenRmClient(self.config)
        if self.config.training.use_bt_rm_reward:
            self.bt_rm_client = T2iBtRmClient(self.config)

    def build_dataset_and_dataloader(self):
        """Build the training dataset, sampler, and dataloader.

        Dynamically imports the dataset builder from the configured Python
        path and calls it with ``(config, dp_rank, dp_size)`` to obtain the
        dataset, sampler, and dataloader.

        Examples
        --------
        .. code-block:: python

            import os
            import time

            from torch.utils.data import Dataset
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler

            class SimpleTextPromptDataset(Dataset):
                def __init__(self, dataset_file):
                    self.file_path = dataset_file
                    with open(self.file_path, 'r') as f:
                        self.prompts = [line.strip() for line in f.readlines()]

                def __len__(self):
                    return len(self.prompts)

                def __getitem__(self, idx):
                    return {"prompt": self.prompts[idx], "metadata": {}}


            def collate_fn(examples):
                prompts = [example["prompt"] for example in examples]
                metadatas = [example["metadata"] for example in examples]
                return prompts, metadatas


            def get_dataset_and_dataloader(config=None, dp_rank=0, dp_size=1):
                dataset = SimpleTextPromptDataset(config.data.data_pathes[0])
                sampler = DistributedSampler(
                    dataset, rank=dp_rank, num_replicas=dp_size,
                    shuffle=True, seed=config.data.sampler_seed,
                )
                dataloader = DataLoader(
                    dataset,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    batch_size=config.training.rollout_mbs,
                    num_workers=config.data.dataloader_num_workers,
                    drop_last=True,
                )
                return {
                    'train_dataset': dataset,
                    'train_sampler': sampler,
                    'train_dataloader': dataloader,
                }
        """

        fn = import_fn_from_path(self.config.data.py_path, self.config.data.fn_name)
        fn_kwargs = inspect.signature(fn).parameters

        cond1 = all(
            [
                len(fn_kwargs) == 3,
                'config' in fn_kwargs,
                'dp_rank' in fn_kwargs,
                'dp_size' in fn_kwargs,
            ]
        )
        if cond1:
            fn_ret = fn(
                config=self.config,
                dp_rank=mpu.get_data_parallel_rank(),
                dp_size=mpu.get_data_parallel_world_size(),
            )
        else:
            raise ValueError(f'unexpected signature {fn_kwargs}')

        self.train_dataset = fn_ret.get('train_dataset')
        self.sampler = fn_ret.get('train_sampler')
        self.train_dataloader = fn_ret.get('train_dataloader')
        self.train_sampler = fn_ret.get('train_sampler', None)

        self.train_iter = None
        self.auto_calc_ppo_step()

    def auto_calc_ppo_step(self):
        """Auto-calculate total PPO steps and gradient accumulation steps."""
        training_config = self.config.training
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()

        assert training_config.rollout_gbs * training_config.sampling_repeat_n >= training_config.train_gbs_wo_timestep
        train_gas_wo_timestep = training_config.train_gbs_wo_timestep // (
            dp_size * training_config.train_mbs
        )
        rollout_gas = training_config.rollout_gbs // (dp_size * training_config.rollout_mbs)
        # TODO: train_dataloader 允许用户自定义的话，len() 在 dp rank 之间会不会不 match，导致程序
        # hang，有待处理。
        ppo_step_per_epoch = (len(self.train_dataloader) // rollout_gas)
        total_ppo_step = ppo_step_per_epoch * training_config.num_train_epoches

        self.config.training.total_ppo_step = total_ppo_step
        self.config.training.ppo_step_per_epoch = ppo_step_per_epoch
        self.config.training.train_gas_wo_timestep = train_gas_wo_timestep
        log(
            f"train_dataset length: {len(self.train_dataloader)=} {len(self.train_dataset)=} "
            f"{dp_size=} {self.config.training.total_ppo_step=}"
        )

    def _calc_advantage(self, rewards: List[torch.Tensor], repeat_n: int, adv_eps: float):
        """Compute group-relative advantages from reward scores.

        Parameters
        ----------
        rewards : list of torch.Tensor
        repeat_n : int
        adv_eps : float
            Epsilon for numerical stability in std normalization.

        Returns
        -------
        torch.Tensor
            Normalized advantages.
        """
        scores = torch.stack(rewards).view(-1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = scores.view(-1, repeat_n).mean(dim=1)
        std_grouped_rewards = scores.view(-1, repeat_n).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(repeat_n, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(repeat_n, dim=0)
        advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + adv_eps)
        return advantages

    @torch.no_grad()
    def calculate_advantage(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
    ):
        """Calculate advantages from reward model outputs.

        Parameters
        ----------
        rollout_batches : list of dict
            Rollout batches containing reward keys.

        Returns
        -------
        list of dict
            Input batches augmented with ``'advantages'`` key.
        """
        gen_rm_info = self.config.gen_rm.reward_model_info
        bt_rm_info = self.config.bt_rm.reward_model_info
        num_gen_rm = len(gen_rm_info) if self.config.training.use_gen_rm_reward else 0
        num_bt_rm = len(bt_rm_info) if self.config.training.use_bt_rm_reward else 0

        rollout_mbs = self.config.training.rollout_mbs
        repeat_n = self.config.training.sampling_repeat_n
        ppo_config = self.config.ppo

        for rb in rollout_batches:
            total_advantages = []

            def process_one_reward(rewards, reward_weight):
                nonlocal total_advantages
                if isinstance(rewards, dict):
                    # support multiple reward from one reward actor
                    for (k, v) in rewards.items():
                        process_one_reward(v["rewards"], v["weight"] * reward_weight)
                else:
                    advantages = self._calc_advantage(
                        rewards, repeat_n, ppo_config.grpo_advantage_epsilon
                    )
                    total_advantages.append(advantages * reward_weight)

            for gen_rmi in range(num_gen_rm):
                rm_key = f"reward_gen_rm_{gen_rmi}"
                assert rm_key in rb.keys(), f'{rm_key} not in {rb.keys()}'
                rewards = rb[rm_key]
                process_one_reward(rewards, gen_rm_info[gen_rmi].reward_weight)

            for bt_rmi in range(num_bt_rm):
                rm_key = f"reward_bt_rm_{bt_rmi}"
                assert rm_key in rb.keys(), f'{rm_key} not in {rb.keys()}'
                rewards = rb[rm_key]
                process_one_reward(rewards, bt_rm_info[bt_rmi].reward_weight)

            # TODO: 计算 bt rm reward 算出来的 advantage num bt rm
            advantages = sum(v for v in total_advantages).view(-1, 1)
            rb["advantages"] = unbind_tensor_to_list(advantages)
            assert rollout_mbs * repeat_n == len(
                rb["advantages"]
            ), f"{repeat_n=} {len(rb['advantages'])=}"

        return rollout_batches

    async def get_reward_fn(
        self, rbs: List[Dict[str, List[Any]]], rm_client, ppo_step_i: int, sample_idx: int,
        rm_idx: int
    ):
        """Issue reward-computation requests to a reward model client.

        Parameters
        ----------
        rbs : list of dict
            Rollout batches to score.
        rm_client : object
        ppo_step_i : int
        sample_idx : int
        rm_idx : int

        Returns
        -------
        list
            Gathered reward response dicts.
        """
        cos = []
        for rbi, rollout_batch in enumerate(rbs):
            co = rm_client.generate_rewards(rm_idx, ppo_step_i, sample_idx + rbi, rollout_batch)
            cos.append(co)
        return await asyncio.gather(*cos)

    @torch.no_grad()
    async def generate_gen_rm_reward(
        self, rollout_batches: List[Dict[str, List[Any]]],
        reward_req_batches: List[Dict[str, List[Any]]], ppo_step_i: int
    ):
        """Compute generative reward model scores for rollout batches.

        Parameters
        ----------
        rollout_batches : list of dict
            Rollout results to be augmented with rewards.
        reward_req_batches : list of dict
            Request payloads for the reward model.
        ppo_step_i : int

        Returns
        -------
        list of dict
            Updated rollout batches with gen-RM rewards.
        """
        for rm_idx in range(self.gen_rm_client.num_rms):
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking before gen_rm {rm_idx} ppo step begin", rank=0
                )
            await self.gen_rm_client.mark_ppo_step_begin(rm_idx, ppo_step=ppo_step_i)
            cpu_barrier()

            if is_mp_and_cp_head():
                gen_rm_resp_dicts = await self.get_reward_fn(
                    reward_req_batches, self.gen_rm_client, ppo_step_i, self.sample_idx, rm_idx
                )
                for rbi, (rollout_batch, gen_rm_resp_dict) in enumerate(
                    zip(rollout_batches, gen_rm_resp_dicts)
                ):
                    rollout_batch.update(gen_rm_resp_dict)

            cpu_barrier()
            await self.gen_rm_client.mark_ppo_step_end(rm_idx, ppo_step=ppo_step_i)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking after gen_rm {rm_idx} ppo step end", rank=0
                )

        return rollout_batches

    @torch.no_grad()
    async def calc_bt_rm_reward(
        self, rollout_batches: List[Dict[str, List[Any]]],
        reward_req_batches: List[Dict[str, List[Any]]], ppo_step_i: int
    ):
        """Compute BT reward model scores for rollout batches.

        Parameters
        ----------
        rollout_batches : list of dict
            Rollout results to be augmented with rewards.
        reward_req_batches : list of dict
            Request payloads for the reward model.
        ppo_step_i : int

        Returns
        -------
        list of dict
            Updated rollout batches with BT-RM rewards.
        """
        for rm_idx in range(self.bt_rm_client.num_rms):
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking before bt_rm {rm_idx} ppo step begin", rank=0
                )
            await self.bt_rm_client.mark_ppo_step_begin(rm_idx, ppo_step=ppo_step_i)
            cpu_barrier()

            if is_mp_and_cp_head():
                gen_rm_resp_dicts = await self.get_reward_fn(
                    reward_req_batches, self.bt_rm_client, ppo_step_i, self.sample_idx, rm_idx
                )
                for rbi, (rollout_batch, gen_rm_resp_dict) in enumerate(
                    zip(rollout_batches, gen_rm_resp_dicts)
                ):
                    rollout_batch.update(gen_rm_resp_dict)

            cpu_barrier()
            await self.bt_rm_client.mark_ppo_step_end(rm_idx, ppo_step=ppo_step_i)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details(
                    f"memory tracking after bt_rm {rm_idx} ppo step end", rank=0
                )

        return rollout_batches

    def create_same_rng_for_same_prompt(self, mbs, n_repeat, seed, device='cpu'):
        """Create reproducible RNG generators grouped by prompt.

        Ensures that all repeated samples for the same prompt share
        the same random seed.

        Parameters
        ----------
        mbs : int
            Micro batch size (number of unique prompts).
        n_repeat : int
        seed : int
        device : str, optional

        Returns
        -------
        list of torch.Generator
            ``mbs * n_repeat`` generators.
        """
        rngs = []
        for i in range(mbs):
            for j in range(n_repeat):
                _seed = (seed + i) % (2**31)
                rng = torch.Generator(device=device).manual_seed(_seed)
                rngs.append(rng)
        return rngs

    async def rollout_remotable(self, *args, **kwargs):
        """Remote-callable wrapper around :meth:`rollout`."""
        return await self.rollout(*args, **kwargs)

    @torch.no_grad()
    async def rollout(
        self, epoch_i, ppo_step_i, num_rollout_micro_batches, debug_disable_advantage=False
    ):
        """Generate rollout samples and compute rewards.

        Parameters
        ----------
        epoch_i : int
        ppo_step_i : int
        num_rollout_micro_batches : int
        debug_disable_advantage : bool, optional
            Skip advantage calculation (debug), by default *False*.

        Returns
        -------
        tuple[list[dict], dict or None, dict]
            ``(rollout_batches, metrics, elapsed_times)``.
        """
        repeat_n = self.config.training.sampling_repeat_n
        rollout_mbs = self.config.training.rollout_mbs
        dp_rank = mpu.get_data_parallel_rank()
        rollout_batches = []
        reward_req_batches = []
        log(
            f"begin rollout epoch {epoch_i} ppo_step {ppo_step_i}",
            rank=torch.distributed.get_world_size() - 1
        )

        # TODO: 如果后面 encoder 和 dit model 无法放一起，再把下面的 encode_prompt 一次性全做了，
        with profile_memory_and_time(
            f"prepare rollout", rank=torch.distributed.get_world_size() - 1
        ):
            self.extended_pipeline.model.offload_optimizer()
            self.extended_pipeline.onload_encoder()
            self.extended_pipeline.model.onload_model()

        for rbi in range(num_rollout_micro_batches):
            log(f"rollout micro batch {rbi}", rank=torch.distributed.get_world_size() - 1)
            batched_data = next(self.train_iter)
            assert isinstance(batched_data, dict)

            # encoded_text_cond is a dict of (prompt_embeds, pooled_prompt_embeds, text_ids, ...)
            encoded_text_cond = self.extended_pipeline.encode_prompt(**batched_data)
            self.extended_pipeline.repeat_interleave_tensor_or_list(batched_data, repeat_n)
            self.extended_pipeline.repeat_interleave_tensor_or_list(encoded_text_cond, repeat_n)

            # call pipeline
            if self.config.training.init_same_noise:
                base_seed = self.config.training.seed + dp_rank * 100 + epoch_i * 100 + ppo_step_i * num_rollout_micro_batches + rbi
                rngs = self.create_same_rng_for_same_prompt(
                    rollout_mbs, repeat_n, base_seed, device=torch.cuda.current_device()
                )
            else:
                rngs = None

            # 如果出现与 modeling 不相关，但是 gen-RM 相关的字段，在数据集返回值增加一个返回值做标记即可。这样比较
            # 干净。目前 rm_req_rb 是只会返回 prompt 与 images 两个字段。
            rb, rm_req_rb = self.extended_pipeline(
                **batched_data,
                **encoded_text_cond,
                guidance_scale=self.config.training.guidance_scale,
                generator=rngs,
            )

            if self.config.debug.save_images:
                # by default self.config.debug.images_attr_name = 'images'
                save_images(
                    rb[self.config.debug.images_attr_name], rbi, self.config.debug.save_images_dir
                )

            reward_req_batches.append(rm_req_rb)
            rollout_batches.append(rb)

        with profile_memory_and_time(f"after rollout", rank=torch.distributed.get_world_size() - 1):
            self.extended_pipeline.model.offload_model()
            self.extended_pipeline.offload_encoder()

        gen_rm_elapsed = 0
        if self.config.training.use_gen_rm_reward:
            t1 = sync_cuda_and_get_time()
            rollout_batches = await self.generate_gen_rm_reward(
                rollout_batches, reward_req_batches, ppo_step_i
            )
            t2 = sync_cuda_and_get_time()
            gen_rm_elapsed = t2 - t1

        bt_rm_elapsed = 0
        if self.config.training.use_bt_rm_reward:
            t1 = sync_cuda_and_get_time()
            rollout_batches = await self.calc_bt_rm_reward(
                rollout_batches, reward_req_batches, ppo_step_i
            )
            t2 = sync_cuda_and_get_time()
            bt_rm_elapsed = t2 - t1

        if is_mp_and_cp_head():
            assert check_rollout_batches(rollout_batches), f"Rollout batch check failed"

        # bcast between mp and cp group
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)
        clear_memory()

        if not debug_disable_advantage:
            rollout_batches = self.calculate_advantage(rollout_batches)
        self.sample_idx += num_rollout_micro_batches

        # summary metrics
        metrics = None
        if not debug_disable_advantage:
            metrics = self._summary_reward_and_advantage_metrics(rollout_batches)

        rollout_batches = self.remove_raw_rewards(rollout_batches)

        elapsed = {
            'gen_rm_elapsed': gen_rm_elapsed,
            'bt_rm_elapsed': bt_rm_elapsed,
        }
        return rollout_batches, metrics, elapsed

    def permute_and_expand_rollout_batches(self, rollout_batches: List[Dict[str, List[Any]]]):
        """Permute timesteps and expand rollout batches for training.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        list of dict
            Expanded individual training samples.
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
        """Calculate the number of rollout micro-batches per PPO step.

        Returns
        -------
        int
        """
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        rollout_nb = training_config.rollout_gbs // (dp_size * training_config.rollout_mbs)
        assert rollout_nb > 0, f"rollout_nb {rollout_nb}"
        return rollout_nb

    async def train_one_ppo_step(
        self,
        epoch_i,
        ppo_step_i,
        cur_epoch_ppo_step,
    ):
        """Execute one full PPO step: rollout, reward, train.

        Parameters
        ----------
        epoch_i : int
        ppo_step_i : int
        cur_epoch_ppo_step : int

        Returns
        -------
        dict
            Aggregated training metrics.
        """
        training_config = self.config.training
        rollout_nb = self.get_num_rollout_micro_batches()
        assert rollout_nb > 0, f"rollout_nb {rollout_nb}"
        rollout_gbs = self.config.training.rollout_gbs
        rollout_mbs = self.config.training.rollout_mbs
        repeat_n = self.config.training.sampling_repeat_n

        # shape 解释：
        # rollout_NB = rollout_GBS / rollout_MBS / DP_size
        # len(rollout_batches) == rollout_NB
        # v = rollout_batches[0][k]
        # len(v) == rollout_MBS * repeat_n
        rollout_begin_t = sync_cuda_and_get_time()
        rollout_batches, metrics, elapsed = await self.rollout(epoch_i, ppo_step_i, rollout_nb)
        rollout_end_t = sync_cuda_and_get_time()

        assert len(rollout_batches) == rollout_nb, f'{len(rollout_batches)=} {rollout_nb=}'
        assert len(next(iter(rollout_batches[0].values()))) == rollout_mbs * repeat_n

        # shape 解释：
        # len(expanded_rbs) == rollout_NB * rollout_MBS * repeat_n
        expanded_rbs = self.permute_and_expand_rollout_batches(rollout_batches)
        assert len(expanded_rbs) == rollout_nb * rollout_mbs * repeat_n
        postprocess_end_t = sync_cuda_and_get_time()

        self.extended_pipeline.model.onload_model()
        self.extended_pipeline.model.onload_optimizer()

        log(
            f"begin train_step epoch {epoch_i} ppo_step {ppo_step_i}",
            rank=torch.distributed.get_world_size() - 1
        )
        for e2_i in range(training_config.ppo_max_epochs_2):
            num_train_global_steps = rollout_gbs * repeat_n // training_config.train_gbs_wo_timestep
            ppo_step_iters = get_iterator_k_split_list(expanded_rbs, num_train_global_steps)
            for _, rollout_batch in enumerate(ppo_step_iters):
                _metrics = self.extended_pipeline.ppo_train_step(rollout_batch)
                extend_value_to_dict(metrics, _metrics)

        train_end_t = sync_cuda_and_get_time()

        self.extended_pipeline.model.lr_scheduler.step()
        lr = self.extended_pipeline.model.lr_scheduler.get_last_lr()[0]
        metrics["policy/lr"] = lr

        output_metrics = reduce_metrics(metrics)
        output_metrics["time/rollout_time"] = rollout_end_t - rollout_begin_t
        output_metrics["time/postprocess_time"] = postprocess_end_t - rollout_end_t
        output_metrics["time/train_time"] = train_end_t - postprocess_end_t
        output_metrics["time/gen_rm_elapsed"] = elapsed['gen_rm_elapsed']
        output_metrics["time/bt_rm_elapsed"] = elapsed['bt_rm_elapsed']

        if is_last_rank():
            log_prefix = f"training ppo_step {ppo_step_i}/{training_config.total_ppo_step} epoch {epoch_i}"
            TrainReporterSingleton.log_and_report(output_metrics, ppo_step_i, log_prefix=log_prefix)

        return output_metrics

    def maybe_set_epoch(self, epoch):
        """Set the epoch on the distributed sampler if present.

        Parameters
        ----------
        epoch : int
        """
        if self.train_sampler:
            assert isinstance(self.train_sampler, DistributedSampler)
            self.train_sampler.set_epoch(epoch)
        elif hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)
        self.train_iter = iter(self.train_dataloader)

    async def train_loop(self):
        """Main training loop over epochs and PPO steps.

        Returns
        -------
        list[dict]
            Per-step metrics if debug collection is enabled.
        """
        training_config = self.config.training
        ppo_step = self.prev_ppo_step
        init_ppo_step = self.prev_ppo_step
        init_epoch = init_ppo_step // training_config.ppo_step_per_epoch
        init_ppo_step = init_ppo_step % training_config.ppo_step_per_epoch
        metrics = []

        cpu_barrier()
        for epoch in range(init_epoch, training_config.num_train_epoches):
            self.maybe_set_epoch(epoch)

            if epoch == init_epoch:
                start_steps_per_epoch = init_ppo_step
            else:
                start_steps_per_epoch = 0
            for cur_epoch_ppo_step in range(
                start_steps_per_epoch, training_config.ppo_step_per_epoch
            ):
                self.last_progress_time = time.time()
                if ppo_step + 1 == training_config.exit_step:  # ppo_step starts from 0, exit_step starts from 1
                    break
                if not training_config.skip_train_step:
                    ppo_step_metrics = await self.train_one_ppo_step(
                        epoch, ppo_step, cur_epoch_ppo_step
                    )
                    if self.config.debug.trainer_return_ppo_step_metrics:
                        metrics.append(ppo_step_metrics)

                if (ppo_step + 1) % training_config.save_interval == 0:
                    self.extended_pipeline.model.save(ppo_step + 1)
                ppo_step += 1
        self.train_step_finished = True

        # save final ckpt
        cpu_barrier()
        self.extended_pipeline.model.save(ppo_step)

        if is_last_rank():
            TrainReporterSingleton.finish()
        return metrics

    def get_prev_ppo_step(self):
        """Return the PPO step from which training will resume.

        Returns
        -------
        int
        """
        return self.prev_ppo_step
