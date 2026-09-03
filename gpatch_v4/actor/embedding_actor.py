import asyncio
import time

import torch
from typing_extensions import override

from megatron.core import mpu, tensor_parallel

from gpatch_v4.actor.finetune_actor import FinetuneActor
from gpatch_v4.actor.mixin import TrainingPltMixin
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank
from gpatch_v4.training_backend.loss_factory import FinetuneLossInput
from gpatch_v4.training_backend.megatron_backend.megatron_utils import unwrap_model
from gpatch_v4.utils import (
    TimerSingleton,
    TrainReporterSingleton,
    log,
    record_time_to_metrics,
)
from gpatch_v4.utils.common_utils import import_fn_from_path
from gpatch_v4.utils.test_utils import save_data
from gpatch_v4.utils.training_utils import get_iterator_k_split_list

try:
    from megatron.core.gcore_utils import (
        clear_gathered_routing_info,  # only branch wxdev support
    )
except ImportError:
    clear_gathered_routing_info = None


class EmbeddingActor(FinetuneActor):
    """GradCache 版对比学习 SFT actor (wemm embedding)。

    复用 ``FinetuneActor`` 的数据管线 / 训练循环 / checkpoint, 只把「训练一步」
    从标准 ``finetune_step`` 替换为 ``McoreEngine.contrastive_gradcache_step``:
    用 GradCache (Gao et al., RepL4NLP 2021) 把编码器前向与对比 loss 头解耦,
    逐 mbs 前向/反向, 使激活显存与 ``train_gbs`` 解耦, 同时与单次前向的
    ``wemm_embedding_loss`` 梯度严格等价。

    仅支持 TP=PP=CP=1、``use_linear_ce=True``、``loss_func=custom``
    (指向 wemm_embedding_loss.py)。
    """

    train_log_tag = "EMB"

    async def init(self, config):
        await super().init(config)
        self.embedding_cache_backward_fn = None
        if self.config.training.use_gbs_embedding_in_loss:
            py_path = self.config.training.gradcache_loss_py_path
            py_name = self.config.training.gradcache_loss_py_name
            assert py_path and py_name, (
                "use_gbs_embedding_in_loss=True (GradCache 回放) 需要指定 "
                "gradcache_loss_py_path 和 gradcache_loss_py_name"
            )
            # GradCache 阶段2: 在 GBS 池(cache 叶子)上重算 loss 并 backward 的
            self.embedding_cache_backward_fn = import_fn_from_path(py_path, py_name)
            # 回放路径尚未在 pp>1 下验证: 需要把 gather_embs/阶段2 gate 到
            # last stage、加 barrier、并显式传递 mbs index, 先防呆。
            # assert self.config.policy.dist_config.pipeline_model_parallel_size == 1, (
            #     "use_gbs_embedding_in_loss=True (GradCache 回放) 目前仅支持 "
            #     "pipeline_model_parallel_size == 1"
            # )

    @override
    def auto_calc_train_step(self):
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()

        gas = training_config.train_gbs // (dp_size * training_config.train_mbs)
        assert gas > 0, f"gradient_accumulation_steps must be positive, got {gas}"
        # TODO: train_dataloader 允许用户自定义的话，len() 在 dp rank 之间会不会不 match，导致程序
        # hang，有待处理。
        if self.config.policy.model_arch in (
            MODEL_ARCH.WEMM3_5_EMBEDDING,
            MODEL_ARCH.WEMM3_5_MOE_EMBEDDING,
        ) and self.config.training.use_gbs_embedding_in_loss:
            train_step_per_epoch = len(self.train_dataloader)
        else:
            train_step_per_epoch = (len(self.train_dataloader) // gas)
        total_training_step = train_step_per_epoch * training_config.num_train_epoches

        self.config.training.total_training_step = total_training_step
        self.config.training.train_step_per_epoch = train_step_per_epoch
        self.config.training.gradient_accumulation_steps = gas
        log(
            f"train_dataset length: {len(self.train_dataloader)=} {len(self.train_dataset)=} "
            f"{dp_size=} {self.config.training.total_training_step=}"
        )

        if self.config.training.eval_interval > 0:
            assert self.eval_dataset is not None, f"enable eval:{self.config.training.eval_interval} should have eval_dataset and eval_dataloader"
            assert self.eval_dataloader is not None, f"enable eval:{self.config.training.eval_interval} should have eval_dataset and eval_dataloader"
            eval_step = (len(self.eval_dataloader) // gas)
            self.config.training.total_eval_step = eval_step
            assert eval_step > 0, f"{len(self.eval_dataloader)=} > {gas=}"

    @override
    def _training_plt_report(self, train_state: TrainingPltMixin.TrainState, data: dict):
        self.training_plt_report(
            "EmbeddingActor", self.config.policy.model_arch, TrainingPltMixin.TrainType.SFT,
            train_state, data
        )

    def extract_embs(self, model_output, batch):
        hidden_states = model_output['hidden_states']
        output_layer = model_output['output_layer']
        if output_layer is not None and getattr(output_layer, "sequence_parallel", False):
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False
            )
        elif output_layer is not None:
            tp_group = getattr(output_layer, "tp_group", None)
            if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
                assert not hidden_states.requires_grad, (
                    "tensor_model_parallel_size > 1 且 sequence_parallel=False 时, "
                    "hidden_states 在各 tp rank 间是复制而不是切分的, 但反向不会自动 "
                    "all-reduce d_hidden, 直接用会导致梯度错误。请开启 sequence_parallel "
                    "(通常和 tp>1 一起自动启用), 或改用其它方式处理。"
                )

        # [seqlen, batch, hidden] -> [batch, seqlen, hidden]
        last_hidden_state = hidden_states.transpose(0, 1).contiguous()
        if "eos_positions" in batch and batch["eos_positions"] is not None:
            eos_positions = batch["eos_positions"].to(device=last_hidden_state.device)
            assert last_hidden_state.size(0) == 1, (
                f"packed THD expects batch dim 1, got {last_hidden_state.shape}"
            )
            return last_hidden_state[0, eos_positions]

        attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
        eos_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_indices, eos_positions]

    def gather_embs(self, model_outputs):
        embs_l = []
        is_source_l = []
        for model_output in model_outputs:
            batch = model_output['batch']
            hidden_states = self.extract_embs(model_output, batch)
            embs_l.append(hidden_states)
            is_source_l.append(batch['is_source'])
        eos_embedding = torch.cat(embs_l, dim=0)
        is_source = torch.cat(is_source_l, dim=0)

        assert is_source.shape[0] == eos_embedding.shape[0]

        dp_group = mpu.get_data_parallel_group()
        dp_size = mpu.get_data_parallel_world_size()

        all_embedding = torch.empty(
            (dp_size, *eos_embedding.shape), dtype=eos_embedding.dtype, device=eos_embedding.device
        )
        torch.distributed.all_gather_into_tensor(all_embedding, eos_embedding, group=dp_group)
        all_is_source = torch.empty(
            (dp_size, *is_source.shape), dtype=is_source.dtype, device=is_source.device
        )
        torch.distributed.all_gather_into_tensor(all_is_source, is_source, group=dp_group)

        return all_embedding, all_is_source

    def grad_cache_step(self, model_outputs, num_microbatches, batches):
        """阶段2: GBS 池当叶子, 重算完整 loss 并 backward (GradCache 核心)。

        为什么这一步能修复塌缩 bug: 池内**每个 entry 都是图上的叶子**, backward 时
        每行 CE 分母的 ``∂CE_i/∂t_j`` 全部落到 cache 上 —— 跨样本/跨 rank/跨 mbs
        的斥力梯度一项不缺 (旧 v1 里池子是 detached 常量, 这些项全部丢失)。

        流程:
        1. 池子 detach 成叶子 cache; logit_scale 同样叶子化(必须 detach, 否则
           阶段2 backward 会绕过 Megatron 的 ÷gas/DDP/clip 直接污染参数梯度);
           clamp 仍作用于真实参数, 与 v0 的副作用一致;
        2. 按 mbs 循环, 用与阶段3完全相同的路由(CoSENT / MRL 对比+distill)在
           cache 切片上重算 loss, Σ 后一次 backward;
        3. ``reduce_scatter_tensor`` 取回本 rank slice 的梯度之和(每个 rank 只回放
           自己的条目), all_reduce 取回 logit_scale 梯度之和;
        4. 把回放所需数据写回每个 sample:
           - ``eos_embeddings_grad``: [gas*N, D], 本 rank 各 mbs 的 ∂(Σ_ranks L)/∂e
           - ``logit_scale_grad``:    标量
           - ``cache_metrics``:       阶段2 真实指标 (跨 mbs 平均)

        缩放约定: 阶段3 每个 mbs 返回 ⟨e, g⟩ (不额外乘系数), 由 Megatron 的 ÷gas
        和 DDP 的 ÷dp 自动凑出 (1/(dp*gas)) * Σ_{r,j} ∂L_{r,j}/∂θ —— 与 v0 的
        梯度语义严格一致。唯一的近似: cache 来自阶段1前向(dropout 掩码 A), 回放
        的 e 来自阶段3前向(dropout 掩码 B), 梯度在 A 处计算、施加在 B 处 —— 这是
        GradCache 论文的标准近似。
        """
        all_embeddings, all_is_source = self.gather_embs(model_outputs)
        dp_group = mpu.get_data_parallel_group()
        dp_rank = mpu.get_data_parallel_rank()

        unwrapped_model = unwrap_model(self.model_engine.model)[0]
        cache = all_embeddings.detach().clone().requires_grad_(True)
        ls_leaf = unwrapped_model.logit_scale.detach().clone().requires_grad_(True)
        flat_embeddings = cache.view(-1, cache.size(-1))
        flat_is_source = all_is_source.view(-1)
        metric_sums = None
        total_loss = 0.0
        n = all_embeddings.size(1) // num_microbatches
        for i in range(num_microbatches):
            eos_embeddings = cache[dp_rank, i * n:(i + 1) * n]
            is_source = all_is_source[dp_rank, i * n:(i + 1) * n]
            batch = model_outputs[i].pop('batch')
            batch['rank_embeddings'] = eos_embeddings
            batch['rank_is_source'] = is_source
            batch['all_embeddings'] = flat_embeddings
            batch['all_is_source'] = flat_is_source
            batch['logit_scale'] = ls_leaf

            loss_input = FinetuneLossInput(
                logits=model_outputs[i] if isinstance(model_outputs[i], torch.Tensor) else None,
                batch=batch,
                unwrapped_model=unwrapped_model,
                skip_cp_loss_reduce=self.model_engine.calc_per_token_loss,
                linear_ce_input=model_outputs[i] if isinstance(model_outputs[i], dict) else None,
                cp_group=batch.get("cp_group", None),
                cur_mbs_step=i,
                num_mbs=num_microbatches,
            )
            loss, metrics = self.embedding_cache_backward_fn(self.config, loss_input)
            total_loss += loss
            if metric_sums is None:
                metric_sums = {k: v.detach().float().clone() for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    metric_sums[k] += v.detach().float().clone()

        cache_metrics = {k: v / num_microbatches for k, v in metric_sums.items()}
        # ---- 3. backward + 梯度规约 ----
        # 只碰 cache/ls_leaf 两个叶子, 不触任何模块参数, 不会触发 DDP hook。
        total_loss.backward()
        my_grad = torch.empty_like(cache[dp_rank])  # [gas*N, D]
        torch.distributed.reduce_scatter_tensor(my_grad, cache.grad.contiguous(), group=dp_group)
        # 纯 CoSENT 路由时 logit_scale 不在 loss 图里 (λ 是 config 常量
        # cosent_logit_scale), grad 为 None —— 补零保持回放公式统一。
        g_ls = None
        if ls_leaf.grad is not None:
            g_ls = ls_leaf.grad.detach().clone()
            torch.distributed.all_reduce(g_ls, op=torch.distributed.ReduceOp.AVG, group=dp_group)
            g_ls = g_ls / num_microbatches

        data_iter = get_iterator_k_split_list(batches, num_microbatches)
        for i in range(num_microbatches):
            mbs_batches = next(data_iter)
            g_embs = my_grad[i * n:(i + 1) * n]
            for batch in mbs_batches:
                batch['embeddings_grad'] = g_embs
                batch['logit_scale_grad'] = g_ls
                batch['cache_metrics'] = cache_metrics

    async def _train_loop(self):
        self.setup_profile()
        timers = TimerSingleton.get_timer()
        training_config = self.config.training
        num_microbatches = training_config.gradient_accumulation_steps
        train_step = self.train_step
        init_step = self.train_step
        init_epoch = init_step // training_config.train_step_per_epoch
        init_step = init_step % training_config.train_step_per_epoch
        eval_before_train_flag = self.config.training.eval_before_train
        collected_metrics = []

        cpu_barrier()
        for epoch in range(init_epoch, training_config.num_train_epoches):
            if epoch == init_epoch and init_step > 0:
                reset_start_index = False
            else:
                reset_start_index = True
            self.maybe_set_epoch(epoch, reset_start_index)

            self.train_iter = iter(self.train_dataloader)
            if epoch == init_epoch:
                start_steps_per_epoch = init_step
            else:
                start_steps_per_epoch = 0
            for cur_epoch_train_step in range(
                start_steps_per_epoch, training_config.train_step_per_epoch
            ):
                await asyncio.sleep(0.01)
                timers("train_step_total", log_level=0).start(barrier=True)
                self.last_progress_time = time.time()

                if eval_before_train_flag and self.config.training.total_eval_step > 0:
                    self._eval_loop(train_step)
                    eval_before_train_flag = False

                timers("get_batched_data", log_level=0).start(barrier=True)
                batched_data = []
                while len(batched_data) < num_microbatches:
                    new_data = next(self.train_iter)
                    if isinstance(new_data, list):
                        batched_data.extend(new_data)
                    else:
                        batched_data.append(new_data)
                assert len(batched_data) == num_microbatches
                expanded_rbs = self.process_batched_data(batched_data)
                timers("get_batched_data").stop()
                if self.config.debug.save_every_rollout_data or (
                    self.config.debug.save_first_rollout_data and train_step == 0
                ):
                    save_data(
                        expanded_rbs, "debug-tmp",
                        f"{self.train_log_tag.lower()}_batches_{train_step}_{torch.distributed.get_rank()}.pt"
                    )

                should_dump = training_config.ppo_dump_metrics_interval > 0 and (
                    train_step + 1
                ) % training_config.ppo_dump_metrics_interval == 0
                self.model_engine.should_dump_metrics = should_dump

                timers("train_step", log_level=0).start(barrier=True)
                self.profile_start(train_step)
                if not training_config.skip_train_step:
                    if self.config.training.use_gbs_embedding_in_loss:
                        metrics_micro_batch = self.model_engine._embedding_forward_only(
                            expanded_rbs, num_microbatches, train_step
                        )
                        # GradCache 阶段2: GBS 池当叶子 backward, 产出阶段3
                        # 回放所需的 embedding 梯度 / logit_scale 梯度 / 指标。
                        if mpu.is_pipeline_last_stage():
                            self.grad_cache_step(
                                metrics_micro_batch, num_microbatches, expanded_rbs
                            )

                    metric = self.model_engine.finetune_step(
                        expanded_rbs, num_microbatches, train_step
                    )
                self.profile_end(train_step)
                timers("train_step").stop()
                self._training_plt_report(
                    TrainingPltMixin.TrainState.TRAIN_STEP, dict(step=train_step)
                )

                if should_dump:
                    self._save_dumped_metrics(metric, expanded_rbs, train_step)

                if clear_gathered_routing_info is not None:
                    clear_gathered_routing_info()

                timers("train_step_total").stop()
                time_log_keys = ["get_batched_data", "train_step", "train_step_total"]
                metric = record_time_to_metrics(timers, time_log_keys, metric, reset=True)

                mfu, avg_mfu = self.flops_counter_calc(
                    train_step,
                    expanded_rbs,
                    metric['time_perf/train_step'],
                    metric['finetune/seq_length'],
                    seqlen_sum=metric.get('finetune/dyn_cp_seqlen_sum'),
                    seqlen_sq_sum=metric.get('finetune/dyn_cp_seqlen_sq_sum'),
                )
                if mfu is not None:
                    metric['finetune/mfu'] = mfu
                    metric['finetune/avg_mfu'] = avg_mfu

                if is_last_rank():
                    log_prefix = f"[{self.train_log_tag}] training train_step {train_step}/{training_config.total_training_step} epoch {epoch}"
                    TrainReporterSingleton.log_and_report(metric, train_step, log_prefix=log_prefix)
                if self.config.debug.trainer_return_ppo_step_metrics:
                    collected_metrics.append(metric)
                cpu_barrier()

                if self.config.training.total_eval_step is not None and self.config.training.total_eval_step > 0 and (
                    train_step + 1
                ) % training_config.eval_interval == 0:
                    self._eval_loop(train_step)

                train_step += 1
                if train_step % training_config.save_interval == 0 and not self.config.debug.disable_save_checkpoint:
                    self.model_engine.save_checkpoint(train_step, dataloader=self.train_dataloader)

                if train_step == training_config.exit_step:
                    break

            if train_step == training_config.exit_step:
                break

        if self.compact_thread is not None:
            self.compact_thread.join()

        self.train_step_finished = True

        cpu_barrier()
        if not self.config.debug.disable_save_checkpoint:
            if train_step % training_config.save_interval != 0:
                self.model_engine.save_checkpoint(train_step, dataloader=self.train_dataloader)

        if is_last_rank():
            TrainReporterSingleton.finish()

        return collected_metrics
