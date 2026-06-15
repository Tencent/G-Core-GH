import asyncio
import time
import traceback

from typing_extensions import override

from gpatch_v4 import orches
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.orches.placement_group import (
    create_bt_rm_group,
    create_placement_groups,
    create_rollout_controller,
    create_sampler_group,
    create_train_group,
)
from gpatch_v4.trainer.grpo_trainer import GrpoTrainer
from gpatch_v4.trainer.helper import set_nnodes_default
from gpatch_v4.utils import log


class GrpoSingleCtrlTrainer(GrpoTrainer):
    """Single-controller rollout GRPO trainer.

    Inherits ``GrpoTrainer`` to reuse ``launch()`` (sampler/RM/train group
    creation and init). Replaces the training loop with a centralized
    ``RolloutController`` that manages data reading, sampler generation,
    and RM reward scoring. GPU actors only handle logps, PPO data, and
    training.

    Disaggregated async rollout runs a sliding streaming pipeline: prime
    with ``s + 1`` PPO steps of rollout (``s = rollout_max_staleness``),
    then train ``s`` steps per sampler-weight window. Colocate
    single-controller training uses the same fire/collect loop with
    ``s = 0`` and no overlapping inflight steps.
    """
    _supports_async_rollout = True

    def __init__(self):
        super().__init__()
        # Pipeline timeline: {ppo_step: {"fire": t, "collect": t, "train_done": t,
        # "update_start": t?, "update_done": t?}}.  ``update_*`` are only set on
        # the last trained step of each sampler-weight window.  All timestamps
        # are recorded in the driver process so they are comparable.
        self._pipeline_ts = {}

    def get_pipeline_stats(self):
        """Return pipeline timeline stats for overlap verification.

        Returns
        -------
        dict[int, dict[str, float]]
        """
        return dict(self._pipeline_ts)

    @override
    async def debug_update_weight(self, config: RlConfig):
        """Debug helper for rollout-controller weight updates."""

        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.sampler_group = create_sampler_group(config, pgs)
        await self.sampler_group.init()

        if config.training.use_gen_rm_reward:
            await self._init_gen_rm_groups(config, pgs)

        if config.training.use_bt_rm_reward:
            self.bt_rm_group = create_bt_rm_group(config, pgs)
            await self.bt_rm_group.init()

        self.train_group = create_train_group(config, pgs)
        await self.train_group.init()

        await self.train_group.setup_client()
        await self.train_group.setup_rollout_generator()

        await self.train_group.debug_update_weight(stage=1)
        await self.train_group.setup_model_and_optimizer()

        await self.train_group.debug_update_weight(stage=2)

    @override
    async def launch_then_run_with_recovery(self, config: RlConfig):
        """Launch async rollout training with centralized controller.

        Parameters
        ----------
        config : RlConfig
        """
        await self.launch(config)

        # Create the RolloutController (0-GPU Ray actor)
        self.rollout_controller = create_rollout_controller(config)
        train_actors = self.train_group._actor_handlers if config.placement_type == "colocate" else None
        await self.rollout_controller.setup.remote(train_actors=train_actors)

        return await self.run_train_loop(config)

    async def _run_train_loop(self, config: RlConfig):
        """Controller-side rollout/training loop.

        Parameters
        ----------
        config : RlConfig
            Must have ``training.async_rollout=True``.
        """
        tg = self.train_group
        rc = self.rollout_controller
        training_cfg = config.training
        is_colocate = config.placement_type == "colocate"
        is_disaggregated = config.placement_type == "disaggregated"
        assert is_colocate or is_disaggregated, (
            f"unsupported placement_type={config.placement_type}"
        )
        assert training_cfg.single_controller, (
            "single-controller trainer requires training.single_controller=True"
        )
        if is_colocate:
            assert not training_cfg.async_rollout, (
                "colocate requires training.async_rollout=False"
            )
            assert training_cfg.rollout_max_staleness == 0, (
                "colocate requires rollout_max_staleness == 0"
            )
        else:
            assert training_cfg.async_rollout, (
                "disaggregated requires training.async_rollout=True"
            )
        max_stale = training_cfg.rollout_max_staleness
        # Window size: s steps trained per sampler-weight window (or 1 step
        # when s=0, giving fully synchronous behavior).
        window_size = max(max_stale, 1)

        train_info = await tg.init_train_state()
        total_ppo_step = train_info["total_ppo_step"]
        ppo_step_per_epoch = train_info["ppo_step_per_epoch"]
        prev_ppo_step = train_info["prev_ppo_step"]

        train_step = prev_ppo_step
        fired_step = prev_ppo_step
        steps_since_update = 0
        # DEBUG: short-circuit driver-side rollout entirely.
        debug_skip_rollout = config.debug.skip_rollout_load_from_disk

        # First update uses offload=True so model/optimizer go through one
        # offload cycle, initializing swap bookkeeping (e.g.
        # gcore_untyped_storage_data_size on grad buffers).
        # compute_log_probs will onload_model back; after that everything
        # stays on GPU for the rest of training.
        await tg.update_weights(offload=True, flush_cache=True)
        if is_colocate:
            await tg.log_memory("memory tracking after initial update_weights")

        def epoch_for(step: int) -> int:
            return step // ppo_step_per_epoch

        async def fire_up_to(target: int) -> None:
            """Fire PPO steps until ``fired_step == target``.

            Clips ``target`` at ``total_ppo_step`` and at each epoch
            boundary, emitting one ``fire_generation_requests`` call per
            epoch.  Splitting on the epoch boundary is required because
            a single call pins an epoch via ``maybe_set_epoch`` and then
            reads ``num_mb * n`` batches sequentially from the dataloader;
            spanning epoch boundaries would read past the end of the
            current epoch's iterator.

            All PPO-step indices in a single fire call share the same
            ``fire`` timestamp (same wall-clock instant the batch was
            dispatched), matching the original loop's semantics.

            When ``debug.skip_rollout_load_from_disk`` is enabled, the
            sampler dispatch is skipped entirely (actors will
            ``torch.load`` cached rollouts from ``debug-tmp``); fire
            timestamps are still recorded so downstream timing math
            stays consistent.
            """
            nonlocal fired_step
            nonlocal debug_skip_rollout
            target = min(target, total_ppo_step)
            if fired_step < target:
                await self.sampler_group.write_engine_log_marker(
                    fired_step, phase=f"fire ppo_step {fired_step}..{target - 1}"
                )
                if self.gen_rm_group is not None:
                    for grp in self.gen_rm_group:
                        await grp.write_engine_log_marker(
                            fired_step, phase=f"fire ppo_step {fired_step}..{target - 1}"
                        )
            while fired_step < target:
                epoch = epoch_for(fired_step)
                epoch_end = (epoch + 1) * ppo_step_per_epoch
                chunk_end = min(target, epoch_end)
                n = chunk_end - fired_step
                assert n > 0, (
                    f"fire_up_to: non-positive chunk size "
                    f"(fired={fired_step}, target={target}, epoch_end={epoch_end})"
                )
                if not debug_skip_rollout:
                    await rc.fire_generation_requests.remote(epoch, fired_step, n)
                now = time.monotonic()
                for offset in range(n):
                    self._pipeline_ts.setdefault(fired_step + offset, {})["fire"] = now
                delay_s = training_cfg.load_aware_sampler_dispatch_stagger_s
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                log(f"[SingleCtrlTrainer] fire_up_to: sleep {delay_s} s")

                fired_step += n

        # Cold start: prime the queue with up to ``s + 1`` batches.
        await tg.log_memory(f"memory tracking before fire ppo_step {train_step}")
        await fire_up_to(train_step + (max_stale + 1))

        ret_metrics = []

        while train_step < total_ppo_step:
            # Collect + train the current step.  By default
            # ``collect_rollout_step`` pulls from the first-finished queue;
            # with ``rollout_ordered_collection`` it waits for the original
            # ``train_step`` rollout and preserves microbatch order.
            if debug_skip_rollout:
                dp_refs = None
                t_collect_done = time.monotonic()
                fire_rollout_elapsed = 0.0
                collect_elapsed = 0.0
            else:
                t_collect_start = time.monotonic()
                gen_result = await rc.collect_rollout_step.remote(train_step)
                dp_refs = gen_result.dp_refs
                t_collect_done = time.monotonic()
                fire_time = self._pipeline_ts[train_step]["fire"]
                fire_rollout_elapsed = t_collect_done - fire_time
                collect_elapsed = t_collect_done - t_collect_start
            self._pipeline_ts.setdefault(train_step, {})["collect"] = t_collect_done
            await tg.log_memory(f"memory tracking after collect ppo_step {train_step}")

            t_train_start = time.monotonic()
            driver_timing = {
                "time_perf/fire_rollout_elapsed": fire_rollout_elapsed,
                "time_perf/rollout": collect_elapsed,
            }
            step_metrics_per_actor = await tg.train_step(
                epoch_for(train_step),
                train_step,
                dp_refs,
                extra_metrics=driver_timing,
            )
            t_train_done = time.monotonic()
            self._pipeline_ts.setdefault(train_step, {})["train_done"] = t_train_done

            train_elapsed = t_train_done - t_train_start
            log(
                f"[SingleCtrlTrainer] step {train_step} timing: "
                f"fire_rollout={fire_rollout_elapsed:.1f}s, "
                f"rollout_wait={collect_elapsed:.1f}s, "
                f"train={train_elapsed:.1f}s"
            )

            if config.debug.trainer_return_ppo_step_metrics:
                if not ret_metrics:
                    ret_metrics = [[] for _ in step_metrics_per_actor]
                assert len(ret_metrics) == len(step_metrics_per_actor)
                for actor_metrics, step_metric in zip(
                    ret_metrics, step_metrics_per_actor, strict=True
                ):
                    actor_metrics.append(step_metric)

            last_trained_step = train_step
            train_step += 1
            steps_since_update += 1

            reached_window_boundary = steps_since_update >= window_size
            at_end_of_training = train_step == total_ppo_step

            if reached_window_boundary or at_end_of_training:
                # Save-checkpoint first, THEN wait for sampler to drain.
                if train_step % training_cfg.save_interval == 0:
                    await tg.save_checkpoint(train_step)

                # Drain inflight rollout so the sampler is idle before we
                # change its weights.  Skipped when
                # ``debug_skip_rollout`` is enabled (no sampler activity).
                if not debug_skip_rollout:
                    await rc.wait_all_inflight.remote()

                # update_weights is called at every window close
                update_start = time.monotonic()
                self._pipeline_ts.setdefault(last_trained_step, {})["update_start"] = update_start
                await tg.update_weights(
                    offload=config.placement_type == "colocate", flush_cache=True
                )
                update_done = time.monotonic()
                self._pipeline_ts.setdefault(last_trained_step, {})["update_done"] = update_done

                steps_since_update = 0

                if not at_end_of_training:
                    await fire_up_to(train_step + 1 + max_stale)

                    window_elapsed = update_done - fire_time
                    log(
                        f"[SingleCtrlTrainer] window closed: "
                        f"train_step={train_step - 1}/{total_ppo_step}, "
                        f"prefetch_depth={max_stale + 1}, "
                        f"window_elapsed={window_elapsed:.1f}s"
                    )

        # final save checkpoint, 避免 train_step % training_cfg.save_interval 保存两次
        if train_step % training_cfg.save_interval != 0:
            await tg.save_checkpoint(train_step)

        return ret_metrics

    async def run_train_loop(self, config: RlConfig):
        """Public entry that wraps ``_run_train_loop`` with a top-level
        exception logger so failures inside the loop don't get swallowed
        by the surrounding asyncio / Ray machinery.

        Parameters
        ----------
        config : RlConfig

        Returns
        -------
        Same as :meth:`_run_train_loop`.
        """
        try:
            return await self._run_train_loop(config)
        except Exception as e:
            log(f"run_train_loop error: {e}")
            traceback.print_exc()
            raise
