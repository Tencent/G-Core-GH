"""Unit tests for ``GrpoSingleCtrlTrainer._run_train_loop`` (rolling
prefetch).

These tests drive the loop in-process against a ``_FakeRolloutController``
and ``_FakeTrainGroup``, asserting the event sequence for a range of
``(max_stale, total_ppo_step, save_interval, prev_ppo_step,
ppo_step_per_epoch)`` configurations.  No Ray, no GPU, no real sampler.
"""

import unittest
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Tuple

from gpatch_v4.trainer.grpo_single_ctrl_trainer import GrpoSingleCtrlTrainer

# ---- Fakes ---------------------------------------------------------- #


class RlConfig(SimpleNamespace):
    pass


class _FakeEngineGroup:
    """No-op stand-in for sampler_group / gen_rm_group."""
    async def write_engine_log_marker(self, *args, **kwargs):
        pass


@dataclass
class _GenerateResult:
    """Mimic ``GenerateResult`` with a per-step dp_refs list."""
    dp_refs: List[str] = field(default_factory=list)


class _RemoteMethod:
    """Mimic ``rc.<method>.remote(...)`` returning an awaitable."""
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return await self._fn(*args, **kwargs)


class _FakeRolloutController:
    """Tracks fire / collect / wait events and guards:
      - each ``fire_generation_requests`` call has a single ``epoch``;
      - ``collect_rollout_step(step)`` is only legal if ``step`` has been
        fired and not yet collected (i.e. is sitting in the queue).
    """
    def __init__(self, events: List[Tuple], ppo_step_per_epoch: int):
        self.events = events
        self.ppo_step_per_epoch = ppo_step_per_epoch
        # A FIFO queue of PPO steps that have been fired but not yet
        # collected.  Used to simulate first-finished ordering: we expect
        # the trainer loop to collect steps in the same order it fires them.
        self._pending_steps: List[int] = []

        self.fire_generation_requests = _RemoteMethod(self._fire_generation_requests)
        self.collect_rollout_step = _RemoteMethod(self._collect_rollout_step)
        self.wait_all_inflight = _RemoteMethod(self._wait_all_inflight)
        self.save_data_source = _RemoteMethod(self._save_data_source)

    async def _fire_generation_requests(self, epoch: int, ppo_step: int, num_ppo_steps: int):
        # Validate single-epoch invariant: caller must never fire across
        # an epoch boundary within one call.
        expected_epoch = ppo_step // self.ppo_step_per_epoch
        assert epoch == expected_epoch, (
            f"fire: epoch arg {epoch} != epoch_for(ppo_step={ppo_step})"
            f"={expected_epoch}"
        )
        last_step = ppo_step + num_ppo_steps - 1
        last_epoch = last_step // self.ppo_step_per_epoch
        assert last_epoch == expected_epoch, (
            f"fire: span {ppo_step}..{last_step} crosses an epoch "
            f"(start epoch={expected_epoch}, end epoch={last_epoch})"
        )

        self.events.append(("fire", epoch, ppo_step, num_ppo_steps))
        for i in range(num_ppo_steps):
            self._pending_steps.append(ppo_step + i)

    async def _collect_rollout_step(self, ppo_step: int):
        assert self._pending_steps, (f"collect({ppo_step}) called but no steps have been fired")
        # Trainer passes train_step; we model the queue as FIFO.
        fired_step = self._pending_steps.pop(0)
        self.events.append(("collect", ppo_step))
        return _GenerateResult(dp_refs=[f"dp_ref_{fired_step}"])

    async def _wait_all_inflight(self):
        self.events.append(("wait_all_inflight", ))

    async def _save_data_source(self, step: int):
        self.events.append(("save_data_source", step))


class _FakeSamplerGroup:
    """Minimal sampler group that records ``write_engine_log_marker`` calls."""
    def __init__(self, events: List[Tuple]):
        self.events = events

    async def write_engine_log_marker(self, ppo_step: int, phase: str = ""):
        self.events.append(("sampler_log_marker", ppo_step, phase))


class _FakeTrainGroup:
    """Minimal train group whose ``init_train_state`` is parameterizable."""
    def __init__(
        self,
        events: List[Tuple],
        total_ppo_step: int,
        ppo_step_per_epoch: int,
        prev_ppo_step: int = 0,
    ):
        self.events = events
        self.total_ppo_step = total_ppo_step
        self.ppo_step_per_epoch = ppo_step_per_epoch
        self.prev_ppo_step = prev_ppo_step

    async def init_train_state(self):
        # num_train_epoches is informative only; total_ppo_step is what
        # the loop actually uses.
        return {
            "total_ppo_step": self.total_ppo_step,
            "ppo_step_per_epoch": self.ppo_step_per_epoch,
            "num_train_epoches": max(1, self.total_ppo_step // self.ppo_step_per_epoch),
            "prev_ppo_step": self.prev_ppo_step,
        }

    async def update_weights(self, offload: bool = True, flush_cache: bool = False):
        self.events.append(("update_weights", offload, flush_cache))

    async def train_step(self, epoch: int, ppo_step: int, dp_refs, extra_metrics=None):
        self.events.append(("train_step", epoch, ppo_step, tuple(dp_refs)))
        return [{"policy/loss": 0.0}]

    async def save_checkpoint(self, step: int):
        self.events.append(("save_checkpoint", step))

    async def prepare_for_final_save(self):
        self.events.append(("prepare_for_final_save", ))

    async def log_memory(self, tag: str):
        pass


def _make_config(
    max_stale: int,
    save_interval: int = 100,
    placement_type: str = "disaggregated",
) -> Any:
    is_colocate = placement_type == "colocate"
    cfg_cls = RlConfig if placement_type == "partial_colocated" else SimpleNamespace
    cfg = cfg_cls(
        placement_type=placement_type,
        training=SimpleNamespace(
            async_rollout=not is_colocate,
            single_controller=True,
            rollout_max_staleness=max_stale,
            save_interval=save_interval,
            exit_step=None,
            load_aware_sampler_dispatch_stagger_s=0,
            use_gen_rm_reward=placement_type == "partial_colocated",
            use_bt_rm_reward=False,
        ),
        debug=SimpleNamespace(
            trainer_return_ppo_step_metrics=False,
            skip_rollout_load_from_disk=False,
            disable_save_checkpoint=False,
        ),
    )
    if placement_type == "partial_colocated":
        dc_policy = SimpleNamespace(nnodes=2, num_gpus_per_node=8)
        dc_sampler = SimpleNamespace(
            nnodes=1,
            num_gpus_per_node=8,
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
        )
        dc_gen_rm = SimpleNamespace(nnodes=1, num_gpus_per_node=8)
        cfg.policy = SimpleNamespace(dist_config=dc_policy)
        cfg.sampler = SimpleNamespace(
            dist_config=dc_sampler,
            backend="sglang",
            model_info=[SimpleNamespace()],
            infer_engine_configs=[SimpleNamespace(dist_config=dc_sampler)],
        )
        cfg.gen_rm = SimpleNamespace(
            dist_config=dc_gen_rm,
            backend="sglang",
            reward_model_info=[SimpleNamespace()],
            infer_engine_configs=[SimpleNamespace(dist_config=dc_sampler)],
        )
        cfg.bt_rm = SimpleNamespace(dist_config=SimpleNamespace(nnodes=0, num_gpus_per_node=8))
    return cfg


async def _run_loop(
    max_stale: int,
    total_ppo_step: int,
    ppo_step_per_epoch: int = 100,
    save_interval: int = 100,
    prev_ppo_step: int = 0,
    placement_type: str = "disaggregated",
):
    events: List[Tuple] = []
    trainer = object.__new__(GrpoSingleCtrlTrainer)
    trainer.train_group = _FakeTrainGroup(events, total_ppo_step, ppo_step_per_epoch, prev_ppo_step)
    trainer.rollout_controller = _FakeRolloutController(events, ppo_step_per_epoch)
    trainer.sampler_group = _FakeSamplerGroup(events)
    trainer._pipeline_ts = {}
    trainer.sampler_group = _FakeEngineGroup()
    trainer.gen_rm_group = None

    await trainer._run_train_loop(
        _make_config(
            max_stale,
            save_interval=save_interval,
            placement_type=placement_type,
        )
    )
    return events, trainer.get_pipeline_stats()


# ---- Helpers -------------------------------------------------------- #


def _train_step_events(events):
    return [e for e in events if e[0] == "train_step"]


def _fire_events(events):
    return [e for e in events if e[0] == "fire"]


def _save_events(events):
    return [e for e in events if e[0] == "save_checkpoint"]


def _update_events_non_initial(events):
    """Return update events after the first initialization update."""
    updates = [e for e in events if e[0] == "update_weights"]
    return updates[1:]


# ---- Tests ---------------------------------------------------------- #


class SlidingPrefetchLoopTest(unittest.IsolatedAsyncioTestCase):
    async def test_s_eq_0_sync_behavior(self):
        """s=0 degenerates to synchronous: fire(1) per step, wait+update
        after every train, tail save only.
        """
        events, _ = await _run_loop(max_stale=0, total_ppo_step=3)

        # Each train step is preceded by a fire of exactly 1.
        fires = _fire_events(events)
        assert fires == [
            ("fire", 0, 0, 1),
            ("fire", 0, 1, 1),
            ("fire", 0, 2, 1),
        ]

        trains = _train_step_events(events)
        assert [e[2] for e in trains] == [0, 1, 2]
        assert [e[3] for e in trains] == [("dp_ref_0", ), ("dp_ref_1", ), ("dp_ref_2", )]

        # One update_weights per window close, including the final window
        # (so downstream eval sees the latest weights) → 3 mid updates.
        updates = _update_events_non_initial(events)
        assert len(updates) == 3

        saves = _save_events(events)
        assert saves == [("save_checkpoint", 3)]  # tail save only

    async def test_colocate_single_controller_is_s_eq_0_fire_collect_loop(self):
        """Colocate single-controller reuses the s=0 fire/collect schedule."""
        events, _ = await _run_loop(
            max_stale=0,
            total_ppo_step=3,
            placement_type="colocate",
        )

        assert _fire_events(events) == [
            ("fire", 0, 0, 1),
            ("fire", 0, 1, 1),
            ("fire", 0, 2, 1),
        ]
        assert [e[2] for e in _train_step_events(events)] == [0, 1, 2]

        update_events = [e for e in events if e[0] == "update_weights"]
        assert update_events
        assert all(e[1] is True for e in update_events)

    async def test_partial_colocated_drains_rollout_before_training(self):
        """partial_colocated sleeps inference engines before policy train_step."""
        events, _ = await _run_loop(
            max_stale=0,
            total_ppo_step=1,
            placement_type="partial_colocated",
        )

        collect_idx = events.index(("collect", 0))
        train_idx = next(i for i, e in enumerate(events) if e[0] == "train_step")
        wait_indices = [i for i, e in enumerate(events) if e[0] == "wait_all_inflight"]

        assert any(collect_idx < i < train_idx for i in wait_indices)

    async def test_colocate_rejects_nonzero_staleness(self):
        trainer = object.__new__(GrpoSingleCtrlTrainer)
        trainer.train_group = _FakeTrainGroup([], total_ppo_step=1, ppo_step_per_epoch=1)
        trainer.rollout_controller = _FakeRolloutController([], ppo_step_per_epoch=1)
        trainer._pipeline_ts = {}
        trainer.sampler_group = _FakeEngineGroup()
        trainer.gen_rm_group = None

        with self.assertRaises(AssertionError):
            await trainer._run_train_loop(
                _make_config(max_stale=1, placement_type="colocate")
            )

    async def test_s_eq_1_cold_start_fires_2_then_1_per_step(self):
        """s=1: cold fires 2 batches; after each train, wait+update+fire(1)
        until training ends.
        """
        events, _ = await _run_loop(max_stale=1, total_ppo_step=4)

        fires = _fire_events(events)
        # Cold fire(2) + 2 mid-loop fires of 1 (no fire after the last window).
        assert fires == [
            ("fire", 0, 0, 2),
            ("fire", 0, 2, 1),
            ("fire", 0, 3, 1),
        ]

        # Cold fire primes 2; first train can start immediately.
        # Critical property: fire for step 2 happens BEFORE train of step 1.
        fire_2_idx = events.index(("fire", 0, 2, 1))
        train_1_idx = events.index(("train_step", 0, 1, ("dp_ref_1", )))
        assert fire_2_idx < train_1_idx, (
            "Rolling prefetch must re-prime after update before next train"
        )

    async def test_s_eq_2_full_window_sequence(self):
        """s=2, total=7: walk the full plan-documented table.

        After the cold-start fire of 3 batches, every window boundary
        re-primes by firing only ``s = 2`` new batches, because the
        ``1`` carry-over batch that survives across ``update_weights``
        already accounts for 1 of the ``s + 1`` in-system budget.  This
        is the rolling-prefetch invariant.
        """
        events, _ = await _run_loop(max_stale=2, total_ppo_step=7)

        fires = _fire_events(events)
        # cold fire_up_to(0+3)=3: fire(epoch=0, 0, 3) [fired=3]
        # after train 0,1 → window close; fire_up_to(2+3)=5:
        #     fire(epoch=0, 3, 2) [fired=5]
        # after train 2,3 → window close; fire_up_to(4+3)=7:
        #     fire(epoch=0, 5, 2) [fired=7]
        # after train 4,5 → window close; fire_up_to(6+3) clipped to 7:
        #     no fire (already at target).
        # after train 6 (at_end): wait, update (so sampler has latest
        #     weights for eval), skip fire, tail save.
        assert fires == [
            ("fire", 0, 0, 3),
            ("fire", 0, 3, 2),
            ("fire", 0, 5, 2),
        ]

        trains = _train_step_events(events)
        assert [e[2] for e in trains] == list(range(7))

        updates = _update_events_non_initial(events)
        # Mid-training updates happen at every window boundary:
        # train_step ∈ {2, 4, 6} (after training steps 1, 3, 5) plus the
        # final boundary at train_step == 7 (after training step 6).
        # update_weights always runs so the sampler ends with the latest
        # weights for any downstream eval.  → 4 updates.
        assert len(updates) == 4

    async def test_save_at_last_step_total_divisible_by_save_interval(self):
        """When ``total % save_interval == 0`` the boundary branch saves
        at ``train_step == total``; the tail branch is guarded by
        ``% != 0`` and does NOT run.  Exactly one save at ``total``.
        """
        events, _ = await _run_loop(max_stale=2, total_ppo_step=6, save_interval=6)
        saves = _save_events(events)
        assert saves == [("save_checkpoint", 6)], f"got {saves}"

    async def test_save_at_last_step_tail_branch(self):
        """When ``save_interval`` coincides with ``total``, the last
        boundary matches the modulo → boundary save runs at ``total``.
        Tail save is skipped by the ``% != 0`` guard.
        """
        events, _ = await _run_loop(max_stale=2, total_ppo_step=7, save_interval=7)
        saves = _save_events(events)
        # Only step 7 matches save_interval=7; boundary fires, tail skipped.
        assert saves == [("save_checkpoint", 7)], f"got {saves}"

    async def test_unconfigured_small_window_not_multiple(self):
        """Coverage for ``max_stale > 1`` with ``total`` NOT a multiple of
        ``window_size``: ``s=2, total=5``.  The last window trains only 1
        step before ``at_end_of_training`` fires.  ``fire_up_to`` must
        clip at ``total_ppo_step`` and never over-fire.
        """
        events, _ = await _run_loop(max_stale=2, total_ppo_step=5, save_interval=100)

        fires = _fire_events(events)
        # cold fire_up_to(0+3)=3:            fire(epoch=0, 0, 3) [fired=3]
        # after train 0,1 → boundary:
        #     fire_up_to(2+3)=5:             fire(epoch=0, 3, 2) [fired=5]
        # after train 2,3 → boundary:
        #     fire_up_to(4+3)=min(7,5)=5:    no fire (already at target)
        # after train 4 (at_end):            skip fire
        assert fires == [
            ("fire", 0, 0, 3),
            ("fire", 0, 3, 2),
        ], f"fires={fires}"

        trains = _train_step_events(events)
        assert [e[2] for e in trains] == [0, 1, 2, 3, 4]

        updates = _update_events_non_initial(events)
        # Boundaries fire updates at train_step ∈ {2, 4, 5}.  All run
        # (final window update too, so sampler has latest weights for
        # eval).  → 3 mid updates.
        assert len(updates) == 3

        saves = _save_events(events)
        # save_interval=100, no boundary hits; tail runs once.
        assert saves == [("save_checkpoint", 5)], f"saves={saves}"

    async def test_final_save_always_happens_exactly_once(self):
        """Across a sweep of (s, total, save_interval), the final step
        must be saved exactly once (either via boundary or tail; the two
        are mutually exclusive via ``% save_interval``).
        """
        for s in (0, 1, 2, 3):
            for total in (1, 2, 3, 5, 6, 7, 10):
                for si in (1, 2, 3, total):
                    events, _ = await _run_loop(max_stale=s, total_ppo_step=total, save_interval=si)
                    save_steps = [e[1] for e in _save_events(events)]
                    final_saves = [step for step in save_steps if step == total]
                    assert len(final_saves) == 1, (
                        f"s={s} total={total} si={si}: "
                        f"expected exactly one final save, got {save_steps}"
                    )
                    # Save-step must be unique (no duplicates anywhere).
                    assert len(save_steps) == len(
                        set(save_steps)
                    ), (f"s={s} total={total} si={si}: "
                        f"duplicate saves: {save_steps}")

    async def test_data_source_is_saved_after_each_model_checkpoint(self):
        events, _ = await _run_loop(
            max_stale=1,
            total_ppo_step=5,
            save_interval=2,
        )

        model_save_indices = [
            index for index, event in enumerate(events)
            if event[0] == "save_checkpoint"
        ]
        self.assertEqual(
            [events[index] for index in model_save_indices],
            [
                ("save_checkpoint", 2),
                ("save_checkpoint", 4),
                ("save_checkpoint", 5),
            ],
        )
        for index in model_save_indices:
            self.assertEqual(
                events[index + 1],
                ("save_data_source", events[index][1]),
            )

    async def test_cross_epoch_fires_split_per_epoch(self):
        """Reviewer blocking issue 1: the trainer must never ask a single
        ``fire_generation_requests`` call to cross an epoch boundary.

        ``_FakeRolloutController._fire_generation_requests`` asserts this
        invariant; this test sets up a config that would violate it if
        ``fire_up_to`` didn't clip on the epoch end.
        """
        # s=2 means cold fire targets 3 batches.  With ppo_step_per_epoch=4
        # and total=8 (2 epochs), after train 0..3 the next fire_up_to
        # target is step 4+3=7, starting from fired_step=4 which is the
        # boundary into epoch 1 -- but the preceding fires straddle the
        # boundary exactly.  Walk through:
        #   cold fire_up_to(3): fire(epoch=0, 0, 3)  [fired=3]
        #   after train 0,1; update; fire_up_to(2+3)=fire_up_to(5)
        #       fired=3, epoch 0 end=4 -> fire(epoch=0, 3, 1) [fired=4]
        #       fired=4, epoch 1 end=8 -> fire(epoch=1, 4, 1) [fired=5]
        #   after train 2,3; update; fire_up_to(4+3)=7
        #       fired=5, epoch 1 end=8 -> fire(epoch=1, 5, 2) [fired=7]
        #   after train 4,5; update; fire_up_to(6+3)=fire_up_to(8)
        #       fired=7, epoch 1 end=8 -> fire(epoch=1, 7, 1) [fired=8]
        events, _ = await _run_loop(
            max_stale=2,
            total_ppo_step=8,
            ppo_step_per_epoch=4,
        )

        expected_fires = [
            ("fire", 0, 0, 3),
            ("fire", 0, 3, 1),
            ("fire", 1, 4, 1),
            ("fire", 1, 5, 2),
            ("fire", 1, 7, 1),
        ]
        assert _fire_events(events) == expected_fires, (f"fires mismatch: {_fire_events(events)}")

    async def test_resume_mid_epoch(self):
        """Resume from ``prev_ppo_step=2`` with ``ppo_step_per_epoch=4``
        and ``s=2``: cold fire must split at the epoch boundary.
        """
        events, _ = await _run_loop(
            max_stale=2,
            total_ppo_step=5,
            ppo_step_per_epoch=4,
            prev_ppo_step=2,
        )

        fires = _fire_events(events)
        # cold fire_up_to(2+3)=5:
        #   fired=2, epoch 0 end=4 -> fire(epoch=0, 2, 2) [fired=4]
        #   fired=4, epoch 1 end=8 -> fire(epoch=1, 4, 1) [fired=5]
        assert fires[:2] == [
            ("fire", 0, 2, 2),
            ("fire", 1, 4, 1),
        ]
        # And the training only runs 3 steps: 2, 3, 4.
        trains = _train_step_events(events)
        assert [e[2] for e in trains] == [2, 3, 4]

    async def test_total_ppo_step_one(self):
        """Degenerate ``total_ppo_step=1``: cold fire(1), train 1, save."""
        events, _ = await _run_loop(max_stale=1, total_ppo_step=1, save_interval=100)
        fires = _fire_events(events)
        # Cold fire wants to prime 2, clipped to 1 by total.
        assert fires == [("fire", 0, 0, 1)]
        assert [e[2] for e in _train_step_events(events)] == [0]
        # End-of-training still does wait_all_inflight + update_weights
        # (so downstream eval sees the latest weights).  No new fires,
        # no double updates.
        assert ("wait_all_inflight", ) in events
        assert len(_update_events_non_initial(events)) == 1
        assert _save_events(events) == [("save_checkpoint", 1)]

    async def test_wait_before_every_update(self):
        """Every mid-training ``update_weights`` must be preceded by a
        ``wait_all_inflight`` (with no other ``update_weights`` between
        them).  A ``save_checkpoint`` is allowed to interleave because
        save overlaps with wait on disjoint resources.
        """
        events, _ = await _run_loop(max_stale=2, total_ppo_step=7)
        for i, ev in enumerate(events):
            if ev[0] == "update_weights" and ev[1] is False:
                # Scan backwards for the matching wait_all_inflight.
                found = False
                for j in range(i - 1, -1, -1):
                    if events[j] == ("wait_all_inflight", ):
                        found = True
                        break
                    # Can't cross another update_weights going back.
                    assert events[j][0] != "update_weights", (
                        f"update_weights at {i} not preceded by wait"
                    )
                assert found, (f"update_weights at {i} has no wait_all_inflight before it")

    async def test_epoch_arg_always_matches_ppo_step(self):
        """The fake controller already asserts this per-call; this test
        just triggers a wide config sweep to exercise the invariant.
        """
        for s in (0, 1, 2, 3):
            for total in (1, 3, 5, 8):
                for ppe in (1, 2, 4, total):
                    for prev in (0, ) + ((ppe - 1, ) if ppe > 1 else ()):
                        if prev >= total:
                            continue
                        await _run_loop(
                            max_stale=s,
                            total_ppo_step=total,
                            ppo_step_per_epoch=ppe,
                            prev_ppo_step=prev,
                        )

    async def test_pipeline_timestamps_recorded(self):
        """Every trained step must have ``fire``, ``collect``, ``train_done``;
        every window-closing step (including the final one) must also have
        ``update_start``/``update_done``.
        """
        _, stats = await _run_loop(max_stale=2, total_ppo_step=7)
        for step in range(7):
            assert "fire" in stats[step], f"step {step}: missing 'fire'"
            assert "collect" in stats[step], f"step {step}: missing 'collect'"
            assert "train_done" in stats[step], f"step {step}: missing 'train_done'"
            assert stats[step]["fire"] <= stats[step]["collect"]
            assert stats[step]["collect"] <= stats[step]["train_done"]
        # Window closes at steps 1, 3, 5 (s=2 boundaries) and at step 6
        # (final step).  update_weights runs at every close.
        for closing_step in (1, 3, 5, 6):
            assert "update_start" in stats[closing_step], (
                f"step {closing_step}: missing update_start"
            )
            assert "update_done" in stats[closing_step]

    async def test_no_collect_before_corresponding_fire(self):
        """Sanity: each ``collect`` event must follow the matching fire.

        This is also enforced by ``_FakeRolloutController._collect_rollout_step``
        (which asserts the step was fired), but the explicit test documents
        intent.
        """
        for s in (0, 1, 2):
            for total in (1, 4, 7):
                events, _ = await _run_loop(max_stale=s, total_ppo_step=total)
                fired_steps: Counter = Counter()
                for ev in events:
                    if ev[0] == "fire":
                        _, _epoch, start, n = ev
                        for i in range(n):
                            fired_steps[start + i] += 1
                    elif ev[0] == "collect":
                        step = ev[1]
                        assert fired_steps[step] >= 1, (
                            f"collect({step}) before any fire at s={s} "
                            f"total={total}: {events}"
                        )
                        fired_steps[step] -= 1


if __name__ == "__main__":
    unittest.main()
