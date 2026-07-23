import asyncio
import importlib
import time
from typing import Optional

from gpatch_v4 import orches
from gpatch_v4.orches.exceptions import RayActorError, RayTaskError
from gpatch_v4.orches.failure import FailureEvent, FailureType
from gpatch_v4.orches.node_replacer import NodeReplacer
from gpatch_v4.orches.resource_allocator import (
    SHARED_WITH_POLICY,
    allocation_from_config,
)
from gpatch_v4.utils import log
from gpatch_v4.utils.placement import is_partial_colocated


def _instantiate_node_replacer(
    cls_path: Optional[str],
    config=None,
) -> Optional[NodeReplacer]:
    """Instantiate a NodeReplacer from its fully qualified class path.

    Parameters
    ----------
    cls_path : str or None
        e.g. ``"gpatch_v4.orches.node_replacer.MockNodeReplacer"``;
        *None* means no node replacement.
    config : object, optional
        Forwarded to the constructor when accepted (e.g.
        ``GeminiNodeReplacer``).

    Returns
    -------
    NodeReplacer or None
    """
    if cls_path is None:
        return None
    module_path, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert issubclass(cls, NodeReplacer), (f"{cls_path} is not a subclass of NodeReplacer")
    try:
        return cls(config=config)
    except TypeError:
        return cls()


class TrainerRetryMixin:
    """Mixin that adds automatic retry logic to trainers.

    Monitors the training loop for liveness (hang detection) and catches
    actor crashes (RayActorError). On failure, optionally replaces
    faulty nodes via a :class:`NodeReplacer` and restarts training
    from the latest checkpoint.
    """
    async def launch_then_run_with_recovery(self, config):
        """Launch training with automatic retry on failure.

        Handles two failure modes:
        - **Hang**: detected by ``check_liveness()`` timeout.
        - **Crash**: detected by catching ``RayActorError`` / ``RayTaskError``
          propagated from the training loop.

        Parameters
        ----------
        config : object
            Must have ``training.max_restart_attempts``; optionally
            ``training.node_replacer_cls``.

        Raises
        ------
        RuntimeError
            If all restart attempts are exhausted or the cluster fails to recover.
        """
        max_launch_attempts = (config.training.max_restart_attempts or 0) + 1
        node_replacer = _instantiate_node_replacer(
            config.training.node_replacer_cls,
            config=config,
        )
        launch_count = 0
        failure_event: Optional[FailureEvent] = None

        while launch_count < max_launch_attempts:
            launch_count += 1
            await self.launch(config)

            task_train_loop = asyncio.create_task(self.train_group.train_loop())
            task_check_liveness = asyncio.create_task(self.train_group.check_liveness())

            failure_event = None
            try:
                done, pending = await asyncio.wait(
                    [task_train_loop, task_check_liveness],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if task_check_liveness in done:
                    liveness_result = task_check_liveness.result()
                    if liveness_result is None:
                        # Training finished normally
                        log(
                            "liveness check passed, train step is not timelimited "
                            "or training finished normally"
                        )

                        training_plt_group_loop = None
                        if hasattr(self, "training_plt_group"):
                            log("found training_plt_group, start training_plt_group loop")
                            training_plt_group_loop = asyncio.create_task(
                                self.training_plt_group.run_loop()
                            )

                        loop_ret = await task_train_loop
                        if training_plt_group_loop is not None:
                            self.training_plt_group.stop()
                            await training_plt_group_loop

                        return loop_ret
                    else:
                        # Hang detected
                        failure_event = liveness_result

                elif task_train_loop in done:
                    # train_loop completed first — cancel the liveness checker
                    task_check_liveness.cancel()
                    try:
                        await task_check_liveness
                    except asyncio.CancelledError:
                        pass
                    # This may raise if the train_loop ended with an exception (crash)
                    return task_train_loop.result()

            except (RayActorError, RayTaskError) as e:
                # Crash: actor died, exception propagated from train_loop or await
                failed_ips = self._extract_node_ips_from_error(e)
                failure_event = FailureEvent(
                    failure_type=FailureType.CRASH,
                    failed_node_ips=failed_ips,
                    failed_actor_names=[],
                    timestamp=time.time(),
                    details=str(e),
                )

            # --- Restart path (only if we have remaining attempts) ---
            if launch_count >= max_launch_attempts:
                break

            log(
                f"Failure detected: {failure_event}, "
                f"launch #{launch_count}/{max_launch_attempts}"
            )

            # Cancel both tasks
            for t in [task_train_loop, task_check_liveness]:
                if not t.done():
                    t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            orches.shutdown()
            shutdown_deadline = time.time() + 60
            while orches.is_initialized():
                if time.time() > shutdown_deadline:
                    log("WARNING: orches.shutdown() did not complete within 60s, proceeding")
                    break
                await asyncio.sleep(0.5)

            # Node replacement (if configured)
            if node_replacer and failure_event:
                node_replacer.evict_nodes(failure_event.failed_node_ips)
                node_replacer.provision_nodes(len(failure_event.failed_node_ips))
                expected_nnodes = self._get_expected_nnodes(config)
                if not node_replacer.wait_cluster_ready(expected_nnodes):
                    raise RuntimeError(
                        f"Cluster did not recover to {expected_nnodes} nodes "
                        f"after {launch_count} launch attempts"
                    )

            await asyncio.sleep(3)

        # Exhausted all launch attempts
        raise RuntimeError(
            f"Training failed after {max_launch_attempts} launch attempts. "
            f"Last failure: {failure_event}"
        )

    def _get_expected_nnodes(self, config) -> int:
        """Compute the expected number of nodes from config.

        Sums per-role allocation from :func:`allocation_from_config`.
        Roles in ``SHARED_WITH_POLICY`` (``kv`` / ``training_plt``) are
        excluded because they share the ``policy`` placement-group prefix.
        Under ``colocate`` non-shared roles share the same nodes, so
        expected nodes = ``max(role_nnodes)``; under ``disaggregated``
        roles occupy distinct nodes, so expected nodes = ``sum(role_nnodes)``.

        Parameters
        ----------
        config : object

        Returns
        -------
        int
        """
        allocation = allocation_from_config(config)
        considered = [
            nn for role, nn in allocation.role_nnodes.items()
            if role not in SHARED_WITH_POLICY and nn > 0
        ]
        assert considered, "allocation has no non-shared roles"
        if config.placement_type == "colocate":
            return max(considered)
        if is_partial_colocated(config):
            return allocation.role_nnodes["policy"]
        else:
            # disaggregated: roles occupy distinct nodes, sum them.
            return sum(considered)

    def _extract_node_ips_from_error(self, error: Exception) -> list:
        """Best-effort extraction of node IP from a Ray error.

        Parameters
        ----------
        error : Exception

        Returns
        -------
        list[str]
            May be empty if extraction fails.

        Note
        ----
        TODO: implement proper node IP extraction from Ray error metadata.
        When this returns empty, node_replacer skips targeted eviction.
        """
        # 该函数目前仅用于测试，切勿使用！
        try:
            if hasattr(error, 'actor_id'):
                # In newer Ray versions, we could query GCS for the actor's node.
                pass
        except Exception:
            pass
        return []
