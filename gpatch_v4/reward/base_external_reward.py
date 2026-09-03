import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseExternalReward(ABC):
    """Abstract base class for external rewards.

    Subclasses must implement :meth:`calc_external_reward` as an async coroutine.

    **Important**: do NOT call synchronous collective operations
    (``broadcast``, ``barrier``, etc.) inside ``calc_external_reward``.
    The actor handles broadcast after ``await``.

    Call flow in the actor::

        started = asyncio.Event()
        task = asyncio.create_task(
            reward.calc_external_reward(rollout_batches, ..., _started_event=started))
        await started.wait()             # task fires off async I/O, then sets event
        ... synchronous GPU work ...     # overlaps with network round-trip
        reward_updates = await task
        reward_updates = BroadcastUtils.broadcast_rollout_batch(reward_updates)
        for rb, upd in zip(rollout_batches, reward_updates):
            rb.update(upd)
    """

    # Whether several ``calc_external_reward`` calls may overlap on this instance.
    # The batched pipeline awaits each call before issuing the next, so per-call
    # state on the instance, the class or a module singleton is safe there;
    # ``training.stream_external_reward`` overlaps them and is declined unless a
    # reward sets this, meaning every client, buffer and singleton it touches
    # tolerates concurrent callers.
    supports_concurrent_calls: bool = False

    def __init__(self, config=None, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    async def calc_external_reward(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
        is_eval: bool = False,
        _started_event: Optional[asyncio.Event] = None,
    ) -> List[Dict[str, Any]]:
        """Compute external rewards.

        Parameters
        ----------
        rollout_batches : list of dict
            Read-only reference to rollout data.  Do **not** mutate;
            return the fields to merge instead.
        ppo_step : int
            Current PPO step.
        is_eval : bool
            Whether this is an evaluation rollout.
        _started_event : asyncio.Event, optional
            If provided, the implementation **must** call ``set()`` after
            all async sub-tasks are scheduled (so the actor can resume
            synchronous work while network I/O is in flight).

        Returns
        -------
        list of dict
            One update dict per rollout batch.
        """
        ...


def declares_concurrent_calls(reward) -> bool:
    """Whether ``reward`` has opted in to overlapping ``calc_external_reward`` calls.

    Reads the field ``BaseExternalReward`` declares, which defaults to off:
    inheriting it and saying nothing means no.  A reward that never subclassed
    the base class -- ``_setup_external_reward`` requires only a
    ``calc_external_reward`` attribute -- raises here instead, which is where
    a config asking for streaming should stop.
    """
    return bool(reward.supports_concurrent_calls)


def stream_external_reward_declined_reason(reward) -> Optional[str]:
    """Why ``training.stream_external_reward`` must be declined, or None to proceed."""
    if declares_concurrent_calls(reward):
        return None
    return (
        f"{type(reward).__name__} has not set supports_concurrent_calls=True; "
        "streaming keeps several calc_external_reward calls in flight on one "
        "instance, so any per-call state it parks on the class or in a "
        "singleton is shared between them"
    )
