import asyncio
from abc import ABC, abstractmethod


class RolloutCoordinator:
    """Coordinate rollout collection and abort excess generation.

    Parameters
    ----------
    sampler_client : object
        Client with an ``abort`` method to stop the sampler.
    """


class RolloutCoordinator:
    """Coordinate rollout collection and abort excess generation.

    Parameters
    ----------
    sampler_client : object
        Client with an ``abort`` method to stop the sampler.
    """
    def __init__(self, sampler_client):
        self.reset()
        self.abort_client = sampler_client
        self.lock = asyncio.Lock()

    def reset(self):
        """Reset coordinator state for a new step."""
        self.rollout_records = set()
        self.aborted = False

    def start_step(self, target_rollout_num):
        """Begin a new rollout step.

        Parameters
        ----------
        target_rollout_num : int
            Expected number of successful rollouts.
        """
        print("start_step", flush=True)
        self.reset()
        self.target_rollout_num = target_rollout_num

    async def report_rollout(self, rollout_record):
        """Report a completed rollout; abort the sampler if target reached.

        Parameters
        ----------
        rollout_record : object
        """
        do_abort = False
        async with self.lock:
            self.rollout_records.add(rollout_record)
            if len(self.rollout_records) > self.target_rollout_num and not self.aborted:
                self.aborted = True
                do_abort = True
        if do_abort:
            await self._abort()

    async def _abort(self):
        """Abort the sampler engine when enough rollouts are collected."""
        print("coordinator abort gen")
        return await self.abort_client.abort()


class BaseRolloutCoordinatorClient(ABC):
    """Abstract client for communicating with a RolloutCoordinator."""
    def __init__(self):
        pass

    @abstractmethod
    def start_step(self, target_rollout_num):
        """Notify the coordinator that a new step has begun.

        Parameters
        ----------
        target_rollout_num : int
            Expected number of rollouts. Only rank 0 calls this.
        """
        ...

    @abstractmethod
    async def report_rollout(self, rollout_record):
        """Report a successful rollout to the coordinator.

        Parameters
        ----------
        rollout_record : object
        """
        ...
