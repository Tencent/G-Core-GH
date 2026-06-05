import asyncio
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from PIL import Image


class RollStatus(Enum):
    """Status of a rollout request or repeat."""

    PENDING = 1
    ABORTED = 2
    TRUNCATED = 3
    COMPLETED = 4


@dataclass
class RepeatResponse:
    """Response for a single repeat within a rollout group.

    Attributes
    ----------
    id : int
        Repeat index.
    finish_reason : str
        ``'stop'`` / ``'length'`` / ``'abort'``.
    generated : list of int
        Generated token IDs.
    logprob : list of float
        Per-token log probabilities.
    """
    id: int
    finish_reason: str
    generated: List[int] = field(default_factory=list)
    logprob: List[float] = field(default_factory=list)


@dataclass
class RolloutGroupResponse:
    """Aggregated response for all repeats of one request group.

    Attributes
    ----------
    repeats : list of RepeatResponse
    """
    repeats: List[RepeatResponse] = field(default_factory=list)

    def is_aborted(self):
        """``True`` if any repeat finished with ``'abort'``.

        Returns
        -------
        bool
        """
        return any(e.finish_reason == "abort" for e in self.repeats)


@dataclass
class RepeatRequestState:
    """Tracks the state of a single repeat request.

    Attributes
    ----------
    id : int
        Repeat index.
    generated : list of int
        Accumulated generated tokens.
    logprob : list of float
        Accumulated log probabilities.
    status : RollStatus
    """
    id: int
    generated: List[int] = field(default_factory=list)
    logprob: List[float] = field(default_factory=list)
    status: RollStatus = RollStatus.PENDING

    def update(self, response: RepeatResponse):
        """Update state with a new partial response.

        Parameters
        ----------
        response : RepeatResponse
        """
        self.generated.extend(response.generated)
        self.logprob.extend(response.logprob)

        if response.finish_reason == "abort":
            self.status = RollStatus.ABORTED
        elif response.finish_reason == "length":
            self.status = RollStatus.TRUNCATED
        else:
            assert response.finish_reason == "stop"
            self.status = RollStatus.COMPLETED

    def is_done(self):
        """``True`` if completed or truncated.

        Returns
        -------
        bool
        """
        # truncated is also considerd done for now
        return (self.status == RollStatus.COMPLETED) or (self.status == RollStatus.TRUNCATED)

    def is_truncated(self):
        """``True`` if generation was truncated by length.

        Returns
        -------
        bool
        """
        return self.status == RollStatus.TRUNCATED

    def started(self):
        """Number of tokens generated so far.

        Returns
        -------
        int
        """
        return len(self.generated)

    def is_aborted(self):
        """``True`` if the repeat was aborted.

        Returns
        -------
        bool
        """
        return self.status == RollStatus.ABORTED


@dataclass
class RolloutRequestGroup:
    """Manage data and state transitions of a rollout request group.

    State machine::

        pending ---------> rolling ------> training
          |                  |
          |                  |
          <------------------+

    Attributes
    ----------
    batch_data : dict
        Input data for the request.
    cache_keys : list
        Keys to hide/expose around sampling.
    status : RollStatus
    repeats : list of RepeatRequestState
    cached : dict
        Temporarily cached fields.
    """
    batch_data: Dict
    cache_keys: List
    status: RollStatus = RollStatus.PENDING
    repeats: List[RepeatRequestState] = field(default_factory=list)
    cached: Dict = field(default_factory=dict)

    def __getstate__(self):
        """Custom pickle: exclude cached data not needed by sampler/gen-RM."""
        state = self.__dict__.copy()
        del state["cached"]
        # picle for sample
        if not self.is_done():
            repeats = state["repeats"]
            repeats = [e for e in repeats if not e.is_done()]
        return state

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)

    def build_repeats(self, repeat_num=1):
        """Append a new repeat request state.

        Parameters
        ----------
        repeat_num : int, optional
            Unused (reserved).

        Returns
        -------
        RolloutRequestGroup
            Self, for chaining.
        """
        self.repeats.append(RepeatRequestState(id=len(self.repeats)))
        return self

    def hide_fields_before_rollout(self):
        """Move cache-key fields from ``batch_data`` to ``cached``.

        Returns
        -------
        RolloutRequestGroup
            Self, for chaining.
        """
        for key in self.cache_keys:
            if key in self.batch_data:
                self.cached[key] = self.batch_data.pop(key)
        return self

    def expose_fileds_after_rollout(self):
        """Restore cache-key fields from ``cached`` back to ``batch_data``.

        Returns
        -------
        RolloutRequestGroup
            Self, for chaining.
        """
        for key in self.cache_keys:
            if key in self.cached:
                self.batch_data[key] = self.cached.pop(key)
        return self

    def _check_status_update(self):
        if all(r.is_done() for r in self.repeats):
            self.status = RollStatus.COMPLETED
        elif any(r.started() for r in self.repeats):
            self.status = RollStatus.ABORTED

    def update(self, res: RolloutGroupResponse):
        """Apply a group response to all repeat states.

        Parameters
        ----------
        res : RolloutGroupResponse

        Returns
        -------
        RolloutRequestGroup
            Self, for chaining.
        """
        for r in res.repeats:
            self.repeats[r.id].update(r)
        self._check_status_update()
        return self

    def is_done(self):
        """``True`` if the group has completed."""
        return self.status == RollStatus.COMPLETED

    def is_started(self):
        """``True`` if rolling has started."""
        return self.status != RollStatus.PENDING

    def is_aborted(self):
        """``True`` if the group was aborted."""
        return self.status == RollStatus.ABORTED

    def abort(self):
        """Abort the group without calling ``gen_rollouts``.

        Returns
        -------
        RolloutGroupResponse
            Response with all repeats marked as aborted.
        """
        response = RolloutGroupResponse()
        for r in self.repeats:
            response.repeats.append(RepeatResponse(id=r.id, finish_reason="abort"))
        return response

    async def gen_rollouts(self, args, engine, sampling_params, tokenizer=None):
        """Generate rollout completions asynchronously.

        Parameters
        ----------
        args : object
            Generation arguments.
        engine : object
            Inference engine with ``async_generate``.
        sampling_params : object
            Sampling parameters (temperature, top_p, etc.).
        tokenizer : object, optional
            Unused in current impl.

        Returns
        -------
        RolloutGroupResponse
            Aggregated generation results.
        """
        imgs_np_array_list = self.batch_data["imgs_np_array_list"]
        tokens_for_gen = self.batch_data.get("tokens_for_gen", [])
        prompt_texts_or_ids = []
        raw_images = []
        labels = []
        assert len(imgs_np_array_list) == 1
        imgs_np_array = imgs_np_array_list[0]
        imgs = None
        if imgs_np_array is not None:
            imgs = [Image.fromarray(img) for img in imgs_np_array]
        prompt = tokens_for_gen[0].tolist()
        assert len(self.repeats) == 1

        max_new_tokens = sampling_params.max_new_tokens

        async def task_func(r, sampling_params):
            sampling_params_tmp = copy.copy(sampling_params)
            assert max_new_tokens > len(r.generated)
            sampling_params_tmp.max_new_tokens = max_new_tokens - len(r.generated)
            input_ids = prompt + r.generated
            res = await engine.async_generate(
                input_ids=prompt,
                sampling_params=sampling_params_tmp.__dict__,
                image_data=imgs,
                return_logprob=True
            )
            output_ids = res['output_ids']
            finish_reason = res['meta_info']['finish_reason']['type']
            output_token_logprobs = [e[0] for e in res['meta_info']['output_token_logprobs']]
            # return_logprob for tis
            response = RepeatResponse(
                id=r.id,
                finish_reason=finish_reason,
                generated=output_ids,
                logprob=output_token_logprobs
            )
            return response

        tasks = []
        for r in self.repeats:
            task = task_func(r, sampling_params)
            tasks.append(task)

        res = await asyncio.gather(*tasks)
        res = RolloutGroupResponse(repeats=res)
        return res
