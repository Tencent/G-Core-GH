import abc
import asyncio
import time
import traceback
from contextlib import nullcontext
from typing import Any, Dict, List

from transformers import AutoTokenizer

from gpatch_v4.client import GenRmClient, SamplerClient
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.rollout_generator.async_rollout.mixin import ColocateAgentMixin
from gpatch_v4.utils import import_fn_from_path, log, log_debug


class BaseAgentLoopActor(ColocateAgentMixin, abc.ABC):
    """Abstract base for per-microbatch pipeline actors.

    Subclass to inject a custom agent loop into the ``RolloutController``
    worker pool. The controller calls :meth:`setup` once after construction
    then repeatedly invokes :meth:`agent_loop`.

    For colocate mode the controller partitions microbatches across N agents.
    All agent types reuse :meth:`agent_loop` with batched inputs and
    ``use_colocate=True``. Agents coordinate GPU lifecycle
    (``mark_ppo_step_begin/end``) via a gloo process group: only the leader
    (rank 0) sends RPCs while all ranks synchronise at barriers.

    Parameters
    ----------
    config : RlConfig
    worker_id : int
    """
    def __init__(self, config: RlConfig, worker_id: int):
        self.config = config
        self.training_config = config.training
        self.worker_id = worker_id
        self._init_colocate_state()

    @abc.abstractmethod
    async def setup(self):
        """One-time async initialisation after the Ray actor is created."""
        ...

    @abc.abstractmethod
    async def agent_loop(
        self,
        cleaned_data: Dict[str, Any],
        ppo_step: int,
        microbatch_idx: int,
        sample_idx: int,
    ) -> List[Dict[str, List[Any]]]:
        """Execute the per-microbatch pipeline and return rollout data.

        Parameters
        ----------
        cleaned_data : dict
        ppo_step : int
        microbatch_idx : int
        sample_idx : int

        Returns
        -------
        list[dict]
            One or more rollout batches; each dict maps string keys to
            equal-length lists.
        """
        ...


class AgentLoopActor(BaseAgentLoopActor):
    """Default agent-loop actor: sampler generation + gen_rm scoring.

    Each instance holds its own ``SamplerClient`` and ``GenRmClient`` in
    standalone mode (``dp_rank=0, dp_size=1``). Multiple actors can coexist
    safely because the clients are stateless RPC proxies and routing is
    determined by the ``sidx`` argument.
    """
    def __init__(self, config: RlConfig, worker_id: int):
        super().__init__(config, worker_id)
        self.sampler_client = None
        self.gen_rm_client = None
        self.tokenizer = None
        self.external_reward = None

    async def setup(self):
        """Create RPC clients.  Must be called once after actor creation."""
        self.sampler_client = SamplerClient(
            self.config, dp_rank=0, dp_size=1, skip_init_ipc_meta=True
        )
        if self.training_config.use_gen_rm_reward:
            self.gen_rm_client = GenRmClient(self.config, dp_rank=0, dp_size=1)

        if self.training_config.use_external_reward:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.policy.hf_tokenizer_path,
                use_fast=getattr(self.training_config, "use_fast_tokenizer", True),
                trust_remote_code=True,
            )
            self._setup_external_reward()

        log(
            f"[AgentLoopActor-{self.worker_id}] setup done, "
            f"external_reward={type(self.external_reward).__name__ if self.external_reward else None}"
        )

    def _setup_external_reward(self):
        """Instantiate the external reward class from config."""
        external_reward_config = self.config.external_reward
        assert external_reward_config.reward_info is not None, (
            "external_reward.reward_info must be set when use_external_reward=True"
        )
        assert len(external_reward_config.reward_info
                  ) == 1, ("Currently only one external reward is supported")
        rm_info = external_reward_config.reward_info[0]
        assert rm_info.reward_py_path is not None, "reward_py_path must be set"
        assert rm_info.reward_cls_name is not None, "reward_cls_name must be set"

        reward_cls = import_fn_from_path(rm_info.reward_py_path, rm_info.reward_cls_name)
        assert hasattr(reward_cls, 'calc_external_reward'
                      ), (f"{rm_info.reward_cls_name} must implement calc_external_reward")
        self.external_reward = reward_cls(config=self.config, tokenizer=self.tokenizer)
        log(
            f"[AgentLoopActor-{self.worker_id}] external reward initialized: {rm_info.reward_cls_name}"
        )

    async def generate_batches(
        self,
        cleaned_batches: List[Dict[str, Any]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Generate rollouts by firing all requests, then awaiting all results."""
        sampler_idx = 0
        repeat_n = self.training_config.sampling_repeat_n
        if not cleaned_batches and not use_colocate:
            return []

        ctx = self.sampler_phase(ppo_step) if use_colocate else nullcontext()
        async with ctx:
            refs = await asyncio.gather(
                *[
                    self.sampler_client.fire_generate(
                        sampler_idx,
                        ppo_step,
                        sidx,
                        cleaned_data,
                        repeat_n,
                        load_aware=self.training_config.load_aware_sampler_routing,
                    ) for cleaned_data, sidx in zip(cleaned_batches, sample_indices)
                ]
            )
            rbs = list(
                await asyncio.gather(*[self.sampler_client.await_generate(ref) for ref in refs])
            )
        return rbs

    async def _score_one_gen_rm(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        sample_indices: List[int],
        rm_idx: int,
    ) -> List[Dict[str, List[Any]]]:
        """Score one gen-RM for all rollout batches."""
        if not rbs:
            return []
        return list(
            await asyncio.gather(
                *[
                    self.gen_rm_client.generate_rewards(
                        rm_idx,
                        ppo_step,
                        sidx,
                        rb,
                    ) for sidx, rb in zip(sample_indices, rbs)
                ]
            )
        )

    async def score_gen_rm_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Apply gen-RM rewards, optionally wrapped in colocate lifecycle phases."""
        if not self.training_config.use_gen_rm_reward:
            return rbs
        phase_ctx = self.gen_rm_all_phase(ppo_step) if use_colocate else nullcontext()
        async with phase_ctx:
            all_results = await asyncio.gather(
                *[
                    self._score_one_gen_rm(rbs, ppo_step, sample_indices, rm_idx)
                    for rm_idx in range(self.gen_rm_client.num_rms)
                ]
            )
        for rm_results in all_results:
            for rb, result in zip(rbs, rm_results):
                rb.update(result)
        return rbs

    async def score_external_reward_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
    ) -> List[Dict[str, List[Any]]]:
        """Apply CPU external rewards for all rollout batches."""
        if self.external_reward is None:
            return rbs
        t0 = time.time()
        log_debug(f"[external_reward] BEGIN worker_id={self.worker_id} ppo_step={ppo_step}")
        all_updates = await asyncio.gather(
            *[self.external_reward.calc_external_reward([rb], ppo_step) for rb in rbs]
        )
        for rb, updates in zip(rbs, all_updates):
            assert len(updates) == 1, (
                f"calc_external_reward must return one update per input batch; got {len(updates)}"
            )
            rb.update(updates[0])
        log_debug(
            f"[external_reward] END worker_id={self.worker_id} "
            f"ppo_step={ppo_step} elapsed_s={time.time() - t0:.3f}"
        )
        return rbs

    async def score_rollout_batches(
        self,
        rbs: List[Dict[str, List[Any]]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Shared reward pipeline for default single-turn rollouts."""
        rbs = await self.score_gen_rm_batches(
            rbs,
            ppo_step,
            sample_indices,
            use_colocate=use_colocate,
        )
        return await self.score_external_reward_batches(rbs, ppo_step)

    async def _batch_loop(
        self,
        cleaned_batches: List[Dict[str, Any]],
        ppo_step: int,
        sample_indices: List[int],
        use_colocate: bool,
    ) -> List[Dict[str, List[Any]]]:
        """Shared default single-turn pipeline.

        ``use_colocate`` selects only the sampler dispatch style and
        whether gen-RM scoring is wrapped by colocate lifecycle phases.
        The rest of the generation/reward pipeline is shared.
        """
        rbs = await self.generate_batches(
            cleaned_batches,
            ppo_step,
            sample_indices,
            use_colocate=use_colocate,
        )

        if self.worker_id == 0:
            log(
                f"[AgentLoopActor-{self.worker_id}] "
                f"_batch_loop done: {ppo_step=}, "
                f"mode={'colocate' if use_colocate else 'disaggregated'}, "
                f"num_microbatches={len(cleaned_batches)}, "
                f"num_rollout_batches={len(rbs)}"
            )
        rbs = await self.score_rollout_batches(
            rbs,
            ppo_step,
            sample_indices,
            use_colocate=use_colocate,
        )
        return rbs

    async def agent_loop(
        self,
        cleaned_data: Dict[str, Any] | List[Dict[str, Any]],
        ppo_step: int,
        microbatch_idx: int | List[int],
        sample_idx: int | List[int],
        use_colocate: bool = False,
    ) -> List[Dict[str, List[Any]]]:
        """Default single-turn entry for both disaggregated and colocate modes.

        Disaggregated callers pass one microbatch. Colocate callers pass this
        agent's list of microbatches and set ``use_colocate=True`` so sampler
        and gen-RM phases are lifecycle-synchronised.
        """
        try:
            if isinstance(cleaned_data, list):
                cleaned_batches = cleaned_data
                sample_indices = sample_idx
            else:
                cleaned_batches = [cleaned_data]
                sample_indices = [sample_idx]
            assert isinstance(sample_indices,
                              list), ("sample_idx must be a list when cleaned_data is a list")
            return await self._batch_loop(
                cleaned_batches,
                ppo_step,
                sample_indices,
                use_colocate=use_colocate,
            )
        except Exception as e:
            traceback.print_exc()
            raise e
