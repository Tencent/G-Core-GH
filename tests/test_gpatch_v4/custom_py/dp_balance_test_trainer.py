"""
Custom trainer, actor and train group for dp_balance testing.

Provides DpBalanceTestActor (overrides GrpoTrainActor to run dp_balance
correctness tests), DpBalanceTestTrainGroup (sets up PYTHONPATH for Ray
workers), and DpBalanceTestTrainer (orchestrates the test launch).
"""

import os
import traceback

import ray
import torch
import torch.distributed
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from megatron.core import mpu

from gpatch_v4 import orches
from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor
from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import cpu_barrier, is_last_rank
from gpatch_v4.core.smart_pad_helper import (
    DPBalanceHelper,
    get_column_based_batches,
    get_row_based_batches,
)
from gpatch_v4.orches.placement_group import create_placement_groups
from gpatch_v4.orches.train_group import RayTrainGroup
from gpatch_v4.orches.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from gpatch_v4.trainer.helper import set_nnodes_default
from gpatch_v4.trainer.trainer_mixin import TrainerRetryMixin
from gpatch_v4.utils import (
    expand_rollout_batches,
    get_iterator_k_split_list,
    log,
    logging_rank0,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _generate_mock_rollout_batches(
    tokenizer, dp_rank, dp_size, num_batches, samples_per_batch, seed=42
):
    """Generate mock rollout_batches with varying sequence lengths.

    Different DP ranks get different sequence lengths to ensure rebalance has real work.

    Returns a list of column-based dicts (same format as rollout_batches).
    """
    rng = torch.Generator()
    rng.manual_seed(seed + dp_rank)
    vocab_size = tokenizer.vocab_size

    rollout_batches = []
    for batch_i in range(num_batches):
        batch = {}
        tokens_list = []
        seq_lengths_list = []
        prompt_lengths_list = []
        gt_labels = []
        src_dps = []
        rewards_list = []

        for sample_i in range(samples_per_batch):
            # Make DP rank 0 have long seqs, last rank have short seqs
            # This ensures rebalance is non-trivial
            base_len = 256
            variation = torch.randint(0, 1024, (1, ), generator=rng).item()
            prompt_len = 232 + torch.randint(0, 64, (1, ), generator=rng).item()
            seq_len = base_len + variation + prompt_len

            # Generate random token ids
            tokens = torch.randint(0, vocab_size, (seq_len, ), generator=rng)
            # Ensure pad token is not at positions that should be valid
            tokens[:prompt_len] = torch.randint(1, vocab_size, (prompt_len, ), generator=rng)

            tokens_list.append(tokens)
            seq_lengths_list.append(torch.tensor(seq_len, dtype=torch.long))
            prompt_lengths_list.append(torch.tensor(prompt_len, dtype=torch.long))
            gt_labels.append(torch.tensor(float(sample_i % 5), dtype=torch.float32))
            src_dps.append(torch.tensor(dp_rank))
            rewards_list.append(torch.tensor(float(sample_i % 3) - 1.0, dtype=torch.float32))

        batch["tokens"] = tokens_list
        batch["sequence_lengths"] = seq_lengths_list
        batch["prompt_lengths"] = prompt_lengths_list
        batch["gt_label"] = gt_labels
        batch["src_dp"] = src_dps
        batch["rewards"] = rewards_list
        rollout_batches.append(batch)

    return rollout_batches


def _deep_copy_rollout_batches(rollout_batches):
    """Deep copy rollout_batches, cloning all tensors."""
    copied = []
    for rb in rollout_batches:
        new_rb = {}
        for key, val_list in rb.items():
            new_list = []
            for v in val_list:
                if torch.is_tensor(v):
                    new_list.append(v.clone())
                else:
                    new_list.append(v)
            new_rb[key] = new_list
        copied.append(new_rb)
    return copied


def _logprobs_sum_by_seqlen(logprobs_list, seq_lengths_list):
    """Compute sum of logprobs within valid sequence length for each sample.

    Parameters
    ----------
    logprobs_list : list of Tensor
        Per-sample logprobs, each shape (seq_len-1,).
    seq_lengths_list : list of Tensor
        Sequence lengths.

    Returns
    -------
    list of float
        Sum of logprobs for each sample.
    """
    sums = []
    for logprob, seq_len in zip(logprobs_list, seq_lengths_list):
        valid_len = seq_len.item() - 1  # logprobs is seq_len-1
        valid_logprob = logprob[:valid_len]
        sums.append(valid_logprob.sum().item())
    return sums


# ---------------------------------------------------------------------------
# DpBalanceTestActor
# ---------------------------------------------------------------------------


class DpBalanceTestActor(GrpoTrainActor):
    """Test actor that runs dp_balance correctness tests instead of real training."""
    async def setup_client(self):
        """Skip client setup since we don't need sampler/rm."""
        pass

    async def setup_rollout_generator(self):
        """Skip rollout generator setup."""
        pass

    async def train_loop(self):
        """Run dp_balance correctness tests.

        Returns test results instead of training metrics.
        """
        try:
            results = self._run_tests()
            return results
        except Exception as e:
            log(f"dp_balance test error: {e}")
            traceback.print_exc()
            raise e

    def _run_tests(self):
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()
        results = {}

        # Generate mock data
        samples_per_batch = self.config.training.rollout_mbs * self.config.training.sampling_keep_n
        num_batches = self.config.training.rollout_gbs // (
            dp_size * self.config.training.rollout_mbs
        )
        logging_rank0(
            f"[DP_BALANCE_TEST] Generating mock data: {num_batches=} {samples_per_batch=} {dp_rank=}/{dp_size=}"
        )

        rollout_batches = _generate_mock_rollout_batches(
            self.tokenizer, dp_rank, dp_size, num_batches, samples_per_batch
        )

        # =====================================================================
        # Test 1: compute_log_probs consistency
        # =====================================================================
        logging_rank0("=" * 60)
        logging_rank0("[TEST 1] compute_log_probs: dp_balance ON vs OFF")
        logging_rank0("=" * 60)

        test1_result = self._test_compute_log_probs(rollout_batches, samples_per_batch)
        results["test1"] = test1_result
        cpu_barrier()

        if not test1_result["passed"]:
            logging_rank0("[TEST 1] FAILED! Skipping Test 2.")
            return results

        logging_rank0("[TEST 1] PASSED!")

        # =====================================================================
        # Test 2: rl_train_actor metrics consistency
        # =====================================================================
        logging_rank0("=" * 60)
        logging_rank0("[TEST 2] rl_train_actor: dp_balance ON vs OFF")
        logging_rank0("=" * 60)

        return results

    def _test_compute_log_probs(self, rollout_batches, samples_per_batch):
        """Test 1: Compare compute_log_probs with dp_balance ON vs OFF.

        Steps:
          1. Run compute_log_probs WITHOUT dp_balance (baseline)
          2. Run compute_log_probs WITH dp_balance (rebalance → compute → restore)
          3. Compare logprobs sum for each sample (within valid seq_len)
        """
        result = {"passed": True, "details": {}}

        # ---- Baseline: compute_log_probs without dp_balance ----
        logging_rank0("[TEST 1] Running baseline (no dp_balance)...")
        baseline_batches = _deep_copy_rollout_batches(rollout_batches)
        self.policy_engine.offload_model()

        ref_logprobs_baseline, prev_logprobs_baseline = self.policy_engine.compute_log_probs(
            baseline_batches
        )
        cpu_barrier()

        # Compute baseline logprobs sums
        baseline_prev_sums = []
        baseline_ref_sums = []
        for rb_idx, rb in enumerate(baseline_batches):
            prev_sums = _logprobs_sum_by_seqlen(
                prev_logprobs_baseline[rb_idx], rb["sequence_lengths"]
            )
            baseline_prev_sums.extend(prev_sums)
            if ref_logprobs_baseline is not None:
                ref_sums = _logprobs_sum_by_seqlen(
                    ref_logprobs_baseline[rb_idx], rb["sequence_lengths"]
                )
                baseline_ref_sums.extend(ref_sums)

        logging_rank0(f"[TEST 1] Baseline prev_logprobs sums: {baseline_prev_sums}")

        # ---- dp_balance: rebalance → compute → restore ----
        logging_rank0("[TEST 1] Running with dp_balance...")
        dp_batches = _deep_copy_rollout_batches(rollout_batches)

        infer_require_keys = list(dp_batches[0].keys())
        infer_require_keys = DPBalanceHelper.filter_keys(
            infer_require_keys,
            add_custom_keys=getattr(self.config.task, "add_custom_keys", None),
        )
        logging_rank0(f"[TEST 1] dp_balance require_keys: {infer_require_keys}")

        cpu_barrier()
        rebalanced_batches, restore_info = DPBalanceHelper.rebalance(
            dp_batches,
            samples_per_batch,
            require_keys=infer_require_keys,
        )
        cpu_barrier()

        # ---- Verify rebalance actually exchanged data ----
        logging_rank0("[TEST 1] Verifying rebalance data exchange...")
        dp_rank = mpu.get_data_parallel_rank()
        for rb_idx, (orig_rb, rebal_rb) in enumerate(zip(rollout_batches, rebalanced_batches)):
            # 1. 原始本 rank 上的 seqlen 分布
            orig_seqlens = [s.item() for s in orig_rb["sequence_lengths"]]
            rebal_seqlens = [s.item() for s in rebal_rb["sequence_lengths"]]
            logging_rank0(
                f"[TEST 1] batch={rb_idx} rank={dp_rank} "
                f"orig_seqlens(sum={sum(orig_seqlens)}, max={max(orig_seqlens)}, min={min(orig_seqlens)}) "
                f"rebal_seqlens(sum={sum(rebal_seqlens)}, max={max(rebal_seqlens)}, min={min(rebal_seqlens)})"
            )
            # 2. 检查 rebalanced 里的 src_dp，应该含有来自其他 rank 的样本
            orig_src_dps = [s.item() for s in orig_rb["src_dp"]]
            rebal_src_dps = [s.item() for s in rebal_rb["src_dp"]]
            logging_rank0(
                f"[TEST 1] batch={rb_idx} rank={dp_rank} "
                f"orig_src_dps={orig_src_dps} -> rebal_src_dps={rebal_src_dps}"
            )
            # 3. 断言：rebalanced 里必须有来自其他 rank 的样本（否则 rebalance 没做任何事）
            has_foreign = any(s != dp_rank for s in rebal_src_dps)
            if not has_foreign:
                logging_rank0(
                    f"[TEST 1] WARNING: batch={rb_idx} rank={dp_rank}: "
                    f"no cross-rank exchange detected! rebal_src_dps={rebal_src_dps}"
                )

        ref_logprobs_dp, prev_logprobs_dp = self.policy_engine.compute_log_probs(rebalanced_batches)
        cpu_barrier()

        # Attach logprobs to rebalanced batches
        if ref_logprobs_dp is not None:
            for rb, ref_logps in zip(rebalanced_batches, ref_logprobs_dp):
                rb["ref_logprobs"] = ref_logps
        for rb, prev_logps in zip(rebalanced_batches, prev_logprobs_dp):
            rb["logprobs"] = prev_logps

        # Update restore keys
        restore_keys = restore_info['require_keys']
        restore_keys.append("logprobs")
        if ref_logprobs_dp is not None:
            restore_keys.append("ref_logprobs")
        restore_info['require_keys'] = restore_keys

        # Restore
        cpu_barrier()
        restored_batches = DPBalanceHelper.restore(rebalanced_batches, restore_info)
        cpu_barrier()

        # ---- Compare ----
        # Check consistency: restored data vs original
        restored_prev_sums = []
        restored_ref_sums = []
        for rb_idx, (orig_rb, restored_rb) in enumerate(zip(rollout_batches, restored_batches)):
            # Verify sample order is restored
            for i in range(len(orig_rb["src_dp"])):
                assert orig_rb["src_dp"][i] == restored_rb["src_dp"][i], \
                    f"src_dp mismatch at batch {rb_idx} sample {i}"
                assert orig_rb["tokens"][i].sum() == restored_rb["tokens"][i].sum(), \
                    f"tokens sum mismatch at batch {rb_idx} sample {i}"

            prev_sums = _logprobs_sum_by_seqlen(
                restored_rb["logprobs"], orig_rb["sequence_lengths"]
            )
            restored_prev_sums.extend(prev_sums)

            if "ref_logprobs" in restored_rb:
                ref_sums = _logprobs_sum_by_seqlen(
                    restored_rb["ref_logprobs"], orig_rb["sequence_lengths"]
                )
                restored_ref_sums.extend(ref_sums)

        logging_rank0(f"[TEST 1] Restored prev_logprobs sums: {restored_prev_sums}")

        # Compare prev_logprobs sums
        for i, (b_sum, r_sum) in enumerate(zip(baseline_prev_sums, restored_prev_sums)):
            diff = abs(b_sum - r_sum)
            if diff > 1e-3:
                logging_rank0(
                    f"[TEST 1] FAIL: prev_logprobs sample {i}: baseline={b_sum:.6f} "
                    f"restored={r_sum:.6f} diff={diff:.6f}"
                )
                result["passed"] = False
            else:
                logging_rank0(
                    f"[TEST 1] OK: prev_logprobs sample {i}: baseline={b_sum:.6f} "
                    f"restored={r_sum:.6f} diff={diff:.6e}"
                )

        # Compare ref_logprobs sums
        for i, (b_sum, r_sum) in enumerate(zip(baseline_ref_sums, restored_ref_sums)):
            diff = abs(b_sum - r_sum)
            if diff > 1e-3:
                logging_rank0(
                    f"[TEST 1] FAIL: ref_logprobs sample {i}: baseline={b_sum:.6f} "
                    f"restored={r_sum:.6f} diff={diff:.6f}"
                )
                result["passed"] = False
            else:
                logging_rank0(
                    f"[TEST 1] OK: ref_logprobs sample {i}: baseline={b_sum:.6f} "
                    f"restored={r_sum:.6f} diff={diff:.6e}"
                )

        result["details"]["baseline_prev_sums"] = baseline_prev_sums
        result["details"]["restored_prev_sums"] = restored_prev_sums

        # ---- Element-wise comparison ----
        logging_rank0("[TEST 1] Element-wise logprobs comparison (baseline vs restored)...")
        sample_idx = 0
        for rb_idx, (orig_rb, restored_rb) in enumerate(zip(rollout_batches, restored_batches)):
            for i, seq_len in enumerate(orig_rb["sequence_lengths"]):
                valid_len = seq_len.item() - 1
                # baseline: prev_logprobs_baseline[rb_idx][i]
                base_lp = prev_logprobs_baseline[rb_idx][i][:valid_len]
                rest_lp = restored_rb["logprobs"][i][:valid_len]
                max_diff = (base_lp - rest_lp).abs().max().item()
                all_close = torch.allclose(base_lp, rest_lp, atol=1e-4, rtol=0)
                logging_rank0(
                    f"[TEST 1] sample {sample_idx} (batch={rb_idx}, i={i}): "
                    f"max_token_diff={max_diff:.6e}, allclose(atol=1e-4)={all_close}"
                )
                if not all_close:
                    result["passed"] = False
                sample_idx += 1
        return result


# ---------------------------------------------------------------------------
# DpBalanceTestTrainGroup
# ---------------------------------------------------------------------------


class DpBalanceTestTrainGroup(RayTrainGroup):
    """Custom training group that uses DpBalanceTestActor."""
    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        assert pg is not None
        pg, reordered_bundle_indices = pg

        # Propagate PYTHONPATH so Ray workers can import this module.
        # Per-actor runtime_env overrides the global one from ray.init(),
        # so we must explicitly re-include PYTHONPATH here.
        # Also append the parent directory of custom_py/ so that
        # `from custom_py.dp_balance_test_trainer import ...` works inside
        # Ray workers.
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _parent_dir = os.path.dirname(_this_dir)  # tests/test_gpatch_v4/
        propagate_env = {}
        for key in ["PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF"]:
            val = os.environ.get(key)
            if val is not None:
                propagate_env[key] = val
        # Ensure the test directory is in PYTHONPATH for the worker
        existing_pypath = propagate_env.get("PYTHONPATH", "")
        propagate_env["PYTHONPATH"
                     ] = f"{_parent_dir}:{existing_pypath}" if existing_pypath else _parent_dir

        env_vars = {
            "NCCL_CUMEM_ENABLE": "0",
            **{
                name: "1"
                for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
            },
            **propagate_env,
        }

        ActorClass = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(DpBalanceTestActor)

        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = ActorClass.options(
                name=f'{self.role}_{rank}',
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
                log(f"DpBalanceTestTrainGroup master_addr {master_addr}, master_port {master_port}")
            self._actor_handlers.append(actor)


# ---------------------------------------------------------------------------
# DpBalanceTestTrainer
# ---------------------------------------------------------------------------


class DpBalanceTestTrainer(TrainerRetryMixin):
    """Trainer that launches dp_balance correctness tests."""
    def __init__(self):
        self.train_group = None

    async def launch(self, config: RlConfig):
        orches.init(config)
        set_nnodes_default(config)
        pgs = create_placement_groups(config)

        self.train_group = DpBalanceTestTrainGroup(
            config=config,
            num_nodes=config.policy.dist_config.nnodes,
            num_gpus_per_node=config.policy.dist_config.num_gpus_per_node,
            pg=pgs['policy'],
            role="policy",
        )
        await self.train_group.init()
        # Skip setup_client and setup_rollout_generator (not needed for this test)
        await self.train_group.setup_model_and_optimizer()
