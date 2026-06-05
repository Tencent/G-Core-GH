import asyncio
import copy
import inspect
import os
import traceback
import uuid
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed

from megatron.core import mpu
from megatron.core import parallel_state as mcore_parallel_state

from gpatch_v4.actor.mixin import RlTrainerMixin, TokenizerMixin
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_mp_and_cp_head,
)
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.utils import (
    BroadcastUtils,
    clear_memory,
    destroy_process_groups,
    format_config,
    import_fn_from_path,
    log,
    logging_rank0,
    monkey_patch_torch_dist,
    reload_process_groups,
    split_dict_list_by_keys,
    sync_cuda_and_get_time,
)


class DistillTeacherActor(BaseActor, TokenizerMixin, RlTrainerMixin):
    async def init(self, config):
        super().init(config)

        if config.teacher.offload_process_group or config.training.offload_process_group:
            monkey_patch_torch_dist()

        initlize_parallel_state(config, config.teacher.dist_config)
        init_pg(config.teacher.dist_config)

        self.build_tokenizer()
        self.tokenizer = self.teacher_tokenizer
        self.device = torch.device(torch.cuda.current_device())

        self.validated_config()
        logging_rank0(f"{self.__class__.__name__} config {format_config(self.config)}")

        extra_args = {"policy_config": config.teacher, "tokenizer": self.tokenizer}
        self.teacher_engine = TrainingEngineFactory.get_training_engine(config, **extra_args)
        self.lock = asyncio.Lock()
        self.computed = False
        self.batching_reqs: List[Dict[str, Union[int, List[Any]]]] = []
        self.compute_logps_results: Dict[int, Dict[int, Dict]] = {}
        self.load_hf_config()

    def validated_config(self):
        teacher_config = self.config.teacher
        # auto set without_ref and without_optim
        teacher_config.without_ref = True
        teacher_config.without_optim = True
        assert teacher_config.without_ref is True, f"{teacher_config.without_ref=}"
        assert teacher_config.without_optim is True, f"{teacher_config.without_optim=}"

    async def setup_model(self):
        logging_rank0(f"begin setup_model...")
        # 因为只需要跑一波 forward 算 logps，所以其实内部不需要开 optimizer
        self.teacher_engine.setup_model_and_get_optimizer()
        logging_rank0(f"finished setup_model...")

    @property
    def _offload_process_group(self) -> bool:
        return self.config.teacher.offload_process_group or self.config.training.offload_process_group

    @staticmethod
    def _destroy_global_memory_buffer():
        """Release Megatron's GlobalMemoryBuffer to free GPU memory held by
        tensor-parallel / expert-parallel communication buffers."""
        mcore_parallel_state.destroy_global_memory_buffer()

    @staticmethod
    def _ensure_global_memory_buffer():
        """Recreate Megatron's GlobalMemoryBuffer if it was previously destroyed."""
        if mcore_parallel_state._GLOBAL_MEMORY_BUFFER is None:
            mcore_parallel_state._set_global_memory_buffer()

    async def sleep(self):
        self.teacher_engine.offload_model()
        self._destroy_global_memory_buffer()
        if self._offload_process_group:
            log(f"teacher destroy_process_groups", rank=0)
            destroy_process_groups()
        clear_memory()
        return {"ret": True}

    async def wake_up(self):
        if self._offload_process_group:
            reload_process_groups()
        self._ensure_global_memory_buffer()
        self.teacher_engine.onload_model()
        return {"ret": True}

    async def mark_ppo_step_begin(self, req_dict):
        assert len(self.batching_reqs) == 0, f"len of batching_reqs {len(self.batching_reqs)}"
        async with self.lock:
            self.computed = False
            if self._offload_process_group:
                reload_process_groups()
            self._ensure_global_memory_buffer()
            self.teacher_engine.onload_model()
        return {"ret": True}

    async def mark_ppo_step_end(self, req_dict):
        async with self.lock:
            self.teacher_engine.offload_model()
            self.batching_reqs = []
            self.compute_logps_results.clear()
            self._destroy_global_memory_buffer()

            if self._offload_process_group:
                log(f"teacher destroy_process_groups", rank=0)
                destroy_process_groups()
            clear_memory()
        return {"ret": True}

    async def issue_calc_logps(self, req_dict: Dict[str, Any]):
        assert self.computed is False
        async with self.lock:
            if is_mp_and_cp_head():
                self.batching_reqs.append(req_dict)
            else:
                self.batching_reqs.append({})
        return {"ret": True}

    async def get_calc_logps_result(self, req_dict: Dict[str, Any]):
        log(
            f"get_calc_logps_result dp {mpu.get_data_parallel_rank()} tp {mpu.get_tensor_model_parallel_rank()} pp {mpu.get_pipeline_model_parallel_rank()} {req_dict}",
            rank=0
        )
        actor_dp_rank = req_dict["actor_dp_rank"]
        ppo_step = req_dict["ppo_step"]
        sample_idx = req_dict["sample_idx"]
        log_prob_top_k = self.config.ppo.log_prob_top_k
        async with self.lock:
            if not self.computed:
                self.computed = True
                # broadcast batching_reqs

                batched_data = BroadcastUtils.broadcast_object_within_mp_and_cp(
                    self.batching_reqs, make_recursive_clone_in_case_of_view=True
                )

                req_meta_data, real_data = split_dict_list_by_keys(
                    batched_data, ['actor_dp_rank', 'sample_idx', 'ppo_step']
                )
                if not is_mp_and_cp_head():
                    # 非 mp_and_cp_head 节点前面不接受数据
                    self.batching_reqs = req_meta_data

                try:
                    fwd_kwargs = {}
                    if log_prob_top_k > 0:
                        # teacher 在 student 提供的 stu_topk_ids 上 gather log-probs。
                        fwd_kwargs["topk_gather_ids_key"] = "stu_topk_ids"
                    _, teacher_logps = self.teacher_engine.compute_log_probs(
                        real_data, **fwd_kwargs
                    )

                    # prevent seq_len across student batches between dp group
                    assert len(real_data) == len(
                        teacher_logps
                    ), f"len(real_data) {len(real_data)} len(teacher_logps) {len(teacher_logps)}"

                    for ri in range(len(real_data)):
                        data = real_data[ri]
                        sequence_lengths = data["sequence_lengths"]

                        for tj in range(len(teacher_logps[ri])):
                            target_len = int(sequence_lengths[tj].item()) - 1
                            entry = teacher_logps[ri][tj]
                            if log_prob_top_k > 0:
                                assert isinstance(entry, dict), f"unexpected entry type {type(entry)}"
                                entry["logprobs"] = entry["logprobs"][:target_len].detach().clone()
                                entry["gather_logprobs"] = (
                                    entry["gather_logprobs"][:target_len].detach().clone()
                                )
                            else:
                                assert entry.ndim == 1, f"teacher_logps[i][j].ndim {entry.shape}"
                                teacher_logps[ri][tj] = entry[:target_len].detach().clone()

                    for _, (batch, b_teacher_logps) in enumerate(
                        zip(self.batching_reqs, teacher_logps, strict=True)
                    ):
                        b_actor_dp_rank = batch["actor_dp_rank"]
                        b_sample_idx = batch["sample_idx"]
                        b_ppo_step = batch['ppo_step']

                        tmpd = self.compute_logps_results.setdefault(b_actor_dp_rank, {})
                        tmpdd = tmpd.setdefault(b_ppo_step, {})
                        if log_prob_top_k > 0:
                            tmpdd[b_sample_idx] = {
                                "teacher_logprobs": [d["logprobs"] for d in b_teacher_logps],
                                "teacher_on_stu_topk_logprobs":
                                    [d["gather_logprobs"] for d in b_teacher_logps],
                            }
                        else:
                            tmpdd[b_sample_idx] = {"teacher_logprobs": b_teacher_logps}

                    self.batching_reqs = []
                    clear_memory()
                except Exception as e:
                    log(f"compute_log_probs error {e}", rank=0)
                    traceback.print_exc()
                    import sys
                    sys.exit()

        assert actor_dp_rank in self.compute_logps_results, f"{self.compute_logps_results.keys()=} {actor_dp_rank=}"
        tmpd = self.compute_logps_results[actor_dp_rank][ppo_step]
        resp_dict = tmpd[sample_idx]
        return resp_dict

    async def issue_calc_logits(self, req_dict: Dict[str, Any]):
        assert self.computed is False
        async with self.lock:
            if is_mp_and_cp_head():
                self.batching_reqs.append(req_dict)
            else:
                self.batching_reqs.append({})
        return {"ret": True}

    async def get_calc_logits_result(self, req_dict: Dict[str, Any]):
        log(
            f"get_calc_logits_result dp {mpu.get_data_parallel_rank()} tp {mpu.get_tensor_model_parallel_rank()} pp {mpu.get_pipeline_model_parallel_rank()} {req_dict}",
            rank=0
        )
        actor_dp_rank = req_dict["actor_dp_rank"]
        ppo_step = req_dict["ppo_step"]
        sample_idx = req_dict["sample_idx"]
        async with self.lock:
            if not self.computed:
                self.computed = True
                batched_data = BroadcastUtils.broadcast_object_within_mp_and_cp(
                    self.batching_reqs, make_recursive_clone_in_case_of_view=True
                )

                req_meta_data, real_data = split_dict_list_by_keys(
                    batched_data, ['actor_dp_rank', 'sample_idx', 'ppo_step']
                )
                if not is_mp_and_cp_head():
                    # 因为非 mp_and_cp_head 节点前面不接受数据
                    self.batching_reqs = req_meta_data

                try:
                    _, teacher_logits = self.teacher_engine.compute_logits(real_data)
                    # prevent seq_len across student batches between dp group
                    assert len(real_data) == len(
                        teacher_logits
                    ), f"len(real_data) {len(real_data)} len(teacher_logits) {len(teacher_logits)}"

                    for ri in range(len(real_data)):
                        data = real_data[ri]
                        # FIXME (@yeazhao) 这个 data['tokens'] 已经是 1-D tensor了，再for就错了
                        # 不过这个函数目前基本是 deprecated 了，没怎么用
                        max_seq_len = max([token.shape[-1] for token in data['tokens']])
                        pad_to_multi_of = self.config.training.pad_to_mulitiple_of
                        max_token_len = (
                            (max_seq_len + pad_to_multi_of - 1) // pad_to_multi_of
                        ) * pad_to_multi_of

                        if teacher_logits is None or teacher_logits[ri] is None:
                            continue
                        for tj in range(len(teacher_logits[ri])):
                            t_logits = teacher_logits[ri][tj]
                            assert t_logits.ndim == 2, f"teacher_logits[i][j].ndim {t_logits.shape}"
                            teacher_logits[ri][tj] = t_logits[:max_token_len].detach().clone()

                    for _, (batch, b_teacher_logits) in enumerate(
                        zip(self.batching_reqs, teacher_logits, strict=True)
                    ):
                        b_actor_dp_rank = batch["actor_dp_rank"]
                        b_sample_idx = batch["sample_idx"]
                        b_ppo_step = batch['ppo_step']

                        tmpd = self.compute_logps_results.setdefault(b_actor_dp_rank, {})
                        tmpdd = tmpd.setdefault(b_ppo_step, {})
                        tmpdd[b_sample_idx] = {"teacher_logits": b_teacher_logits}

                    self.batching_reqs = []
                    clear_memory()
                except Exception as e:
                    log(f"compute_logits error {e}", rank=0)
                    traceback.print_exc()
                    import sys
                    sys.exit()

        assert actor_dp_rank in self.compute_logps_results
        tmpd = self.compute_logps_results[actor_dp_rank][ppo_step]
        resp_dict = tmpd[sample_idx]
        return resp_dict

    def test_ray_rpc_data(self, data_type, dtype, shape_meta):
        log(f"test_ray_rpc_data {data_type} {dtype}")
        b, s, v = shape_meta

        t1 = sync_cuda_and_get_time()
        data = torch.empty((b, s, v), dtype=dtype)
        t2 = sync_cuda_and_get_time()
        if data_type == "tensor":
            log(f"time {t2 - t1} {data.sum()}")
            return data
        elif data_type == "numpy":
            data = data.numpy()
            t3 = sync_cuda_and_get_time()
            log(f"time {t2 - t1} {t3 - t2} {data.sum()}")
            return data
        else:
            raise ValueError(f"Invalid data_type {data_type}")
