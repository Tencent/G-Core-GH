"""Thin public mixins for sampler weight-update factories."""
from __future__ import annotations

from gpatch_v4.client.update_weight_factory import (
    UpdateWeightContext,
    UpdateWeightFactory,
    get_update_weight_factory,
)
from gpatch_v4.utils import logging_rank0


class UpdateWeightMixin:
    """IPC and distributed weight-update entrypoints backed by a cached factory."""

    _cached_update_weight_factory: UpdateWeightFactory | None = None
    # Set by SamplerClient.build_update_from_tensor_meta (colocated IPC only).
    _ipc_gather_dst_rank = None
    _ipc_gather_group = None
    _ipc_target = None
    # Set by init_distributed_weight_group (disaggregated / NCCL path).
    _dist_weight_group = None
    _dist_weight_group_name = None

    def _build_update_weight_context(self) -> UpdateWeightContext:
        sampler_configs = self.config.sampler.infer_engine_configs
        engine_gpu_counts = [
            min(
                config.dist_config.tensor_model_parallel_size *
                config.dist_config.pipeline_model_parallel_size,
                config.dist_config.num_gpus_per_node,
            ) for config in sampler_configs
        ]
        return UpdateWeightContext(
            infer_backend=self.infer_backend,
            update_weight_max_size_bytes=self.update_weight_max_size_bytes,
            update_weight_use_bucketed_ipc=self.update_weight_use_bucketed_ipc,
            rpc_client_lst=self.rpc_client_lst,
            svr_cluster_num_per_sampler=self.svr_cluster_num_per_sampler,
            ipc_gather_dst_rank=self._ipc_gather_dst_rank,
            ipc_gather_group=self._ipc_gather_group,
            ipc_target=self._ipc_target,
            dist_weight_group=self._dist_weight_group,
            dist_weight_group_name=self._dist_weight_group_name,
            sampler_engine_gpu_counts=engine_gpu_counts,
            moe_deepgemm=bool(getattr(sampler_configs[0], "enable_deepep_moe", False))
            if sampler_configs else False,
            sglang_export_fp4_qdq=bool(getattr(self.config.policy, "sglang_export_fp4_qdq", False)),
            placement_type=self.config.placement_type,
            wake_up=self.wake_up,
            sleep=self.sleep,
        )

    def _sync_update_weight_context(self, context: UpdateWeightContext) -> None:
        """Refresh mutable transport state owned by the client onto a cached context."""
        context.ipc_gather_dst_rank = self._ipc_gather_dst_rank
        context.ipc_gather_group = self._ipc_gather_group
        context.ipc_target = self._ipc_target
        context.dist_weight_group = self._dist_weight_group
        context.dist_weight_group_name = self._dist_weight_group_name

    def _update_weight_factory(self) -> UpdateWeightFactory:
        factory = self._cached_update_weight_factory
        if factory is None:
            factory = get_update_weight_factory(
                context=self._build_update_weight_context(),
                model_arch=self.config.policy.model_arch,
            )
            self._cached_update_weight_factory = factory
        else:
            # dist / ipc group fields can change after init, destroy, or
            # build_update_from_tensor_meta; keep the cached context in sync.
            self._sync_update_weight_context(factory.context)
        return factory

    async def update_weights_by_ipc_handle(self, sampler_idx, model_engine, replace_zeros=False):
        return await self._update_weight_factory().update_weights_by_ipc_handle(
            sampler_idx, model_engine, replace_zeros
        )

    def update_weights_by_distributed(self, sampler_idx, model_engine, replace_zeros=False):
        return self._update_weight_factory().update_weights_by_distributed(
            sampler_idx, model_engine, replace_zeros
        )

    async def init_distributed_weight_group(self, group_name="weight_update_group"):
        factory = self._update_weight_factory()
        group, name = await factory.init_distributed_weight_group(group_name)
        self._dist_weight_group = group
        self._dist_weight_group_name = name
        self._sync_update_weight_context(factory.context)

    async def destroy_distributed_weight_group(self):
        factory = self._update_weight_factory()
        group, name = await factory.destroy_distributed_weight_group()
        self._dist_weight_group = group
        self._dist_weight_group_name = name
        self._sync_update_weight_context(factory.context)


class TestFuncMixin:
    async def test_generate(self, sampler_idx):
        prompts = [
            "Hello, what is your name?",
            "Who is the president of the United States?",
            "What is the capital of France?",
            "What is the future of AI?",
        ]
        if self._is_rpc_leader():
            response = await self._batch_rpc_call(
                sampler_idx, "test_generate", {"prompts": prompts}
            )
            logging_rank0(f"test_generate resp: {response}")

    async def test_save_engine_ckpt(self, sampler_idx, save_path):
        if self._is_rpc_leader():
            response = await self._batch_rpc_call(
                sampler_idx, "save_engine_ckpt", {"save_ckpt_dir": f"{save_path}"}
            )
            logging_rank0(f"test_save_engine_ckpt resp: {response}")
