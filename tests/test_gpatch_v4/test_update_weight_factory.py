"""Unit tests for update-weight factory selection and mixin ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gpatch_v4.client.mixin import UpdateWeightMixin
from gpatch_v4.client.update_weight_factory import (
    DeepSeekV4SglangUpdateWeightFactory,
    DeepSeekV4VllmUpdateWeightFactory,
    SglangUpdateWeightFactory,
    UpdateWeightContext,
    VllmUpdateWeightFactory,
    get_update_weight_factory,
)
from gpatch_v4.core.constants import MODEL_ARCH


def _make_sglang_context(**overrides):
    values = dict(
        infer_backend="sglang",
        update_weight_max_size_bytes=1024,
        update_weight_use_bucketed_ipc=False,
        rpc_client_lst=[],
        svr_cluster_num_per_sampler=[1],
        sampler_engine_gpu_counts=[2],
        placement_type="disaggregated",
    )
    values.update(overrides)
    return UpdateWeightContext(**values)


def _make_mixin_client(**overrides):
    class _Client(UpdateWeightMixin):
        def __init__(self):
            self.infer_backend = "sglang"
            self.update_weight_max_size_bytes = 1024
            self.update_weight_use_bucketed_ipc = False
            self.rpc_client_lst = []
            self.svr_cluster_num_per_sampler = [1]
            self.config = SimpleNamespace(
                placement_type="disaggregated",
                policy=SimpleNamespace(model_arch=MODEL_ARCH.QWEN3),
                sampler=SimpleNamespace(infer_engine_configs=[]),
            )
            self.wake_up = None
            self.sleep = None
            for key, value in overrides.items():
                setattr(self, key, value)

    return _Client()


class UpdateWeightFactoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = _make_sglang_context(
            svr_cluster_num_per_sampler=[], sampler_engine_gpu_counts=None
        )

    def test_selects_standard_sglang_factory(self):
        factory = get_update_weight_factory(
            context=self.context,
            model_arch=MODEL_ARCH.QWEN3,
        )

        self.assertIsInstance(factory, SglangUpdateWeightFactory)
        self.assertNotIsInstance(factory, DeepSeekV4SglangUpdateWeightFactory)

    def test_selects_dsv4_sglang_subclass(self):
        factory = get_update_weight_factory(
            context=self.context,
            model_arch=MODEL_ARCH.DEEPSEEK_V4,
        )

        self.assertIsInstance(factory, DeepSeekV4SglangUpdateWeightFactory)
        self.assertIsInstance(factory, SglangUpdateWeightFactory)

    def test_selects_standard_and_dsv4_vllm_factories(self):
        context = _make_sglang_context(
            infer_backend="vllm",
            update_weight_use_bucketed_ipc=True,
            svr_cluster_num_per_sampler=[],
            sampler_engine_gpu_counts=None,
        )
        standard = get_update_weight_factory(
            context=context,
            model_arch=MODEL_ARCH.QWEN3,
        )
        dsv4 = get_update_weight_factory(
            context=context,
            model_arch=MODEL_ARCH.DEEPSEEK_V4,
        )

        self.assertIsInstance(standard, VllmUpdateWeightFactory)
        self.assertNotIsInstance(standard, DeepSeekV4VllmUpdateWeightFactory)
        self.assertIsInstance(dsv4, DeepSeekV4VllmUpdateWeightFactory)

    def test_rejects_unknown_backend(self):
        context = _make_sglang_context(
            infer_backend="unknown",
            svr_cluster_num_per_sampler=[],
            sampler_engine_gpu_counts=None,
        )
        with self.assertRaisesRegex(AssertionError, "only sglang and vllm"):
            get_update_weight_factory(
                context=context,
                model_arch=MODEL_ARCH.QWEN3,
            )

    def test_context_and_factory_do_not_retain_client(self):
        self.assertNotIn("client", self.context.__dict__)
        factory = get_update_weight_factory(
            context=self.context, model_arch=MODEL_ARCH.QWEN3
        )
        self.assertFalse(hasattr(factory, "client"))

    def test_dsv4_sglang_uses_atomic_bucket_iterator(self):
        factory = get_update_weight_factory(
            context=self.context, model_arch=MODEL_ARCH.DEEPSEEK_V4
        )
        sentinel = object()
        weights = iter(())
        engine = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(fp4_qat=False)),
            export_weights=lambda: weights
        )
        with patch(
            "gpatch_v4.generation_backend.sglang_model_specific."
            "sglang_weight_update_dsv4.iter_sglang_dsv4_weight_buckets",
            return_value=sentinel,
        ) as iterator:
            self.assertIs(factory.iter_dsv4_update_buckets(engine, 1024), sentinel)
        iterator.assert_called_once_with(
            weights, 1024, moe_deepgemm=False, fp4_qat=False
        )

    def test_dsv4_sglang_reads_fp4_qat_from_model_config(self):
        factory = get_update_weight_factory(
            context=self.context, model_arch=MODEL_ARCH.DEEPSEEK_V4
        )
        sentinel = object()
        weights = iter(())
        engine = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(fp4_qat=True)),
            export_weights=lambda: weights,
        )
        with patch(
            "gpatch_v4.generation_backend.sglang_model_specific."
            "sglang_weight_update_dsv4.iter_sglang_dsv4_weight_buckets",
            return_value=sentinel,
        ) as iterator:
            self.assertIs(factory.iter_dsv4_update_buckets(engine, 1024), sentinel)
        iterator.assert_called_once_with(
            weights, 1024, moe_deepgemm=False, fp4_qat=True
        )

    def test_mixin_caches_factory_and_syncs_dist_group(self):
        client = _make_mixin_client()
        first = client._update_weight_factory()
        second = client._update_weight_factory()
        self.assertIs(first, second)

        client._dist_weight_group = object()
        client._dist_weight_group_name = "weight_update_group"
        cached = client._update_weight_factory()
        self.assertIs(cached, first)
        self.assertIs(cached.context.dist_weight_group, client._dist_weight_group)
        self.assertEqual(cached.context.dist_weight_group_name, "weight_update_group")

    def test_mixin_syncs_ipc_fields_onto_cached_context(self):
        client = _make_mixin_client()
        factory = client._update_weight_factory()

        client._ipc_gather_dst_rank = 2
        client._ipc_gather_group = object()
        client._ipc_target = 1
        cached = client._update_weight_factory()

        self.assertIs(cached, factory)
        self.assertEqual(cached.context.ipc_gather_dst_rank, 2)
        self.assertIs(cached.context.ipc_gather_group, client._ipc_gather_group)
        self.assertEqual(cached.context.ipc_target, 1)

    async def test_init_returns_group_without_mutating_context(self):
        endpoint = MagicMock()
        endpoint.init_weights_update_group.remote.return_value = "ref"
        rpc_client = MagicMock()
        rpc_client.get_target_endpoint.return_value = endpoint
        context = _make_sglang_context(rpc_client_lst=[rpc_client])
        factory = get_update_weight_factory(context=context, model_arch=MODEL_ARCH.QWEN3)
        sentinel_group = object()

        with patch("gpatch_v4.client.update_weight_factory.dist") as mock_dist, patch(
            "gpatch_v4.client.update_weight_factory.ray"
        ) as mock_ray, patch(
            "gpatch_v4.client.update_weight_factory.cpu_barrier"
        ), patch(
            "gpatch_v4.client.update_weight_factory.socket"
        ) as mock_socket, patch(
            "sglang.srt.utils.init_custom_process_group",
            return_value=sentinel_group,
        ) as mock_init_pg:
            mock_dist.get_rank.return_value = 0
            mock_ray._private.services.get_node_ip_address.return_value = "127.0.0.1"
            sock = MagicMock()
            sock.getsockname.return_value = ("127.0.0.1", 29500)
            mock_socket.socket.return_value.__enter__.return_value = sock

            def _broadcast(meta, src=0):
                meta[0] = "127.0.0.1"
                meta[1] = 29500

            mock_dist.broadcast_object_list.side_effect = _broadcast

            group, name = await factory.init_distributed_weight_group("wg")

        self.assertIs(group, sentinel_group)
        self.assertEqual(name, "wg")
        self.assertIsNone(context.dist_weight_group)
        self.assertIsNone(context.dist_weight_group_name)
        mock_init_pg.assert_called_once()
        mock_ray.get.assert_called_once()

    async def test_destroy_returns_none_pair(self):
        endpoint = MagicMock()
        endpoint.destroy_weights_update_group.remote.return_value = "ref"
        rpc_client = MagicMock()
        rpc_client.get_target_endpoint.return_value = endpoint
        existing_group = object()
        context = _make_sglang_context(
            rpc_client_lst=[rpc_client],
            dist_weight_group=existing_group,
            dist_weight_group_name="wg",
        )
        factory = get_update_weight_factory(context=context, model_arch=MODEL_ARCH.QWEN3)

        with patch("gpatch_v4.client.update_weight_factory.dist") as mock_dist, patch(
            "gpatch_v4.client.update_weight_factory.ray"
        ) as mock_ray, patch(
            "gpatch_v4.client.update_weight_factory.cpu_barrier"
        ):
            mock_dist.get_rank.return_value = 0
            group, name = await factory.destroy_distributed_weight_group()

        self.assertIsNone(group)
        self.assertIsNone(name)
        # Factory must not clear ownership fields; mixin owns write-back.
        self.assertIs(context.dist_weight_group, existing_group)
        self.assertEqual(context.dist_weight_group_name, "wg")
        mock_dist.destroy_process_group.assert_called_once_with(existing_group)
        mock_ray.get.assert_called_once()

    async def test_destroy_noop_when_group_name_missing(self):
        context = _make_sglang_context(dist_weight_group=None, dist_weight_group_name=None)
        factory = get_update_weight_factory(context=context, model_arch=MODEL_ARCH.QWEN3)

        group, name = await factory.destroy_distributed_weight_group()

        self.assertIsNone(group)
        self.assertIsNone(name)

    async def test_mixin_init_destroy_owns_state_and_syncs_context(self):
        client = _make_mixin_client()
        factory = client._update_weight_factory()
        sentinel_group = object()

        with patch.object(
            factory,
            "init_distributed_weight_group",
            new=AsyncMock(return_value=(sentinel_group, "wg")),
        ):
            await client.init_distributed_weight_group("wg")

        self.assertIs(client._dist_weight_group, sentinel_group)
        self.assertEqual(client._dist_weight_group_name, "wg")
        self.assertIs(factory.context.dist_weight_group, sentinel_group)
        self.assertEqual(factory.context.dist_weight_group_name, "wg")
        self.assertIs(client._cached_update_weight_factory, factory)

        with patch.object(
            factory,
            "destroy_distributed_weight_group",
            new=AsyncMock(return_value=(None, None)),
        ):
            await client.destroy_distributed_weight_group()

        self.assertIsNone(client._dist_weight_group)
        self.assertIsNone(client._dist_weight_group_name)
        self.assertIsNone(factory.context.dist_weight_group)
        self.assertIsNone(factory.context.dist_weight_group_name)

    # async def test_dsv4_vllm_requires_bucketed_ipc(self):
    #     context = _make_sglang_context(
    #         infer_backend="vllm",
    #         update_weight_use_bucketed_ipc=False,
    #         svr_cluster_num_per_sampler=[],
    #         sampler_engine_gpu_counts=None,
    #     )
    #     factory = get_update_weight_factory(
    #         context=context, model_arch=MODEL_ARCH.DEEPSEEK_V4
    #     )
    #     self.assertIsInstance(factory, DeepSeekV4VllmUpdateWeightFactory)

    #     with self.assertRaisesRegex(AssertionError, "bucketed IPC"):
    #         await factory.update_weights_by_ipc_handle(0, SimpleNamespace(), False)
