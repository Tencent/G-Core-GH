"""Unit tests for partial-colocated sampler IPC rank mapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gpatch_v4.client.sampler_client as sampler_mod
from gpatch_v4.client.sampler_client import SamplerClient


def _dist_config(nnodes=4, gpus_per_node=8, tp=8, pp=1):
    return SimpleNamespace(
        nnodes=nnodes,
        num_gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )


def _partial_config():
    sampler_dc = _dist_config()
    return SimpleNamespace(
        placement_type="partial_colocated",
        sampler=SimpleNamespace(
            infer_engine_configs=[SimpleNamespace(dist_config=sampler_dc)],
        ),
    )


class TestPartialColocatedIpcMapping(unittest.TestCase):
    def _build_client_for_rank(self, rank: int):
        client = object.__new__(SamplerClient)
        client.config = _partial_config()
        client._ipc_gather_dst_rank = None
        client._ipc_gather_group = None
        client._ipc_target = None

        groups = []

        def _new_group(ranks, backend):
            groups.append((tuple(ranks), backend))
            return f"group-{len(groups) - 1}"

        with patch.object(sampler_mod.dist, "get_rank", return_value=rank), patch.object(
            sampler_mod.dist, "new_group", side_effect=_new_group
        ):
            client.build_update_from_tensor_meta()

        return client, groups

    def test_sampler_prefix_rank_maps_to_its_tp_head(self):
        client, groups = self._build_client_for_rank(rank=9)

        assert groups == [
            (tuple(range(0, 8)), "gloo"),
            (tuple(range(8, 16)), "gloo"),
            (tuple(range(16, 24)), "gloo"),
            (tuple(range(24, 32)), "gloo"),
        ]
        assert client._ipc_gather_dst_rank == 8
        assert client._ipc_gather_group == "group-1"
        assert client._ipc_target == 1

    def test_gen_rm_suffix_rank_does_not_send_sampler_weights(self):
        client, _ = self._build_client_for_rank(rank=40)

        assert client._ipc_gather_dst_rank is None
        assert client._ipc_gather_group is None
        assert client._ipc_target is None


if __name__ == "__main__":
    unittest.main()
