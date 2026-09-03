from typing import Any, Dict, List, Set, Tuple

import torch
from typing_extensions import override

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.extended_model.base import ApplySamplingRolloutAttrBase
from gpatch_v4.utils import BroadcastUtils, log


class ApplySamplingRolloutAttrLLM(ApplySamplingRolloutAttrBase):
    """LLM-specific rollout attribute handler (no-ops for text models).

    Parameters
    ----------
    config : RlConfig
    """
    def __init__(self, config: RlConfig):
        self.config = config
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.data_used: Set[str] = set()

    @override
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        assert "unique_id" in rollout_batch

        unique_id_list = rollout_batch["unique_id"]
        cache_keys = rollout_batch.pop("cache_keys", None)
        if cache_keys is None:
            log("Warning: cache_keys is None, no data will be cached", rank=0)
        for idx, unique_id in enumerate(unique_id_list):
            assert unique_id not in self.data_cache
            self.data_cache[unique_id] = {}
            if cache_keys is not None:
                for key in cache_keys:
                    assert key in rollout_batch
                    self.data_cache[unique_id][key] = rollout_batch[key][idx]

        if cache_keys is not None:
            for key in cache_keys:
                del rollout_batch[key]
        return rollout_batch

    @override
    def cached_rollout_attrs(self) -> Dict[str, Dict[str, Any]]:
        return self.data_cache

    @override
    def remove_rollout_attr(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        #TODO: 文本训练暂时用不着，后面再说
        return rollout_batch

    @override
    def add_back_rollout_attr_after_sampling(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> List[Dict[str, List[Any]]]:
        # Broadcast data_cache to TP/PP non-head ranks.
        # Skip when running in a single-process context (e.g. RolloutController)
        # where torch.distributed is not initialized.
        if torch.distributed.is_initialized():
            self.data_cache = BroadcastUtils.broadcast_object_within_mp_and_cp(
                self.data_cache, make_recursive_clone_in_case_of_view=True
            )
        for rollout_batch in rollout_batches:
            assert "unique_id" in rollout_batch
            unique_id_list = rollout_batch.pop("unique_id")
            parent_uid_list = rollout_batch.pop("parent_unique_id", None)
            for idx, unique_id in enumerate(unique_id_list):
                lookup_id = unique_id
                if lookup_id not in self.data_cache and parent_uid_list is not None:
                    lookup_id = parent_uid_list[idx]
                assert lookup_id in self.data_cache, f"error: {lookup_id=} {self.data_cache.keys()=}"

                self.data_used.add(lookup_id)
                for k, v in self.data_cache[lookup_id].items():
                    if k not in rollout_batch:
                        rollout_batch[k] = []
                    rollout_batch[k].append(v)
        return rollout_batches

    @override
    def add_back_rollout_attr(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> List[Dict[str, List[Any]]]:
        #TODO: 文本训练暂时用不着，后面再说
        return rollout_batches

    @override
    def replay_rollout_batch(self, rollout_batch):
        # 这里没有测试过
        return rollout_batch

    @override
    def clear_data_cache(self):
        data_used_tmp = list(self.data_used)
        for k in data_used_tmp:
            del self.data_cache[k]
            self.data_used.remove(k)
