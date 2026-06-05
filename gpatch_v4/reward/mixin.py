from typing import Any, Dict, List, Union

import torch

from gpatch_v4.core.parallel_state import is_mp_head


class RewardMixin:
    """Mixin providing reward model lifecycle and post-processing utilities."""
    def get_reward_model(self):
        """Return the underlying reward model.

        Returns
        -------
        object
        """
        return self.reward_model

    def _has_no_reward_model(self):
        """``True`` if ``self.reward_model`` is ``None``.

        Returns
        -------
        bool
        """
        return self.reward_model is None

    def sleep(self):
        """Offload the reward model to CPU."""
        assert self.config.placement_type != "disaggregated"
        if not self._has_no_reward_model():
            self.reward_model.offload_model()

    def wake_up(self):
        """Onload the reward model back to GPU."""
        assert self.config.placement_type != "disaggregated"
        if not self._has_no_reward_model():
            self.reward_model.onload_model()

    def post_process_rewards(
        self, resp_dict: Dict[str, torch.Tensor], num_batch: int, sampling_repeat_n: int
    ):
        """Chunk and reorganize reward response tensors by micro-batch.

        Parameters
        ----------
        resp_dict : dict[str, torch.Tensor]
        num_batch : int
        sampling_repeat_n : int
            Repeated samples per prompt.

        Returns
        -------
        list of dict
            One dict per micro-batch with list-valued reward tensors.
        """
        chunked_resp_dicts = []
        if is_mp_head():
            for k in resp_dict.keys():
                if resp_dict[k] is not None:
                    resp_dict[k] = resp_dict[k].chunk(num_batch)

            for b_i in range(num_batch):
                infer_ret: Dict[str, List[Any]] = {}
                for k, v in resp_dict.items():
                    if v is not None:
                        tmp_v = v[b_i]
                        if torch.is_tensor(tmp_v):
                            tmp_v = [e.squeeze(0) for e in tmp_v.chunk(tmp_v.shape[0])]
                        assert isinstance(tmp_v, list)
                        infer_ret[k] = tmp_v
                        assert sampling_repeat_n * self.config.training.rollout_mbs == len(tmp_v), \
                            f"key {k} {sampling_repeat_n=} != {len(tmp_v)=}"
                for k, v in resp_dict.items():
                    if v is None:
                        infer_ret[k] = [None] * sampling_repeat_n * self.config.training.rollout_mbs
                chunked_resp_dicts.append(infer_ret)
        return chunked_resp_dicts
