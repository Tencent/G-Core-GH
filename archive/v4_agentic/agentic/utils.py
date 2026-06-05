import hashlib
import os
import random
import tempfile
from contextlib import contextmanager
from typing import Callable, Dict, Optional

import numpy as np
import torch
from filelock import FileLock
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from gpatch_v4.agentic.proto import DataProto
from gpatch_v4.configs.agentic_config import AgenticConfig, RewardNormalizationConfig
from gpatch_v4.utils.common_utils import log
from gpatch_v4.utils.packages import is_transformers_version_greater_than


@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


@contextmanager
def file_lock_context(lock_path: str):
    temp_lock_path = os.path.join(
        tempfile.gettempdir(), f"{hashlib.md5(lock_path.encode()).hexdigest()}.lock"
    )
    with FileLock(temp_lock_path):
        yield


def prepare_automap_files(model_path: str):
    python_files = []
    for file_name in os.listdir(model_path):
        if file_name.endswith(".py") and os.path.isfile(os.path.join(model_path, file_name)):
            python_files.append(file_name)
    with file_lock_context(model_path):
        for file_name in python_files:
            get_cached_module_file(model_path, file_name)


def default_tokenizer_provider(model_name_or_path: str = None):
    prepare_automap_files(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        split_special_tokens=False,
        trust_remote_code=True,
        padding_side="left",
    )
    return tokenizer


def default_processor_provider(model_name_or_path: str = None):
    prepare_automap_files(model_name_or_path)
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        log(f"processor not found: {e}")
        processor = None
    return processor


def get_extra_data_provider(model_name_or_path: str, processor=None):
    # model_name_or_path = download_model(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    if "qwen2" in config.model_type:
        import types

        from transformers import BatchFeature  # help define a object to accesss attr

        dummy_self = BatchFeature(
            {
                "config":
                    BatchFeature(
                        {
                            "vision_config":
                                BatchFeature(
                                    {"spatial_merge_size": processor.image_processor.merge_size}
                                ),
                            "image_token_id":
                                processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                            "video_token_id":
                                processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                            "vision_start_token_id":
                                processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                        }
                    )
            }
        )
        if is_transformers_version_greater_than("4.52.0"):
            from transformers.models.qwen2_vl import Qwen2VLModel

            get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, dummy_self)
        else:
            from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

            get_rope_index = types.MethodType(
                Qwen2VLForConditionalGeneration.get_rope_index, dummy_self
            )

        def extra_data_provider(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            rope_index = get_rope_index(input_ids, image_grid_thw, video_grid_thw,
                                        attention_mask)[0]
            # (3, bsz, seqlen) -> (bsz, 3, seqlen) to put it into DataProto,
            # transpose it batck to (3, bsz, seqlen) before forward for model
            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider
    return None


class LoggerAdaptor:
    def info(self, *args):
        log(*args)

    def debug(self, *args):
        log(*args)

    def warning(self, *args):
        log(*args)


@torch.no_grad()
def get_score_normalize_fn(rn_cfg) -> Callable:
    grouping, method = rn_cfg.grouping, rn_cfg.method
    if method == "mean_std":
        norm_func = lambda x: (
            (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            if x.numel() > 1 and x.std(dim=-1, keepdim=True).abs().max() > 1e-6 and
            ((x.max() - x.min()) > 0.001) else torch.zeros_like(x)
        )  # stable to bf16 than x.std()
    elif method == "mean":
        norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
    elif method == "asym_clip":
        norm_func = lambda x: (
            (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.numel() >
            1 and x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)
        ).clamp(min=-1, max=3)
    elif method == "identity":
        norm_func = lambda x: x
    else:
        raise ValueError(f"Invalid normalization method: {method}")

    return norm_func


@torch.no_grad()
def compute_discounted_returns(batch: DataProto, adv_estimator, gamma=1.0) -> DataProto:
    """
    Compute discounted returns for each trajectory in the batch.

    Args:
        batch (DataProto): A `DataProto` instance containing trajectories.
        adv_estimator (str): Advantage estimator type; only `"gigpo"` triggers computation here.
        gamma (float, optional): Discount factor applied to future rewards. Defaults to 1.0.

    Returns:
        DataProto: Updated batch where each trajectory contains an extra tensor key
                   `"step_rewards"` holding the computed discounted returns.
    """

    episode_score = batch.non_tensor_batch["episode_scores"].astype(np.float32)
    episode_score = torch.as_tensor(episode_score)
    total_steps = batch.non_tensor_batch["total_step"].astype(np.int64)
    total_steps = torch.as_tensor(total_steps)
    report_rewards = episode_score / total_steps
    batch.batch["report_rewards"] = report_rewards

    if adv_estimator in ["gigpo", "step_reinforce"]:
        batch.batch["sample_order_placeholder"] = torch.arange(
            batch.batch.batch_size[0], device=batch.batch.device
        )
        batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
        for traj_id, traj_batch in batch_group_by_traj.items():

            indices: Tensor = torch.argsort(
                torch.from_numpy(traj_batch.non_tensor_batch["step"].astype(np.int64))
            )
            traj_batch.reorder(indices)
            step_scores = traj_batch.non_tensor_batch["step_scores"].astype(np.float32)
            rewards = torch.as_tensor(step_scores).float()
            discounts = torch.empty_like(rewards)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                discounts[t] = running_return
            # step punishment
            step = traj_batch.non_tensor_batch["step"].astype(np.int64)
            step = torch.as_tensor(step)
            total_steps = traj_batch.non_tensor_batch["total_step"].astype(np.int64)
            total_steps = torch.as_tensor(total_steps)
            # episode score
            episode_score = traj_batch.non_tensor_batch["episode_scores"].astype(np.float32)
            episode_score = torch.as_tensor(episode_score)

            # credit assign
            traj_batch.batch[
                "step_rewards"
            ] = 0.3 * rewards + 0.3 * discounts + 0.3 * episode_score + (step - total_steps) * 0.1

        merged = DataProto.concat(list(batch_group_by_traj.values()))
        merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
        merged.pop("sample_order_placeholder")
        return merged
    # elif adv_estimator == "grpo":
    #     step = batch.non_tensor_batch["step"].astype(np.int64)
    #     step = torch.as_tensor(step)
    #     total_steps = batch.non_tensor_batch["total_step"].astype(np.int64)
    #     total_steps = torch.as_tensor(total_steps)
    #     step_scores = batch.non_tensor_batch["step_scores"].astype(np.float32)
    #     step_scores = torch.as_tensor(step_scores)
    #     episode_score = batch.non_tensor_batch["episode_scores"].astype(np.float32)
    #     episode_score = torch.as_tensor(episode_score)
    #     # step punishments
    #     # NOTE @yeazhao 这里的计算方式有点怪
    #     batch.batch["step_rewards"] = step_scores + episode_score + (step - total_steps) * 0.1
    #     return batch
    else:
        return batch


def grouped_reward_norm(
    batch: "DataProto", reward_normalization: RewardNormalizationConfig
) -> torch.Tensor:
    batch.batch["sample_order_placeholder"] = torch.arange(
        batch.batch.batch_size[0], device=batch.batch.device
    )
    grouping = reward_normalization.grouping
    batch_grouped: Dict[str, DataProto] = {"default": batch}
    if grouping != "batch":
        batch_grouped = batch.group_by(keys=grouping)
    batch_list = []
    for i, (group_name, group_batch) in enumerate(batch_grouped.items()):
        score_norm_fn = get_score_normalize_fn(rn_cfg=reward_normalization)
        normalized_acc_scores = score_norm_fn(group_batch.batch["scores"])
        group_batch.batch["grouped_rewards"] = normalized_acc_scores
        group_batch.batch["grouping_index"] = torch.tensor(
            [i] * group_batch.batch.batch_size[0], device=batch.batch.device
        )
        batch_list.append(group_batch)
    batch = DataProto.concat(batch_list)
    batch.reorder(indices=torch.argsort(batch.batch["sample_order_placeholder"]))
    batch.pop("sample_order_placeholder")
    return batch.batch.pop("grouped_rewards"), batch.batch["grouping_index"]


def build_state_group(batch: "DataProto") -> "DataProto":
    batch.batch["sample_order_placeholder"] = torch.arange(
        batch.batch.batch_size[0], device=batch.batch.device
    )
    batch_group_by_traj_group: Dict[str, DataProto] = batch.group_by(keys="traj_group_id")
    merged = []
    for traj_group_id, traj_group_batch in batch_group_by_traj_group.items():
        batch_group_by_state: Dict[str, DataProto] = traj_group_batch.group_by(keys="state_hash")
        for state, state_batch in batch_group_by_state.items():
            state_batch.non_tensor_batch["state_group_id"] = np.array(
                [state] * state_batch.batch.batch_size[0], dtype=object
            )
            merged.append(state_batch)
    state_batch_size = [len(m) for m in merged]
    merged = DataProto.concat(merged)
    merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
    merged.pop("sample_order_placeholder")
    metrics = merged.meta_info.pop("metrics", {})
    metrics["system/state_batch_size/max"] = np.max(state_batch_size)
    metrics["system/state_batch_size/mean"] = np.mean(state_batch_size)
    metrics["system/state_batch_size/min"] = np.min(state_batch_size)
    merged.meta_info["metrics"] = metrics
    return merged


@torch.no_grad()
def compute_response_level_rewards(
    batch: "DataProto", agentic_config: AgenticConfig
) -> "DataProto":
    if agentic_config.adv_estimator == "gigpo":
        # ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/gigpo/core_gigpo.py#L109
        # step 1
        episode_scores = torch.from_numpy(
            batch.non_tensor_batch["episode_scores"].astype(np.float32)
        )
        scores_to_group = DataProto.from_dict({"scores": episode_scores})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        episode_rewards: torch.Tensor = grouped_reward_norm(
            scores_to_group, reward_normalization=agentic_config.reward_normalization
        )

        # step 2
        batch = build_state_group(batch=batch)

        # step 3
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        step_rewards: torch.Tensor = grouped_reward_norm(
            batch=scores_to_group,
            reward_normalization=RewardNormalizationConfig(
                grouping="state_group_id", method=agentic_config.reward_normalization.method
            )
        )

        batch.batch[
            "response_level_rewards"
        ] = agentic_config.episode_reward_weight * episode_rewards + agentic_config.step_reward_weight * step_rewards
        batch.batch["episode_rewards_norm"] = episode_rewards
        batch.batch["step_rewards_norm"] = step_rewards
    elif agentic_config.adv_estimator == "step_reinforce":
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        batch.batch["response_level_rewards"] = grouped_reward_norm(
            scores_to_group, reward_normalization=agentic_config.reward_normalization
        )
    # elif agentic_config.adv_estimator == "grpo":
    #     # FIXME @yeazhao 这里对吗？按照 ROLLOUT 的逻辑，感觉 grpo 要用  scores 而不是 step_rewards
    #     scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
    #     scores_to_group.non_tensor_batch = batch.non_tensor_batch
    #     batch.batch["response_level_rewards"] = grouped_reward_norm(
    #         scores_to_group, reward_normalization=RewardNormalizationConfig(grouping="grpo_anchor", method=agentic_config.reward_normalization.method)
    #     )

    else:
        scores_to_group = DataProto.from_dict({"scores": batch.batch["scores"].clone().sum(dim=-1)})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        batch.batch["response_level_rewards"], batch.batch["grouping_index"] = grouped_reward_norm(
            scores_to_group, reward_normalization=agentic_config.reward_normalization
        )

    return batch
