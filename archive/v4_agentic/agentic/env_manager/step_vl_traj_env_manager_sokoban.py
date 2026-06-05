import base64
from contextlib import nullcontext, suppress
from threading import Lock
from typing import Dict, List, Optional, Tuple

with suppress(ImportError):
    import gem
import numpy as np
import PIL
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from gpatch_v4.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from gpatch_v4.agentic.env_manager.token_mask_utils import (
    split_by_token,
    token_ids_to_assistant_mask,
)
from gpatch_v4.agentic.env_manager.traj_env_manager import TrajEnvManager
from gpatch_v4.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from gpatch_v4.agentic.proto import DataProto
from gpatch_v4.agentic.utils import LoggerAdaptor
from gpatch_v4.client import SamplerClient
from gpatch_v4.configs.agentic_config import EnvManagerConfig
from gpatch_v4.configs.config import AgenticRlConfig
from gpatch_v4.utils import aggregate_metrics, pad_to_length
from gpatch_v4.utils.constants import GenerateStopReason
from gpatch_v4.utils.str_utils import contains_renderable_field

# from roll.datasets.collator import DataCollatorWithPaddingForMM
# from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
# from roll.utils.env_action_limiter import get_global_limiter

PRE_PROMPT_FORMAT = """
## Env Instruction: {env_instruction}
## State Description:
Prior to this step, you have completed {step_count} steps.
Recent History: Below are the most recent {history_length} observations and your responses.
"""

POST_PROMPT_FORMAT = """
You have {actions_left} actions left.
## Output Format Requirement:
1. output format is '<answer> [your answer] </answer>' with no extra text.
2. Max response length: {max_response_length} words (tokens).
Strictly follow this format.
Decide the next action:
"""


class StepVLTrajEnvManager(TrajEnvManager):
    def __init__(
        self,
        rl_config: AgenticRlConfig,
        manager_config: EnvManagerConfig,
        env_config: Dict,
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin,
        sampler_client: SamplerClient,
        output_queue: "GroupQueueManager",
        thread_lock: Lock,
        mode='train',
        extra_data_provider=None,
        *args,
        **kwargs
    ):
        """
        """
        BaseEnvManager.__init__(self)
        self.logger = LoggerAdaptor()
        self.rl_config = rl_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor: ProcessorMixin = processor
        self.extra_data_provider = extra_data_provider
        self.output_queue = output_queue
        self.mode = mode
        self.sampler_client = sampler_client
        self.manager_config = manager_config

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = 0
        self.current_step = -1
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False)
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()

        # Set environment step concurrency limit

        self.env_step_limiter = nullcontext()
        """
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)
        """

        with self.thread_lock, self.env_step_limiter:
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])

        agentic_config = self.rl_config.training.agentic
        self.cfg_template = agentic_config.env_cfg_template
        self.agent_system_template = self.cfg_template.agent_system_template
        """
        vl messages user content is List[Dict], like:
        [
                {
                    "type": "text",
                    "text":  "{observation}\nTurn {turn_idx}:\nCurrent state is:\n"
                },
                {
                    "type": "image",
                    "image": None
                },
                {
                    "type": "text",
                    "text": self.next_step_template

                }
            ]
        """

        self.pre_step_template = self.cfg_template.pre_step_template
        self.next_step_template = self.cfg_template.next_step_template
        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"pre_step_template: {self.pre_step_template}")
            self.logger.info(f"next_step_template: {self.next_step_template}")

        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            sampler_client=self.sampler_client,
            llm_proxy_config=agentic_config.train_env_manager.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

    def make_decision(self, rollout_cache: RolloutCache):
        prompt_ids, images, messages, prompt_ids_for_train, multi_modal_data = self.format_messages(
            rollout_cache
        )

        if len(prompt_ids_for_train) >= self.rl_config.training.seq_length:
            self.logger.warning(
                f"sequence_length = {self.rl_config.training.seq_length} input_ids length = {len(prompt_ids_for_train)},"
                f"maybe you should increase the response_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        lm_output: DataProto = self.llm_proxy.generate(
            {
                "prompt_token_ids": prompt_ids,
                "images": images,
            }
        )

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output['response_ids']
        content = self.rollout_cache.history[-1]
        messages.append(
            {
                "role": "assistant",
                "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)
            }
        )
        prompt_len = lm_output["prompt_len"]
        assert prompt_len == len(
            prompt_ids_for_train
        ), f"{prompt_len} != {len(prompt_ids_for_train)}"
        content["messages"] = messages
        content["prompt_ids"] = prompt_ids.tolist()
        content["images"] = images
        content["response_ids"] = response_ids
        content["prompt_ids_for_train"] = prompt_ids_for_train
        content["multi_modal_data"] = multi_modal_data
        content["rollout_log_probs"] = lm_output["output_logprobs"]

        lm_output["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: RolloutCache) -> Tuple[DataProto, List[Dict]]:
        messages = [
            {
                "role": "system",
                "content": self.agent_system_template
            },
        ]

        current_cache = history.history[-1]
        memory_history = []
        if "history_length" in self.cfg_template:
            memory_history = history.history[-self.cfg_template["history_length"]:-1]

        hist_images = []
        hist_response = []
        for content in memory_history:
            assert "observation" in content, (
                "The current EnvManager is specifically tailored for standard RL interaction "
                "sequences, following the format of (s, a, r, s, a, r...)."
            )
            hist_images.append(PIL.Image.fromarray(content["observation"], mode='RGB'))
            hist_response.append(content["llm_response"])

        current_image = PIL.Image.fromarray(current_cache["observation"], mode='RGB')

        content_list_dict = [
            {
                "type":
                    "text",
                "text":
                    PRE_PROMPT_FORMAT.format(
                        env_instruction=history.history[0]["env_instruction"],
                        step_count=len(history.history) - 1,
                        history_length=len(memory_history)
                    )
            }
        ]

        for base64_image, response in zip(hist_images, hist_response):
            content_list_dict.append({"type": "text", "text": "\n Then the state is: "})
            content_list_dict.append({
                "type": "image",
                "image_data": f"image_placeholder",
            })
            content_list_dict.append(
                {
                    "type": "text",
                    "text": "\n And your response is: {}".format(response)
                }
            )

        content_list_dict.append(
            {
                "type":
                    "text",
                "text":
                    "\nYou are currently at step {}\nBelow are the current state:".format(
                        len(history.history)
                    )
            }
        )

        content_list_dict.append({
            "type": "image",
            "image_data": "image_placeholder",
        })

        content_list_dict.append(
            {
                "type":
                    "text",
                "text":
                    POST_PROMPT_FORMAT.format(
                        actions_left=current_cache["actions_left"],
                        max_response_length=self.env_config["max_tokens_per_step"]
                    )
            }
        )

        messages.append({"role": "user", "content": content_list_dict})
        images = hist_images + [current_image]

        lm_input_texts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=[lm_input_texts], images=images, padding=False, return_mm_token_type_ids=True
        )
        input_id_for_train = inputs.input_ids[0]
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        # print(f"lm_input_texts :{lm_input_texts}", flush=True)
        prompt_id = self.tokenizer(lm_input_texts)["input_ids"]
        prompt_id = torch.tensor(prompt_id).reshape(-1)
        # print(f"{prompt_id}", flush=True)
        multi_modal_data = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        return prompt_id, images, messages, input_id_for_train, multi_modal_data

    def formulate_rollout_step(
        self, rollout_cache: RolloutCache, step, total_step, episode_score=None
    ):
        success = True
        if episode_score is None:
            # pop last observation
            if 'observation' in rollout_cache.history[-1]:
                rollout_cache.history.pop(-1)
            rewards = [i['reward'] for i in rollout_cache.history]
            episode_score = sum(rewards)
        history = rollout_cache.history[step]
        messages = history["messages"]
        prompt_ids = history["prompt_ids_for_train"]
        response_ids = history["response_ids"]
        multi_modal_data = history["multi_modal_data"]
        rollout_log_probs = history["rollout_log_probs"]

        if len(response_ids) > 400:
            print(f"[DEBUG] responses too long: {messages}")

        token_ids = prompt_ids + response_ids
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        # shift and pad rollout_log_probs
        rollout_log_probs = [1.0] * (len(prompt_ids) - 1) + rollout_log_probs
        rollout_log_probs = rollout_log_probs + (input_ids.shape[1] -
                                                 len(rollout_log_probs)) * [1.0]
        rollout_log_probs = torch.tensor(rollout_log_probs, dtype=torch.float).unsqueeze(0)

        response_mask = [0] * len(prompt_ids) + [1] * (input_ids.shape[1] - len(prompt_ids))
        prompt_mask = [1] * len(prompt_ids) + [0] * (input_ids.shape[1] - len(prompt_ids))
        response_mask = torch.tensor(response_mask, dtype=torch.bool).unsqueeze(0)
        prompt_mask = torch.tensor(prompt_mask, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * input_ids.shape[1], dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = history['reward']

        input_ids = pad_to_length(
            input_ids,
            length=self.rl_config.training.seq_length,
            pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_to_length(
            attention_mask, length=self.rl_config.training.seq_length, pad_value=0
        )
        response_mask = pad_to_length(
            response_mask, length=self.rl_config.training.seq_length, pad_value=0
        )
        prompt_mask = pad_to_length(
            prompt_mask, length=self.rl_config.training.seq_length, pad_value=0
        )
        score_tensor = pad_to_length(
            score_tensor, length=self.rl_config.training.seq_length, pad_value=0
        )
        rollout_log_probs = pad_to_length(
            rollout_log_probs, length=self.rl_config.training.seq_length, pad_value=0.0
        )

        sample = DataProto(
            batch=TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "response_mask": response_mask,
                    "prompt_mask": prompt_mask,
                    "scores": score_tensor,
                    "rollout_log_probs": rollout_log_probs,
                },
                batch_size=input_ids.shape[0]
            ),
            non_tensor_batch={
                "env_ids": np.array([rollout_cache.env_id], dtype=object),
                "group_ids": np.array([rollout_cache.group_id], dtype=object),
                "messages_list": np.array([messages], dtype=object),
                "tags": np.array([rollout_cache.tag], dtype=object),
                "step_scores": np.array([history['reward']], dtype=object),
                "episode_scores": np.array([episode_score], dtype=object),
                "step": np.array([step], dtype=object),
                "total_step": np.array([total_step], dtype=object),
                "raw_rewards": np.array([history['reward']], dtype=object),
                "success": np.array([success], dtype=object),
                "multi_modal_data": np.array([multi_modal_data], dtype=object),
            }
        )

        return sample

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        # TODO: check inconsistent tokenization between successive encode-decode operations
        #  can potentially lead to a training crash. check token in token out
        #  the same as TrajEnvManager.

        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)

        samples: List[DataProto] = []

        rewards = [i['reward'] for i in rollout_cache.history]
        episode_score = sum(rewards)
        print(
            f"[DEBUG] {rollout_cache.env_id}-{rollout_cache.step} rewards: {len(rewards)} {rewards}, episode_score: {episode_score}"
        )
        traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
        traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
        total_step = len(rollout_cache.history)
        # pdb.set_trace()
        for step, history in enumerate(rollout_cache.history):
            sample = self.formulate_rollout_step(rollout_cache, step, total_step, episode_score)
            sample.non_tensor_batch["grpo_anchor"] = np.array([f"{traj_id}_{step}"], dtype=object)
            samples.append(sample)
            # grpo sample
            if self.rl_config.training.agentic.adv_estimator == "grpo":
                trajs = self.tree_eval(rollout_cache, step, 4)
                for e in trajs:
                    sample = self.formulate_rollout_step(e, step, len(e.history), None)
                    sample.non_tensor_batch["grpo_anchor"] = np.array(
                        [f"{traj_id}_{step}"], dtype=object
                    )
                    samples.append(sample)

        batch: DataProto = DataProto.concat(samples)
        response_length = batch.batch["response_mask"].float().sum(-1).mean().item()
        metrics_agg_mode = rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in rollout_cache.history]
        env_metric = aggregate_metrics(
            history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode
        )
        env_metric["num_actions"] = rollout_cache.step

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        batch.meta_info = {"metrics": env_metric}
        # batch.save_to_disk("/mnt/geminigmceph/user_jingtxu/code/for_xiaochengxu/test_roll/roll_1111/batch2.pkl")
        return batch

    def replay(self, rollout_cache):
        tmp = self.reset()
        for s in rollout_cache.history:
            response_ids = s["response_ids"]
            self.step({"response_ids": response_ids})

    def extend_traj(self):
        rollout_cache = self.rollout_cache
        while True:
            lm_output: DataProto = self.make_decision(rollout_cache)
            stop_reason = lm_output.pop("stop_reason")
            if stop_reason == GenerateStopReason.FINISH:
                rollout_cache: RolloutCache = self.step(lm_output)
            if rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH:
                break
        return rollout_cache

    def tree_eval(self, rollout_cache, step, repeat):
        trajs = []
        self.episode_id_back = self.episode_id
        for i in range(repeat):
            self.episode_id = self.episode_id_back - 1
            rollout_cache_i = rollout_cache.fork(step)
            self.replay(rollout_cache_i)
            trajs.append(self.extend_traj())
        self.episode_id = self.episode_id_back
        return trajs
