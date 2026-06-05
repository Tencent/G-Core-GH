import base64
import io
import json
import pdb
import queue
import time
from contextlib import nullcontext, suppress
from threading import Lock
from typing import Dict, List, Optional, Tuple

with suppress(ImportError):
    import gem
import numpy as np
import PIL
import requests
import torch
from codetiming import Timer
from PIL import Image
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from gpatch_v4.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from gpatch_v4.agentic.env_manager.traj_env_manager import TrajEnvManager
from gpatch_v4.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from gpatch_v4.agentic.proto import DataProto
from gpatch_v4.agentic.utils import LoggerAdaptor
from gpatch_v4.client import SamplerClient
from gpatch_v4.configs.agentic_config import EnvManagerConfig
from gpatch_v4.configs.config import AgenticRlConfig
from gpatch_v4.utils import aggregate_metrics, pad_to_length
from gpatch_v4.utils.constants import GenerateStopReason


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
        Args:
            data_source: Path to data file or list of data items containing app_id and instruction
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

        self.env_step_limiter = nullcontext()

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
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            sampler_client=self.sampler_client,
            llm_proxy_config=agentic_config.train_env_manager.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")

        fallback_image_path = "/mnt/geminigmceph/user_kangyuqiao/6_click_110_382.jpg"
        self.fallback_image = Image.open(fallback_image_path)
        if self.fallback_image.mode != 'RGB':
            self.fallback_image = self.fallback_image.convert('RGB')

    def reset(self, index=None, rollout_id=None) -> RolloutCache:
        """
        Reset environment with data from specified index
        Args:
            index: Line index (0-based) to read from JSONL file. If None, use default values.
        Returns:
            RolloutCache: Initialized rollout cache
        """
        # Read data from specified index in JSONL file
        # app_id = 'wx0e6ed4f51db9d078'  # Default value
        # instruction = '帮我订一杯咖啡'  # Default value

        # Initialize environment with loaded data
        self.env.initialize(
            batch_id=self.current_step,
            instruction_id=index * self.group_num + self.env_config['group_id'],
            env_config=self.env_config,
            rollout_id=rollout_id,
            uin=None,
            base_url=None
        )

        self.logger.info(
            f"Initialize Success. training_id {self.env_config['training_id']}, batch_id {self.current_step}, rollout_id {rollout_id if rollout_id is not None else self.env_config['env_id']}, instruction_id {index * self.group_num + self.env_config['group_id']}"
        )

        self.rollout_cache = RolloutCache(
            env_id=self.env_config['env_id'],
            group_id=self.env_config['group_id'],
            tag=self.env_config['tag']
        )

        with self.thread_lock, self.env_step_limiter:
            observation, info, terminated = self.env.reset()

        self.logger.info(
            f"[EnvManager {self.env_config['env_id']}] env.reset() returned - observation {observation}, info: {info}"
        )
        max_steps = self.env_config["max_steps"]
        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": max_steps - self.rollout_cache.step,
                "messages": None,
                **info,
            }
        )
        if terminated:
            self.rollout_cache.terminated = True
        # self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.decode(
            llm_output['response_ids'][:-1], skip_special_tokens=False
        )
        # breakpoint()
        assert isinstance(responses, str)
        with self.thread_lock, self.env_step_limiter:
            observation, reward, terminated, truncated, info = self.env.step(action=responses)

        self.logger.info(
            f"[EnvManager {self.env_config['env_id']}] env.step() returned - observation {observation}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}"
        )

        suffix = info.pop("suffix", None)
        reward_model_suc = info.pop("reward_model_suc", True)
        max_steps = self.env_config["max_steps"]
        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True

        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['llm_response'] = responses
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": max_steps - self.rollout_cache.step,
                "messages": None
            }
        )
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix
        if not reward_model_suc:
            if "err_flag" in self.rollout_cache.history[-1]:
                self.rollout_cache.history[-1]["err_flag"].append("reward_model_err")
            else:
                self.rollout_cache.history[-1]["err_flag"] = ["reward_model_err"]

        if self.mode == "val" and self.pipeline_config.render_save_dir and hasattr(
            self.env, "render"
        ):
            frame = self.env.render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                self.rollout_cache.frames.append(frame)
        return self.rollout_cache

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        # assert not self.running
        assert "seed" in data.meta_info
        current_step = data.meta_info.get("current_step", None)
        self.running = True
        is_sync_training: bool = current_step is not None
        if is_sync_training:
            self.current_step = current_step
        assert self.current_step >= 0
        self.episode_id = 0
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        self.group_num = self.env_config['group_num']
        rollout_cache: RolloutCache = self.reset(self.current_step)
        self.episode_id += 1
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        while self.running:

            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output["stop_reason"]
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if self.running and (
                rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH
            ):
                self.logger.debug(
                    f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}"
                )
                extra_info = {"generate_time": sum(log_stats["generate_time"])}
                report_info = self.env.report_traj(extra_info)
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}_{self.current_step}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array(
                    [traj_group_id] * rollout.batch.batch_size[0], dtype=object
                )
                rollout.non_tensor_batch["traj_id"] = np.array(
                    [traj_id] * rollout.batch.batch_size[0], dtype=object
                )
                rollout.non_tensor_batch["report_info"] = np.array(
                    [report_info] * rollout.batch.batch_size[0], dtype=object
                )
                self.output_queue.put(
                    (self.env_config['group_id'], self.episode_id, start_step, rollout)
                )

                self.current_step += 1
                if not self.running or (
                    is_sync_training and self.episode_id >= self.manager_config.max_traj_per_env
                ):
                    self.rollout_cache: Optional[RolloutCache] = None
                    self.logger.debug(
                        f"env_id: {self.env_config['env_id']} max_traj_per_env {self.manager_config.max_traj_per_env} reached, stopping rollout loop"
                    )
                    break
                rollout_cache = self.reset(self.current_step)
                self.episode_id += 1

    def make_decision(self, rollout_cache: RolloutCache):
        # pdb.set_trace()
        prompt_ids, images, messages, prompt_ids_for_train, multi_modal_data, image_err = self.format_messages(
            rollout_cache
        )
        if len(prompt_ids_for_train) >= self.rl_config.training.seq_length:
            self.logger.warning(
                f"sequence_length = {self.rl_config.training.seq_length:} input_ids length = {len(prompt_ids_for_train)},"
                f"maybe you should increase the response_length"
            )
            print(f"DEBUG sequence_length too short {prompt_ids_for_train}", flush=True)
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        lm_output: DataProto = self.llm_proxy.generate(
            {
                "prompt_token_ids": prompt_ids,
                "images": images,
            }
        )

        if lm_output is None:
            print("DEBUG lm_output is None", flush=True)
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output['response_ids']
        content = self.rollout_cache.history[-1]
        # content["response_ids"] = response_ids
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
        content["rollout_log_probs"] = lm_output["output_logprobs"]
        content["multi_modal_data"] = multi_modal_data

        if image_err:
            if "err_flag" in content:
                content["err_flag"].append("get_image_err")
            else:
                content["err_flag"] = ["get_image_err"]

        lm_output["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def save_images(self, images, step):
        for (i, image) in enumerate(images):
            image.save(f"./gui_images/episode_{self.episode_id}_step_{step}_image_{i}.png")

    def format_messages(self, history: RolloutCache):
        messages = [
            {
                "role": "system",
                "content": self.agent_system_template
            },
        ]

        images = []
        image_paths = []
        current_cache = history.history[-1]
        for msg in current_cache["observation"]:
            messages.append(msg)
        # pdb.set_trace()
        for message in current_cache["observation"]:
            if not isinstance(message.get('content'), list):
                continue
            for content_item in message['content']:
                if content_item.get('type') != 'image_url':
                    continue
                image_url = content_item.get('image_url')
                if isinstance(image_url, dict):
                    image_path = image_url.get('url')
                else:
                    image_path = image_url
                if image_path:
                    image_paths.append(image_path)

        def get_image(image_path):
            get_image_suc = False
            image = None
            for _ in range(3):
                try:
                    if image_path.startswith('http'):
                        # Download image from URL
                        response = requests.get(image_path, timeout=10)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                    else:
                        # Load local image file
                        image = Image.open(image_path)

                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    get_image_suc = True
                    break
                except Exception as e:
                    print(f"Warning: Failed to load image from {image_path}: {e}")
                    continue
            return image, get_image_suc

        image_err = False
        for image_path in image_paths:
            image, get_image_suc = get_image(image_path)
            if get_image_suc:
                images.append(image)
            else:
                image_err = True
                images.append(self.fallback_image)

        lm_input_texts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # print(f"lm_input_texts :{lm_input_texts}", flush=True)
        prompt_id = self.tokenizer(lm_input_texts)["input_ids"]
        prompt_id = torch.tensor(prompt_id).reshape(-1)

        inputs = self.processor(
            text=[lm_input_texts], images=images, padding=False, return_mm_token_type_ids=True
        )
        input_id_for_train = inputs.input_ids[0]
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        multi_modal_data = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        # print(f"{prompt_id}", flush=True)
        return prompt_id, images, messages, input_id_for_train, multi_modal_data, image_err

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        # TODO: check inconsistent tokenization between successive encode-decode operations
        #  can potentially lead to a training crash. check token in token out
        #  the same as TrajEnvManager.

        # reward_model_suc打在了最后一个hist，需要先统计，再pop(-1)
        err_flag = []
        for hist in rollout_cache.history:
            if "err_flag" in hist:
                err_flag.extend(hist["err_flag"])
        err_flag = list(set(err_flag))

        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)

        samples: List[DataProto] = []
        episode_score = sum([i['reward'] for i in rollout_cache.history])
        print(
            f"[DEBUG] {rollout_cache.env_id}-{rollout_cache.step} episode_score: {episode_score}; err_flag: {err_flag}",
            flush=True
        )
        success = float(rollout_cache.history[-1].get('won', False))
        env_info = self.env.get_env_info()
        seq_len = self.rl_config.training.seq_length
        # pdb.set_trace()
        for step, history in enumerate(rollout_cache.history):
            messages = history["messages"]
            prompt_ids = history["prompt_ids_for_train"]
            response_ids = history["response_ids"]
            multi_modal_data = history["multi_modal_data"]
            rollout_log_probs = history["rollout_log_probs"]
            images = history["images"]

            if rollout_cache.env_id == 0 and self.episode_id == 1:
                self.save_images(images, step)

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
                input_ids, length=seq_len, pad_value=self.tokenizer.pad_token_id
            )

            attention_mask = pad_to_length(attention_mask, length=seq_len, pad_value=0)

            response_mask = pad_to_length(response_mask, length=seq_len, pad_value=0)
            prompt_mask = pad_to_length(prompt_mask, length=seq_len, pad_value=0)
            score_tensor = pad_to_length(score_tensor, length=seq_len, pad_value=0)

            rollout_log_probs = pad_to_length(rollout_log_probs, length=seq_len, pad_value=0.0)

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
                    "total_step": np.array([len(rollout_cache.history)], dtype=object),
                    "raw_rewards": np.array([history['reward']], dtype=object),
                    "success": np.array([success], dtype=object),
                    "instructions": np.array([env_info['instruction']], dtype=object),
                    "trace_ids": np.array([env_info['trace_id']], dtype=object),
                    "err_flag": np.array([err_flag], dtype=object),
                    "valid": np.array([len(err_flag) == 0], dtype=object),
                    "multi_modal_data": np.array([multi_modal_data], dtype=object),
                }
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
        return batch
