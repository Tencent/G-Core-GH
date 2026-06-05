# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import suppress

import ray

# import gym
with suppress(ImportError):
    from wxms_common.middleware.sandbox_device_for_training import SandboxDevice
    from wxms_common.model.dao.manager.wetest_cloud_device import WeTestCloudDeviceTag
    from wxms_common.middleware.sandbox_device_for_training import add_event
    from wxms_common.model.api.task import SpecialConfigForTraining
    from wxms_common.model.api.task import RunTrainingRolloutRequest
    from wxms_common.model.service.llm_training import (PlanningReinforcementLearningEventType)

import json
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .reward_model import get_reward_score_from_GLM_Qwen72B

with suppress(ImportError):
    from wxms_common.util.time_util import time_util

# reward_model_name = os.environ.get("REWARD_MODEL_NAME", "eval_kangyuqiao-Qwen2.5-VL-72B-Instruct-1119-12")
# print("Using reward model:", reward_model_name)

EMPTY_OBS = [
    {
        'role':
            'user',
        'content':
            [
                {
                    'type': 'image_url',
                    'image_url': '/mnt/geminigmceph/user_kangyuqiao/6_click_110_382.jpg'
                }, {
                    'type': 'text',
                    'text': 'description'
                }
            ]
    }
]

PROMPT_VLT = """你是一个小程序操作助手，当前指令是{}，请问在当前图像里我要怎么操作？
    当前只能有一种输出选项：只能回复接下来的一个动作，不能回复多个动作，并给出在页面的精确像素坐标（x y），最终输出格式严格遵循<answer>xxx</answer><summary>xxx</summary>这个格式。
    操作动作放在<answer>、</answer>标签内。操作动作只能在以下9 类集合选择：1. click x y(代表点击像素坐标 x,y)，2. input x y t（代表在x y 选中输入框并输入文本t） 3. finish（代表任务成功结束） 4. stop（代表任务无法进行 进程终止）5. scroll -400（代表向下滑动页面，其中400为下滑的像素值，这里400为一个固定值） 6. scroll 400（代表向上滑动页面，其中400为上滑的像素值，这里400同样是一个固定值） 7. scrollleft x y（代表从像素坐标x,y向左滑动） 8. scrollright x y（代表从像素坐标x,y向右滑动） 9. wait t（代表当前页面处于加载中，可以等待t秒）。

    单步操作总结放在<summary>、</summary>标签内，主要对本次的操作动作进行总结，单步总结操作示例为：\\"点击同意按钮,同意用户协议\\"、\\"点击确认按钮,选择当前门店并进入支付页面\\"、\\"点击韩元按钮\\"等。注意：如果是click操作，总结为\\"点击xxx按钮,yyy\\"，其中xxx为按钮的名字，yyy为用户操作意图；如果是input操作，总结为点击xxx并输入yyy，其中xxx为输入框名字，yyy为输入内容；如果是scroll -400操作，总结为\\"向下滑动页面,yyy\\"，其中yyy为下滑的目的；如果是scroll 400操作，总结为\\"向上滑动页面,yyy\\"，其中yyy为上滑的目的；如果是scrollleft操作，总结为\\"向左滑动xxx,yyy\\"，其中xxx为需要左滑的位置，yyy为左滑的目的；如果是scrollright操作，总结为\\"向右滑动xxx,yyy\\"，其中xxx为需要右滑的位置，yyy为右滑的目的；如果是wait操作，总结为\\"等待t秒\\"，其中t为等待的秒数。"""  # pylint: disable=C0301
PROMPT_VLT_V1 = """
你是一个顶级的 AI 小程序操作助手。你的任务是根据给定的高级指令，通过分析屏幕截图和历史操作，生成严谨的思考过程，并规划出最合理的下一步操作。

输入数据:

- 高级指令: {}
- 用户历史操作总结: {}
- 图像: 一系列操作截图，最近一张是当前图像。

最终输出格式严格遵循<think>xxx</think><sub_goal>xxx</sub_goal><answer>xxx</answer>。

思考过程放在<think>、</think>标签内。

规划子目标放在<sub_goal>、</sub_goal>标签内,主要是根据思考过程规划在这一步具体应该做什么操作，一定要简单明确可执行，不要模棱两可，不要增加多余的意图，不要包含额外的解释或目的说明。
规划子目标包括下面9类:
-如果规划子目标是点击操作，则输出\\"点击xxx\\"，其中xxx为点击的内容，如点击登录按钮，点击腾讯北京总部大楼标签
-如果是输入操作，则输出\\"在xxx输入yyy\\"，其中xxx为输入位置，yyy为输入内容，如\\"在搜索框输入test@domain.com\\"， \\"在密码框输入P@ssw0rd!\\"，\\"在文字腾讯北京总部大楼所在的位置，输入马连洼地铁站\\" 。
-如果是下滑操作，则输出\\"向下滑动页面,xxx\\"，其中xxx为下滑的目的。
-如果是上滑操作，则输出\\"向上滑动页面,xxx\\"，其中xxx为上滑的目的。
-如果是左滑操作，则输出\\"向左滑动xxx,yyy\\"，其中xxx为需要左滑的位置，yyy为左滑的目的。
-如果是右滑操作，则输出\\"向右滑动xxx,yyy\\"，其中xxx为需要右滑的位置，yyy为右滑的目的。
-如果需要请求用户接管，则输出\\"请求用户接管\\"。
-如果当前页面处于加载中，则输出\\"等待5s\\"。
-如果认为任务已完成，则输出\\"任务已完成\\"。

操作动作放在<answer>、</answer>标签内。操作动作只能在以下 10 类集合选择:
1. click x y (代表点击像素坐标 x y)
2. input x y t (代表在 x y 选中输入框并输入文本 t)
3. finish (代表任务成功结束)
4. stop (代表任务无法进行，进程终止)
5. scroll -400 (代表向下滑动页面，其中 400 为下滑的像素值，这里 400 为一个固定值)
6. scroll 400 (代表向上滑动页面，其中 400 为上滑的像素值，这里 400 同样是一个固定值)
7. scrollleft x y (代表从像素坐标 x y 向左滑动)
8. scrollright x y (代表从像素坐标 x y 向右滑动)
9. wait t (代表当前页面处于加载中，可以等待 t 秒)
10. calluser (代表当前指令中信息不足以完成任务，需要用户接管)

注意：
1.如果当前页面支持搜索功能，优先使用搜索来实现任务目标。
2.如果当前页面是支付二维码页面或确认付款页面，并且购物车中的商品已确认符合任务要求，则直接输出finish表示任务完成，无需再模拟点击支付按钮。
3.如果当前页面处于确认付款阶段，请检查购物车中的商品是否符合任务要求，若不符合要求，请先删除不需要的商品，再继续下一步操作。
4.如果当前页面有起点和终点信息,页面显示的起点和要求的起点不一致,请务必先点击页面上显示的起点进行修改；如果指令中没有指定起点，则使用默认起点，无需修改起点。
5.如果在整个页面只有用户协议、隐私协议、订票须知、游客须知等，必须向下滑动页面翻阅全部协议内容，只有在页面已经滑到底部，无法再滑动时，才可点击\\"我已阅读/同意\\"等类似按钮。
"""

_data_cache_lock = Lock()
_data_cache = None


def load_data_cache(data_source_path):
    """
    Load all data from JSONL file into memory cache (lazy loading)
    Args:
        data_source_path: Path to JSONL file
    Returns:
        list: List of processed data items
    """
    global _data_cache

    with _data_cache_lock:
        if _data_cache is None:
            print(f"[DataCache] Loading data from: {data_source_path}")
            _data_cache = []
            with open(data_source_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data_item = json.loads(line)
                        # Map fields: appid -> app_id, command -> instruction
                        processed_item = {
                            'app_id': data_item.get('appid', ''),
                            'instruction': data_item.get('command', ''),
                            # Keep other fields as well
                            'source_line': data_item.get('source_line', line_num),
                            'id': data_item.get('id', ''),
                            'scene': data_item.get('scene', ''),
                            'nickname': data_item.get('nickname', ''),
                            'intent': data_item.get('intent', ''),
                            'start_time': data_item.get('start_time', ''),
                            'end_time': data_item.get('end_time', ''),
                            'extinfo': data_item.get('extinfo', ''),
                            'data_index': line_num
                        }
                        _data_cache.append(processed_item)
                    except json.JSONDecodeError as e:
                        print(f"[DataCache] Failed to parse line {line_num}: {e}")
                        continue

            print(f"[DataCache] Loaded {len(_data_cache)} items into cache")

        return _data_cache


class Dataset:
    def __init__(
        self,
        group_id,
        group_size,
        data_source="/mnt/geminigmceph/user_jingtxu/dataset/xiaochengxu/rewrite_instruct_0914to23_930gen_mix_coffee_label_1115.jsonl"
    ):
        self.data_source = data_source
        self.group_id = group_id
        self.group_size = group_size
        self.valid_datas = []
        self.ori_datas = []
        self._load_file()
        self.last_update_time = time_util.get_millisecond_timestamp_of_current_time()
        print(
            f"Create Dataset with group_id={self.group_id}, group_size={self.group_size}, valid_datas={len(self.valid_datas)}, ori_datas={len(self.ori_datas)}"
        )

    def _load_file(self):
        datas = load_data_cache(self.data_source)
        self.ori_datas = datas[self.group_id::self.group_size]
        for data in self.ori_datas:
            if self._is_data_valid(data):
                self.valid_datas.append(data)

    def _refresh_valid_data(self):
        valid_datas = []
        for data in self.ori_datas:
            if self._is_data_valid(data):
                valid_datas.append(data)
        self.valid_datas = valid_datas
        self.last_update_time = time_util.get_millisecond_timestamp_of_current_time()

    def _is_data_valid(self, data_item):
        if data_item['start_time'] != '' and data_item['end_time'] != '':
            try:
                start_hour = int(data_item['start_time'])
                end_hour = int(data_item['end_time'])
                current_hour = time.localtime().tm_hour

                if not (start_hour <= current_hour < end_hour):
                    print(f"skip data: {data_item}")
                    return False
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Invalid time format, continuing anyway. Error: {e}, data_item: {data_item}"
                )
        return True

    def get_data(self, index):
        if time_util.get_millisecond_timestamp_of_current_time() - self.last_update_time > 600000:
            self._refresh_valid_data()

        if not self._is_data_valid(self.valid_datas[index % len(self.valid_datas)]):
            self._refresh_valid_data()
        data = self.valid_datas[index % len(self.valid_datas)]
        return data["app_id"], data["instruction"], data['extinfo']


class MiniprogramEnv:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of SokobanEnv.
    """
    def __init__(self, **kwargs):
        """Initialize the Sokoban eSvironment in this worker"""
        self._clear_state()
        # self.tag_filter_map = {
        #     "android_group_1": WeTestCloudDeviceTagFilter.ANDROID_GROUP_1,
        #     "android_group_2": WeTestCloudDeviceTagFilter.ANDROID_GROUP_2,
        #     "ios_rl_1": WeTestCloudDeviceTagFilter.IOS_RL_1,
        #     "ios_rl_2": WeTestCloudDeviceTagFilter.IOS_RL_2,
        # }
        self.dataset = None

        # self.reach_max_step = False

        # Reset failure statistics
        # self.reset_attempts = 0
        # self.reset_failures = 0
        # self.last_step_timestamp = None

        # Extract parameters from kwargs and assign to instance variables
        # self.prompt_vlt_path = kwargs.get('prompt_vlt_path', None)
        # self.prompt_vlt_v1_path = kwargs.get('prompt_vlt_v1_path', None)
        # self.sandbox_wait_model_base_url = kwargs.get('sandbox_wait_model_base_url', 'http://llmproxy-offline.lubanllm.polaris:9000/v1/test')
        # self.sandbox_wait_model_name = kwargs.get('sandbox_wait_model_name', 'xiaodezhang-waitmodel_7B_606_Qwen2.5-VL-7B-wait-0702_ck1100_export-0917-12')
        # self.vlt_base_url = kwargs.get('vlt_base_url', 'http://llmproxy-offline.lubanllm.polaris:9000/v1/test')
        # self.vlt_model_name = kwargs.get('vlt_model_name', 'eval_zhongpuwang-user_xiaodezhang_llm_luban_xiaochengxu_qwen25vl32b_v20250818_500_rl_v14_export-0915-16')
        # self.vlt_base_url_v1 = kwargs.get('vlt_base_url_v1', 'http://llmproxy-offline.lubanllm.polaris:9000/v1/test')
        # self.vlt_model_name_v1 = kwargs.get('vlt_model_name_v1', 'eval_zhongpuwang-user_xiaodezhang_llm_luban_xiaochengxu_qwen25vl32b_v20250827_700_rl_v17_export-0912-19')
        # self.tag_filter = kwargs.get('tag_filter', 'ios_rl_2')

    def initialize(
        self,
        batch_id: str,
        instruction_id: str,
        # app_id: str,
        # instruction: str,
        env_config: Dict,
        rollout_id: str | None = None,
        uin: int | None = None,
        base_url: str | None = None
    ):
        self._clear_state()
        if self.dataset is None:
            self.dataset = Dataset(
                group_id=env_config["group_id"],
                group_size=env_config["group_num"],
                data_source=env_config['config']["data_path"]
            )

        self.env_config = env_config
        self.init_start_time = time_util.get_millisecond_timestamp_of_current_time()
        self.training_id = env_config['training_id']
        self.batch_id = batch_id
        self.instruction_id = instruction_id
        if rollout_id is None:
            self.rollout_id = env_config['env_id']
        else:
            self.rollout_id = rollout_id
        self.sandbox_wait_model_base_url = env_config['config']["sandbox_wait_model_base_url"]
        self.sandbox_wait_model_name = env_config['config']["sandbox_wait_model_name"]
        self.vlt_base_url = env_config['config']["vlt_base_url"]
        self.vlt_model_name = env_config['config']["vlt_model_name"]
        self.vlt_base_url_v1 = env_config['config']["vlt_base_url_v1"]
        self.vlt_model_name_v1 = env_config['config']["vlt_model_name_v1"]
        self.reward_base_url = env_config['config']["reward_base_url"]
        self.reward_model_name = env_config['config']["reward_model_name"]

        # self.tag_filter = self.tag_filter_map[env_config['config']['tag_filter']]
        self.tag_filter = WeTestCloudDeviceTag(env_config['config']['tag_filter'])
        self.success_code = 1
        app_id, instruction, extinfo = self.dataset.get_data(batch_id)
        self.instruction = instruction

        if extinfo == '' or extinfo is None:
            extinfo_dict = {"longitude": None, "latitude": None}
        else:
            try:
                extinfo_dict = json.loads(extinfo)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to parse extinfo JSON string: {e}")
                print(f"extinfo content: {extinfo}")
                # Use default values if parsing fails
                extinfo_dict = {"longitude": None, "latitude": None}

        if "longitude" not in extinfo_dict or "latitude" not in extinfo_dict:
            extinfo_dict = {"longitude": None, "latitude": None}

        # TODO: change to config
        max_step = env_config['config']['max_steps']
        uin_config_enable_instruction_consistency = env_config['config'][
            'uin_config_enable_instruction_consistency']
        skip_flag = False
        success_flag_config_enable_check_planning_instruction_following = False
        success_flag_config_max_planning_length = 350

        special_config_for_training = SpecialConfigForTraining()
        special_config_for_training.max_step = max_step
        special_config_for_training.success_flag_config_enable_check_planning_instruction_following = success_flag_config_enable_check_planning_instruction_following
        special_config_for_training.success_flag_config_max_planning_length = success_flag_config_max_planning_length
        # special_config_for_training.uin_config_enable_instruction_consistency=uin_config_enable_instruction_consistency
        # special_config_for_training.success_flag_config_wait_stop_flag = True
        special_config_for_training.training_config_run_mode = 1
        special_config_for_training.training_config_check_model_output_format = True
        special_config_for_training.uin_config_force_restart_round = 50

        run_training_rollout_request = RunTrainingRolloutRequest(
            training_id=self.training_id,
            batch_id=str(batch_id),
            instruction_id=str(instruction_id),
            rollout_id=str(self.rollout_id),
            batch_round=batch_id,
            app_id=app_id,
            instruction=instruction,
            prompt_vlt=PROMPT_VLT,
            prompt_vlt_v1=PROMPT_VLT_V1,
            sandbox_wait_model_base_url=self.sandbox_wait_model_base_url,
            sandbox_wait_model_name=self.sandbox_wait_model_name,
            vlt_base_url=self.vlt_base_url,
            vlt_model_name=self.vlt_model_name,
            vlt_base_url_v1=self.vlt_base_url_v1,
            vlt_model_name_v1=self.vlt_model_name_v1,
            tag_filter=self.tag_filter,
            longitude=extinfo_dict['longitude'],
            latitude=extinfo_dict['latitude'],
            uin=uin,
            base_url=base_url,
            skip_flag=skip_flag,
            special_config_for_training=special_config_for_training
        )

        self.env = SandboxDevice(run_training_rollout_request=run_training_rollout_request)

        init_end_time = time_util.get_millisecond_timestamp_of_current_time()
        info = {
            "trace_id": self.env.trace_id,
            "time_period": init_end_time - self.init_start_time,
        }
        self.intialize_info_str = json.dumps(info, ensure_ascii=False)
        self.initialize_time = init_end_time - self.init_start_time

        print(f"Initialization: {self.intialize_info_str}")

    def reset(self):
        """Reset the environment with given seed"""
        reset_start_time = time_util.get_millisecond_timestamp_of_current_time()

        # self.env = SandboxDevice(app_id=app_id, query=query, batch_round=batch_round, uin=uin, base_url=base_url)
        # self.env = MockMiniprogramEnv()
        # if self.query_start_time != '' and self.query_end_time != '':
        #     try:
        #         start_hour = int(self.query_start_time)
        #         end_hour = int(self.query_end_time)
        #         current_hour = time.localtime().tm_hour
        #         print(f"Current Hour: {current_hour}, Query Hour: {start_hour} ~ {end_hour}")

        #         # Check if current time is NOT in the allowed time period
        #         if not (start_hour <= current_hour < end_hour):
        #             empty_obs = [{'role': 'user', 'content':[{'type':'image_url', 'image_url': '/mnt/geminigmceph/user_kangyuqiao/6_click_110_382.jpg'}, {'type':'text', 'text': 'description'}]}]
        #             self.done = True
        #             print("Not in time period, stop.")
        #             return empty_obs, {"time_period": 0.0, "success": False}
        #         else:
        #             print("In time period, continue.")
        #     except (ValueError, TypeError) as e:
        #         print(f"Warning: Invalid time format, continuing anyway. Error: {e}")
        # else:
        #     print("No time period specified, continue.")
        # # old
        # obs = self.env.reset()
        obs, sandbox_device_debug_info = self.env.reset()
        terminated = False
        if (obs == []):
            self.done = True
            obs = EMPTY_OBS
            terminated = True

        reset_end_time = time_util.get_millisecond_timestamp_of_current_time()
        # reset_failure_rate = self.reset_failures / self.reset_attempts if self.reset_attempts > 0 else 0.0

        info = {
            "trace_id": self.env.trace_id,
            "time_period": reset_end_time - reset_start_time,
            "success": not self.done,
        }
        infra_info_str = json.dumps(info, ensure_ascii=False)
        self.reset_time = reset_end_time - reset_start_time
        event = self.env.get_event(
            started_at=reset_start_time,
            ended_at=reset_end_time,
            event_type=PlanningReinforcementLearningEventType.RESET,
            infra_info=infra_info_str,
            sandbox_device_debug_info=sandbox_device_debug_info
        )
        add_event(event)
        self.reset_event_dto = event.model_dump_json()
        # print(f"RESET: {infra_info_str}")

        return obs, info, terminated

    def step(self, action):
        """Execute a step in the environment"""
        step_start_time = time_util.get_millisecond_timestamp_of_current_time()
        self.step_idx += 1
        if self.done:
            if self.success_code == 1:
                self.success_code = self.env.get_result_success_flag()
            empty_obs = EMPTY_OBS
            step_end_time = time_util.get_millisecond_timestamp_of_current_time()
            step_info = {
                "trace_id": self.env.trace_id,
                "time_period": step_end_time - step_start_time,
                "step_idx": self.step_idx,
                "action_is_effective": not self.done,
                "won": self.won,
            }
            truncated = False
            return empty_obs, 0.0, True, truncated, step_info

        # old
        # obs = self.env.step(action)
        obs, sandbox_device_debug_info = self.env.step(action)

        action_effective = not self.done
        reward_debug_str = ""
        reward_debug_dict = {}
        reward_model_time_period = 0.0
        if (obs == []):
            self.reward_model_usage = False

            obs = EMPTY_OBS
            self.success_code = self.env.get_result_success_flag()
            self.done = True
            self.won = (self.success_code == 0)
            if self.won:
                result = self.env.get_result()
                if result is None or len(result) == 0:
                    reward = 0.0
                else:
                    # Filter out None elements from the result list
                    filtered_result = [item for item in result if item is not None]
                    if len(filtered_result) == 0:
                        reward = 0.0
                    else:
                        # Time the reward model call
                        reward_model_start = time_util.get_millisecond_timestamp_of_current_time()
                        self.reward_model_usage = True
                        reward, reward_debug_dict = get_reward_score_from_GLM_Qwen72B(
                            filtered_result, self.reward_model_name
                        )
                        reward_model_end = time_util.get_millisecond_timestamp_of_current_time()
                        reward_model_time_period = reward_model_end - reward_model_start
                        self.episode_reward = reward
                        reward_debug_str = json.dumps(reward_debug_dict, ensure_ascii=False)
                        self.reward_debug_str = reward_debug_str
            else:
                reward = 0.0

        else:
            reward = 0.0

        step_end_time = time_util.get_millisecond_timestamp_of_current_time()
        self.step_time.append(step_end_time - step_start_time - reward_model_time_period)
        self.reward_time.append(reward_model_time_period)
        info = {
            "trace_id": self.env.trace_id,
            "time_period": step_end_time - step_start_time,
            "step_idx": self.step_idx,
            "action_is_effective": action_effective,
            "won": self.won,
            "reward_model_time_period": reward_model_time_period,
            "reward_model_suc": reward_debug_dict.get("reward_model_suc", True),
        }
        infra_info_str = json.dumps(info, ensure_ascii=False)

        event = self.env.get_event(
            started_at=self.init_start_time,
            ended_at=step_end_time,
            event_type=PlanningReinforcementLearningEventType.STEP,
            infra_info=infra_info_str,
            sandbox_device_debug_info=sandbox_device_debug_info
        )
        add_event(event)
        # print(f"STEP {self.step_idx}: {infra_info_str}")

        trancated = False
        return obs, reward, self.done, trancated, info

    def _clear_state(self):
        self.done = False
        self.won = False
        self.init_start_time = None
        self.reset_time = None
        self.initialize_time = None
        self.step_idx = 0
        self.episode_reward = 0.0
        self.reward_debug_str = ""
        self.use_reward_model = False
        self.reward_model_usage = False
        self.step_time = []
        self.reward_time = []
        self.instruction = None
        self.reset_event_dto = None

    def report_traj(self, extra_info):
        episode_end_time = time_util.get_millisecond_timestamp_of_current_time()
        episode_infra_info = {
            "trace_id": self.env.trace_id,
            "success_code": self.success_code,
            "instruction": self.instruction,
            "time_period": episode_end_time - self.init_start_time,
            "reset_time": self.reset_time,
            "initialize_time": self.initialize_time,
            "step_time": sum(self.step_time),
            "reward_time": sum(self.reward_time),
            "total_steps": self.step_idx,
            "success": self.won,
            "reward_model_usage": self.reward_model_usage,
            "uin": self.env.uin
        }
        episode_infra_info.update(extra_info)
        print(f"[DEBUG] Episode Infra Info: {episode_infra_info}")
        episode_reward_info = {
            "reward": self.episode_reward,
            "reward_debug_str": self.reward_debug_str
        }
        episode_infra_info_str = json.dumps(episode_infra_info, ensure_ascii=False)
        episode_reward_info_str = json.dumps(episode_reward_info, ensure_ascii=False)

        # Commented out for deferred reporting
        # add_rollout_event(
        #     started_at=self.init_start_time,
        #     ended_at=episode_end_time,
        #     training_id=self.training_id,
        #     batch_id=self.batch_id,
        #     instruction_id=self.instruction_id,
        #     rollout_id=self.rollout_id,
        #     initialize_info=self.intialize_info_str,
        #     reward_info=episode_reward_info_str,
        #     infra_info=episode_infra_info_str,
        # )

        # Return report info as dict for deferred reporting
        report_info = {
            "started_at": self.init_start_time,
            "ended_at": episode_end_time,
            "training_id": self.training_id,
            "batch_id": self.batch_id,
            "instruction_id": self.instruction_id,
            "rollout_id": self.rollout_id,
            "initialize_info": self.intialize_info_str,
            "reward_info": episode_reward_info_str,
            "infra_info": episode_infra_info_str,
            "reset_event_dto": self.reset_event_dto,
        }
        return report_info

    def get_env_info(self):
        return {"trace_id": self.env.trace_id, "instruction": self.instruction}

    def sample_random_action(self):
        return "<think>根据高级指令，任务目标是查询机票。当前界面是应用首页，点击功能明确的“机票”图标是进入机票查询流程的直接且必要的第一步。</think><sub_goal>在首页界面点击“机票”图标。</sub_goal><|im_end|>\n"
