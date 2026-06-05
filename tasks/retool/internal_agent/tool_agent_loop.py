# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import logging
import os
import copy
import requests
import types
from uuid import uuid4
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from transformers import AutoTokenizer

from tasks.retool.internal_agent.tool_registry import initialize_tools_from_config
from tasks.retool.internal_agent.tool_parser import ToolParser, FunctionCall

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def http_sglang_engine_call(prompt_ids, sampling_params, request_id):
    url = "http://127.0.0.1:30000/generate"
    sampling_params = sampling_params.__dict__
    if 'seed' in sampling_params:
        sampling_params.pop('seed')  # seed is not supported by sglang

    if "max_new_tokens" in sampling_params:
        # sglang 有一个限制:
        # input_prompt_lens + max_new_tokens < max position embedding
        sampling_params["max_new_tokens"] -= 1

    payload = {"input_ids": prompt_ids, "sampling_params": sampling_params, "stream": False}
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    # 4. 解析结果
    result = response.json()
    return result.get("output_ids", [])


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    messages: List[Dict[str, str]]
    request_id: str


class ToolAgentLoop:
    _class_initialized = False

    def __init__(self, engine, tokenizer, **kwargs):
        """Initialize agent loop, each sample will have its own loop instance.
        """
        self.init_class(engine, tokenizer, **kwargs)

    @classmethod
    def init_class(cls, engine, tokenizer, **kwargs):
        if cls._class_initialized:
            # NOTE singleton
            return
        cls._class_initialized = True
        cls.loop = asyncio.get_running_loop()
        cls.engine = engine
        cls.tokenizer = tokenizer
        cls.max_user_turns = 8
        cls.max_assistant_turns = 8
        cls.max_parallel_calls = 1
        cls.max_tool_response_length = 256
        cls.tool_response_truncate_side = "middle"
        tool_config_path = "tasks/retool/internal_agent/sandbox_fusion_tool_config.yaml"
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
        ]
        cls.tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = 2048
        cls.response_length = 16384
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True
        )

    async def _call_infer_engine(self, prompt_ids, sampling_params, request_id, debug=False):
        """Call infer engine.
        Output has the following structure:
        output_ids: list / [1249, 11625, 419, 3491, 11]
        meta_info: dict / None
            id: str / 84ce7b3c3c404b22b12453e617e04dd3
            finish_reason: dict / None
                type: str / stop
                matched: int / 151645
            prompt_tokens: int / 436
            weight_version: str / default
            total_retractions: int / 0
            input_token_logprobs: list / [(None, 198, None)]
            output_token_logprobs: list / [(-0.016213351860642433, 1249, None), ...]
            completion_tokens: int / 1136
            cached_tokens: int / 435
            e2e_latency: float / 4.724360466003418
        
        """
        if debug:
            return http_sglang_engine_call(prompt_ids, sampling_params, str(request_id))
        inp = {
            "prompt_token_ids": prompt_ids,
        }
        _max_new_tokens = 32768 - len(prompt_ids) - 1
        if sampling_params.max_new_tokens:
            _max_new_tokens = min(_max_new_tokens, sampling_params.max_new_tokens)
        sampling_params.max_new_tokens = _max_new_tokens

        output = await self.engine.async_generate(inp, sampling_params, str(request_id))
        response_ids = output["output_ids"]
        return response_ids

    async def run(
        self, messages: list[dict[str, Any]], sampling_params: types.SimpleNamespace
    ) -> AgentLoopOutput:
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []
        total_messages = copy.deepcopy(messages)
        user_turns, assistant_turns = 0, 0
        while True:
            response_ids = await self._call_infer_engine(prompt_ids, sampling_params, request_id)
            # for debug
            # response_ids = http_sglang_engine_call(prompt_ids, sampling_params, str(request_id))

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            total_messages.append(
                {
                    "role": "assistant",
                    "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)
                },
            )

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # no tool calls
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            # call tools
            tasks = []
            for tool_call in tool_calls[:self.max_parallel_calls]:
                tasks.append(self._call_tool(tool_call))
            tool_responses = await asyncio.gather(*tasks)

            if any(isinstance(item, Exception) for item in tool_responses):
                break
            else:
                # 这里因为max_parallel_calls=1，所以直接取第一个 tool_response 了
                total_messages.append(tool_responses[0])

            # append tool_response_ids
            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.
                apply_chat_template(messages, add_generation_prompt=True, tokenize=True),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt):]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            messages=total_messages,
            request_id=request_id,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall) -> dict[str, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            # Be tolerant of raw control characters inside JSON strings.
            tool_args = json.JSONDecoder(strict=False).decode(tool_call.arguments)
            tool = self.tools[tool_name]

            instance_id = await tool.create()
            tool_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[:self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[
                    -length:]

        return {
            "role": "tool",
            "content": tool_response,
        }
