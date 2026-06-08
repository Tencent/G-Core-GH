import json
import os
import random
import string
import unittest

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 22.  # A real function should probably actually get the temperature!


def get_current_wind_speed(location: str):
    """
    Get the current wind speed in km/h at a given location.

    Args:
        location: The location to get the wind speed for, in the format "City, Country"
    """
    return 6.  # A real function should probably actually get the wind speed!


tools = [get_current_temperature, get_current_wind_speed]


class QwenChatbot:
    def __init__(self, model_name="hf-hub/Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768
                                          )[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response


class WelmV4TokenizerTest(unittest.TestCase):

    def beautify_prompt(self, prompt: str) -> str:
        vis = str(prompt).replace('\n', '⏎\n').replace('\t', '→   ')  # .replace(' ', '·')
        return vis

    def test_welm_v4_tokenizer(self):
        # 本测试主要发现：
        # welm v4 的 tokenizer 符合基本的 SPM 性质

        # required welm v4 checkpoint from hf hub
        ckpt_path = "hf-hub/wechat/welm_v4_80A3B_128k_20251015"
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

        # auto tokenizer load welm v4 checkpoint，然后 对一个伪造的对话 apply chat template
        # Fake dialogue for the chat template (user/assistant message style)
        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
            {
                "role": "assistant",
                "content": "旋风哥吃的可能是真的农药"
            },
        ]
        '''
        test1
        ```<|im_start|>system⏎
        You are a friendly chatbot who always responds in the style of a pirate<|im_end|>⏎
        <|im_start|>user⏎
        老八吃的是真的屎吗？<|im_end|>⏎
        <|im_start|>assistant⏎
        老八吃的是真的屎。<|im_end|>⏎
        <|im_start|>user⏎
        旋风哥吃的是真的农药吗？<|im_end|>⏎
        <|im_start|>assistant⏎
        旋风哥吃的可能是真的农药<|im_end|>⏎
        ```
        '''
        chat_template = getattr(tokenizer, "apply_chat_template", None)
        assert chat_template is not None
        prompt1 = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            # add_generation_prompt=True,
            # truncation=True,
            # return_tensors=None
            add_special_tokens=False,
        )
        # print(f'test1\n```{self.beautify_prompt(prompt1)}```')

        # https://huggingface.co/docs/transformers/chat_templating
        # Some tokenizers add special <bos> and <eos> tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with apply_chat_template(tokenize=False), make sure you set add_special_tokens=False if you tokenize later to avoid duplicating these tokens. This isn't an issue if you use apply_chat_template(tokenize=True), which means it's usually the safer option!
        _prompt1 = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_special_tokens=True,
        )
        assert prompt1 == _prompt1
        '''
        test2
        ```<|im_start|>system⏎
        You are a friendly chatbot who always responds in the style of a pirate<|im_end|>⏎
        <|im_start|>user⏎
        老八吃的是真的屎吗？<|im_end|>⏎
        <|im_start|>assistant⏎
        老八吃的是真的屎。<|im_end|>⏎
        <|im_start|>user⏎
        旋风哥吃的是真的农药吗？<|im_end|>⏎
        <|im_start|>assistant⏎
        ```
        '''
        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
        ]
        prompt2 = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        # print(f'test2\n```{self.beautify_prompt(prompt2)}```')
        assert prompt2 == prompt1[:len(prompt2)]
        '''
        test3
        ```<|im_start|>system⏎
        You are a friendly chatbot who always responds in the style of a pirate<|im_end|>⏎
        <|im_start|>user⏎
        老八吃的是真的屎吗？<|im_end|>⏎
        <|im_start|>assistant⏎
        老八吃的是真的屎。<|im_end|>⏎
        <|im_start|>user⏎
        旋风哥吃的是真的农药吗？<|im_end|>⏎
        ```
        '''
        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
        ]
        prompt3 = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        # print(f'test3\n```{self.beautify_prompt(prompt3)}```')
        assert prompt3 == prompt2[:len(prompt3)]

    def test_welm_v4_tokenizer_id(self):
        # 本测试主要发现：
        # welm v4 的 tokenizer 的 token id 符合基本的 SPM 性质

        ckpt_path = "hf-hub/wechat/welm_v4_80A3B_128k_20251015"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
            {
                "role": "assistant",
                "content": "旋风哥吃的可能是真的农药"
            },
        ]

        chat_template = getattr(tokenizer, "apply_chat_template", None)
        assert chat_template is not None
        prompt1 = tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_special_tokens=False,
        )['input_ids']

        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
        ]
        prompt2 = tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True,
        )['input_ids']
        assert prompt2 == prompt1[:len(prompt2)]

        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
        ]
        prompt3 = tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=False,
        )['input_ids']
        assert prompt3 == prompt2[:len(prompt3)]

    def test_welm_v4_tokenizer_tool(self):
        # 本测试主要发现：
        # welm v4 的 tokenizer 的 tool calling 符合预期

        # required welm v4 checkpoint from hf hub
        ckpt_path = "hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240"
        # ckpt_path = "hf-hub/Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

        # auto tokenizer load welm v4 checkpoint，然后 对一个伪造的对话 apply chat template
        # Fake dialogue for the chat template (user/assistant message style)
        # 额外看看 tool 的情况
        # https://huggingface.co/docs/transformers/chat_content_patterns
        # https://huggingface.co/docs/transformers/chat_extras
        # https://huggingface.co/docs/transformers/chat_response_parsing
        # https://huggingface.co/blog/qwen-3-chat-template-deep-dive
        # @astrachang 看看起来 s1s 的数据格式虽然处理奇怪，但从字符串看是正确的。
        # 问了下 orion 说是 welm v4 tokenizer 不支持 enable thinking 这个选项，先不管这个事情。

        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "北京今天天气如何？"
            },
            {
                "role":
                    "assistant",
                "tool_calls":
                    [
                        {
                            "type": "function",
                            "function":
                                {
                                    "name": "get_current_temperature",
                                    "arguments": {
                                        "location": "Paris, France",
                                        "unit": "celsius"
                                    }
                                },
                        },
                    ],
            },
            {
                "role": "tool",
                "content": "weather france is 晴天"
            },
            {
                "role": "assistant",
                "content": "22 度"
            },
            {
                "role": "user",
                "content": "孕妇能不能吃鲨鱼？"
            },
            # {"role": "assistant", "content": "孕妇可以吃北极熊"},
        ]

        chat_template = getattr(tokenizer, "apply_chat_template", None)
        assert chat_template is not None
        prompt1 = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_special_tokens=False,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        print(f'test_welm_v4_tokenizer_tool\n```{self.beautify_prompt(prompt1)}```')

    def test_qwen3_think_history(self):
        # 本测试主要发现：
        # qwen3 的 think 没有单独的 chat history field，就是藏在 assistant 的 content 里。
        # orion 暂时不用，先 <think>\n</think> 写死。welm v4 行为类似。

        # 额外看看 think 的情况
        chatbot = QwenChatbot()

        # First input (without /think or /no_think tags, thinking mode is enabled by default)
        user_input_1 = "How many r's in strawberries?"
        print(f"User: {user_input_1}")
        response_1 = chatbot.generate_response(user_input_1)
        print(f"Bot: {response_1}")
        print("----------------------")

        # Second input with /no_think
        user_input_2 = "Then, how many r's in blueberries? /no_think"
        print(f"User: {user_input_2}")
        response_2 = chatbot.generate_response(user_input_2)
        print(f"Bot: {response_2}")
        print("----------------------")

        # Third input with /think
        user_input_3 = "Really? /think"
        print(f"User: {user_input_3}")
        response_3 = chatbot.generate_response(user_input_3)
        print(f"Bot: {response_3}")

    def test_welm_v4_tokenizer_concat_ok(self):
        # 本测试主要发现：
        # tokenize 完整 history 和 分段 tokenize 再 concat 的 token id 是否一致。
        # 场景：serving 时增量 tokenize 新 response，而不是重新 tokenize 整段 history。
        # 注意：给到的 welm v4 版本没有 enable thinking 这个选项，先不管这个事情。

        ckpt_path = "hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

        history = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "老八吃的是真的屎吗？"
            },
            {
                "role": "assistant",
                "content": "老八吃的是真的屎。"
            },
            {
                "role": "user",
                "content": "旋风哥吃的是真的农药吗？"
            },
        ]
        new_response = {"role": "assistant", "content": "<think>\n</think>\n\n旋风哥吃的可能是真的农药"}

        # --- 方法 A：tokenize 完整 history（含 new response） ---
        full_convo = history + [new_response]
        ids_full = tokenizer.apply_chat_template(
            full_convo,
            tokenize=True,
            add_special_tokens=False,
        )['input_ids']

        # --- 方法 B：先 tokenize history，再单独 tokenize response 并 append ---
        # step1: tokenize history with generation prompt
        ids_prefix = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True,
        )['input_ids']

        # step2: 通过文本差值得到 suffix（response content + <|im_end|> + \n）
        full_text = tokenizer.apply_chat_template(
            full_convo,
            tokenize=False,
            add_special_tokens=False,
        )
        prefix_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        # 检查 prefix_text 和 full_text[:len(prefix_text)] 是否完全一致
        assert prefix_text == full_text[:len(prefix_text)], (
            f"prefix_text and full_text[:len(prefix_text)] are different!\n"
            f"prefix_text=```{self.beautify_prompt(prefix_text)}```\n"
            f"full_text[:prefix]=```{self.beautify_prompt(full_text[:len(prefix_text)])}```"
        )
        suffix_text = full_text[len(prefix_text):]
        # print(f"suffix_text: ```{self.beautify_prompt(suffix_text)}```")

        # step3: 单独 tokenize suffix
        ids_suffix = tokenizer.encode(suffix_text, add_special_tokens=False)

        # 只用 new_response，产生对应的 apply chat template 之后 suffix_test，以及对应的 token id
        # 只用 new_response 生成 chat template 文本
        # 这段逻辑有点奇葩，预估会在业务代码里产生一大堆 specific 的代码
        response_template_text = f'{new_response["content"]}{tokenizer.eos_token}\n'
        # print(f"response_template_text: ```{self.beautify_prompt(response_template_text)}```")
        assert '<think>\n</think>\n\n' + suffix_text == response_template_text
        # 获得对应的 token id
        response_template_ids = tokenizer.encode(
            response_template_text[len('<think>\n</think>\n\n'):], add_special_tokens=False
        )
        # print(f"response_template_ids: {response_template_ids}")
        assert response_template_ids == ids_suffix

        # --- 断言：两种方式得到的 token id 完全一致 ---
        ids_concat = ids_prefix + ids_suffix
        # print(f"len(ids_full)={len(ids_full)}, len(ids_prefix)={len(ids_prefix)}, len(ids_suffix)={len(ids_suffix)}")
        self.assertEqual(
            ids_full, ids_concat, f"token ids mismatch!\n"
            f"  full:   {ids_full}\n"
            f"  concat: {ids_concat}\n"
            f"  prefix matches: {ids_full[:len(ids_prefix)] == ids_prefix}\n"
            f"  suffix full:   {ids_full[len(ids_prefix):]}\n"
            f"  suffix concat: {ids_suffix}"
        )

    def test_welm_v4_tokenizer_vocab_not_merge_newline(self):
        # 遍历 vocab，查找包含换行符的 token，并输出它们
        ckpt_path = "hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        vocab = tokenizer.get_vocab()
        found = False
        for token, idx in vocab.items():
            readable = tokenizer.decode(idx)
            if "\n" in readable:
                # print(f"Token with newline: readable={readable}, id={idx}")
                found = True

        # self.skipTest("welm v4 的 tokenizer 包含了 \n，跳过这个测试")
        # assert not found

        convo = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate"
            },
            {
                "role": "user",
                "content": "北京今天天气如何？"
            },
            {
                "role": "assistant",
                "content": "22 度"
            },
            {
                "role": "user",
                "content": "孕妇能不能吃鲨鱼？"
            },
            {
                "role": "assistant",
                "content": "孕妇可以吃北极熊"
            },
        ]

        chat_template = getattr(tokenizer, "apply_chat_template", None)
        assert chat_template is not None
        prompt1 = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        # print(f'prompt1: ```{self.beautify_prompt(prompt1)}```')

        prompt1 = '<|im_start|>assistant\n'
        ids1 = tokenizer.encode(prompt1, add_special_tokens=False)
        assert ids1 == [154741, 84472, 183]

        # 只要有 trailing 的 space，可能会发生意外。
        prompt1 = '<|im_start|>assistant\n '
        ids1 = tokenizer.encode(prompt1, add_special_tokens=False)
        assert ids1 == [154741, 84472, 183, 205]

        # 只要有 trailing 的 space，可能会发生意外。
        prompt1 = '<|im_start|>assistant\n\n'
        ids1 = tokenizer.encode(prompt1, add_special_tokens=False)
        assert ids1 == [154741, 84472, 257]

        # pretokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(prompt1)
        # for token, (start, end) in pretokenized:
        #     readable = prompt1[start:end]
        #     print(f"chunk: {readable!r}  offsets: ({start}, {end})  raw: {token!r}")

    def test_welm_v4_tokenizer_concat_ok_random(self):
        # 随机生成 100 组不同的对话，验证 tokenize(full) == tokenize(prefix) + encode(suffix)
        # 先跳过，晚点 fix。
        # self.skipTest("skip test_welm_v4_tokenizer_concat_ok_random")

        ckpt_path = "hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

        rng = random.Random(42)

        cn_chars = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出会三要于下得可你年生"
        cn_puncts = "，。！？、；：" "''【】（）《》——…"
        en_words = [
            "hello",
            "world",
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "AI",
            "model",
            "token",
            "chat",
            "system",
            "user",
            "deep",
            "learning",
            "neural",
            "network",
            "transformer",
            "attention",
        ]
        special_frags = [
            "\n",
            "\t",
            "  ",
            "🔥",
            "👍",
            "😂",
            "🎉",
            "<div>",
            "&amp;",
            "hello\nworld",
            "1+1=2",
            "C:\\Users\\test",
            "/usr/bin/python",
            "https://example.com",
            "def foo():\n    return 42",
            '{"key": "value"}',
        ]

        def rand_text():
            """生成一段随机混合文本"""
            parts = []
            n_parts = rng.randint(1, 8)
            for _ in range(n_parts):
                kind = rng.choice(["cn", "en", "special", "digits", "ascii"])
                if kind == "cn":
                    length = rng.randint(2, 30)
                    parts.append(
                        "".join(rng.choice(cn_chars)
                                for _ in range(length)) + rng.choice(cn_puncts)
                    )
                elif kind == "en":
                    length = rng.randint(1, 10)
                    parts.append(" ".join(rng.choice(en_words) for _ in range(length)))
                elif kind == "special":
                    parts.append(rng.choice(special_frags))
                elif kind == "digits":
                    parts.append(str(rng.randint(0, 999999)))
                elif kind == "ascii":
                    length = rng.randint(1, 20)
                    parts.append("".join(rng.choice(string.printable) for _ in range(length)))
            return " ".join(parts)

        n_cases = 100
        n_passed = 0
        failures = []

        for i in range(n_cases):
            n_turns = rng.randint(1, 5)
            history = []
            history.append({"role": "system", "content": rand_text()})
            for _ in range(n_turns):
                history.append({"role": "user", "content": rand_text()})
                history.append({"role": "assistant", "content": rand_text()})
            history.append({"role": "user", "content": rand_text()})

            response_content = rand_text()
            empty_think_text = '<think>\n</think>\n\n'
            new_response = {"role": "assistant", "content": empty_think_text + response_content}

            full_convo = history + [new_response]
            ids_full = tokenizer.apply_chat_template(
                full_convo,
                tokenize=True,
                add_special_tokens=False,
            )['input_ids']

            ids_prefix = tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_special_tokens=False,
                add_generation_prompt=True,
            )['input_ids']

            # 在没有 cot 的情况下，apply chat template 会用 <think>\n</think>\n\n 作为 prefix，这个 prefix 有个很糟糕的地方在于他实际上会
            # 和后续的内容粘连...
            # 实际上 generate 的时候，必定是先出 <think>\n</think>\n\n 再出后续的内容，在 sglang 看来没有粘连
            # 但是训练的时候直接 encode，其实会有粘连... 比如 output 用 \n 开头的时候。
            # 为什么边界不融合？因为有 pre-tokenization https://huggingface.co/learn/llm-course/chapter6/4
            empty_think_ids = tokenizer.encode(empty_think_text, add_special_tokens=False)
            # print(f'empty_think_ids: {empty_think_ids}')
            # print(f'ids_prefix: {ids_prefix}')
            assert ids_prefix[-len(empty_think_ids):] == empty_think_ids
            assert empty_think_ids == [154776, 183, 154777, 257]
            ids_prefix = ids_prefix[:-len(empty_think_ids)]  # 去掉 think 部分

            full_text = tokenizer.apply_chat_template(
                full_convo,
                tokenize=False,
                add_special_tokens=False,
            )
            prefix_text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=True,
            )
            assert prefix_text.endswith(empty_think_text)
            prefix_text = prefix_text[:-len(empty_think_text)]  # 去掉 think 部分
            if prefix_text != full_text[:len(prefix_text)]:
                failures.append((i, "prefix text mismatch"))
                continue

            suffix_text = full_text[len(prefix_text):]
            # print(f'\n{full_text=}\n{prefix_text=}\n{suffix_text=}')
            ids_suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
            ids_concat = ids_prefix + ids_suffix
            # print(f'\n{ids_full=}\n{ids_concat=}\n{ids_prefix=}\n{ids_suffix=}')

            if ids_full == ids_concat:
                n_passed += 1
            else:
                failures.append(
                    (
                        i, f"ids mismatch: len(full)={len(ids_full)} len(concat)={len(ids_concat)} "
                        f"prefix_ok={ids_full[:len(ids_prefix)] == ids_prefix} "
                        f"suffix_full={ids_full[len(ids_prefix):]} "
                        f"suffix_encode={ids_suffix} "
                        f"suffix_text={suffix_text} "
                    )
                )

        # print('--------------------------------')
        # print(f"^{tokenizer.decode([183])}$")
        # print('--------------------------------')
        # print(f"^{tokenizer.decode([597])}$")
        # print('--------------------------------')
        # print(f"^{tokenizer.decode([183, 597])}$")
        # print('--------------------------------')

        print(f"passed {n_passed}/{n_cases}")
        self.assertEqual(
            len(failures), 0, f"{len(failures)}/{n_cases} cases failed:\n" +
            "\n".join(f"  case {idx}: {msg}" for idx, msg in failures[:10])
        )

    def test_data_240414(self):
        # 一个测试数据，临时用的，没有不用管
        # 确保数据正确性，和 orion 确认过，不需要改动。
        data_p = '/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/wechat/data/oriontian_0413_part1_2_3_4_mix.jsonl.parser_doc_list'
        if not os.path.exists(data_p):
            self.skipTest("skip test_data_240414")

        with open(data_p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        json_objs = []
        for line in lines:
            line = line.strip()
            assert line
            obj = json.loads(line)
            json_objs.append(obj)

        for obj in json_objs:
            msgs = obj['messages']
            for msg in msgs:
                if msg['role'] == 'assistant':
                    assert msg['text'].startswith('<think>\n</think>\n\n')

    def test_data_240414_concat_ok(self):
        # 真实数据版 test_welm_v4_tokenizer_concat_ok_random：
        # 对每条对话，把最后一个 assistant turn 作为 new_response，验证
        #   tokenize(full) == tokenize(history + gen_prompt 去掉 <think>\n</think>\n\n) + encode(suffix_text)
        # 依赖 test_data_240414 保证的前提：每个 assistant msg 的 text 都以 <think>\n</think>\n\n 开头。
        data_p = '/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/wechat/data/oriontian_0413_part1_2_3_4_mix.jsonl.parser_doc_list'
        if not os.path.exists(data_p):
            self.skipTest("skip test_data_240414_concat_ok")

        ckpt_path = "hf-hub/wechat/oriontian_0407_part1_2_mix_sft_hf/epoch_008_step_0000240"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

        empty_think_text = '<think>\n</think>\n\n'
        empty_think_ids = tokenizer.encode(empty_think_text, add_special_tokens=False)

        # 采样上限，避免跑太久；设为 None 全量跑。
        # 数据文件 1.3G，用 iterator 只读前 N 行，不要 readlines() 一次性吃满内存。
        max_convos = 1000
        total, passed = 0, 0
        failures = []

        sampled = []
        with open(data_p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_convos is not None and i >= max_convos:
                    break
                sampled.append(line)

        for line_idx, line in enumerate(tqdm(sampled, desc="concat_ok")):
            line = line.strip()
            assert line
            obj = json.loads(line)
            msgs = obj['messages']

            # 真实数据格式（见 test_data_240414 及服务器上的抽样确认）：
            #   - 每个 msg 有 role/text/train（assistant 还可能有 think_info_cut），
            #   - 对话顺序严格为 (system, user, assistant, ..., user, assistant)，
            #   - 每个 assistant.text 以 <think>\n</think>\n\n 开头。
            # 这里只做 text -> content 字段重命名，apply_chat_template 的模板吃的是 content。
            normalized = [{"role": m["role"], "content": m["text"]} for m in msgs]

            # 数据前提：最后一条 msg 一定是 assistant，前一条一定是 user。
            assert normalized[-1]['role'] == 'assistant', (f"line {line_idx}: 对话不是以 assistant 结尾")
            ai = len(normalized) - 1
            history = normalized[:ai]
            new_response = normalized[ai]

            assert history and history[-1]['role'] == 'user', (
                f"line {line_idx}: 最后一个 assistant 之前不是 user"
            )
            assert new_response['content'].startswith(empty_think_text), (
                f"line {line_idx}: assistant content 必须以 {empty_think_text!r} 开头"
            )

            total += 1

            full_convo = history + [new_response]
            ids_full = tokenizer.apply_chat_template(
                full_convo,
                tokenize=True,
                add_special_tokens=False,
            )['input_ids']
            full_text = tokenizer.apply_chat_template(
                full_convo,
                tokenize=False,
                add_special_tokens=False,
            )

            ids_prefix = tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_special_tokens=False,
                add_generation_prompt=True,
            )['input_ids']
            prefix_text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=True,
            )

            # add_generation_prompt=True 会在末尾自动加 <think>\n</think>\n\n，
            # 它和 new_response 开头的 <think>\n</think>\n\n 重合，要去掉再拼接。
            assert ids_prefix[-len(empty_think_ids):] == empty_think_ids, (
                f"line {line_idx}: prefix ids 末尾不是 empty_think_ids"
            )
            ids_prefix = ids_prefix[:-len(empty_think_ids)]

            assert prefix_text.endswith(empty_think_text), (
                f"line {line_idx}: prefix_text 末尾不是 {empty_think_text!r}"
            )
            prefix_text = prefix_text[:-len(empty_think_text)]

            if prefix_text != full_text[:len(prefix_text)]:
                failures.append((line_idx, "prefix text 和 full text 前缀不一致"))
                continue

            suffix_text = full_text[len(prefix_text):]
            ids_suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
            ids_concat = ids_prefix + ids_suffix

            if ids_full == ids_concat:
                passed += 1
            else:
                failures.append(
                    (
                        line_idx,
                        f"ids mismatch: len(full)={len(ids_full)} len(concat)={len(ids_concat)} "
                        f"prefix_ok={ids_full[:len(ids_prefix)] == ids_prefix} "
                        f"suffix_full_head={ids_full[len(ids_prefix):][:20]} "
                        f"suffix_encode_head={ids_suffix[:20]} "
                        f"suffix_text_head={suffix_text[:80]!r}"
                    )
                )

        print(f"passed {passed}/{total} convos")
        self.assertEqual(
            len(failures), 0, f"{len(failures)}/{total} cases failed:\n" +
            "\n".join(f"  line {li}: {msg}" for li, msg in failures[:10])
        )
