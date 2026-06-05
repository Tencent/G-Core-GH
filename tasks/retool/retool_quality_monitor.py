# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# ReTool Quality Monitor - 用于观测模型生成质量的日志模块
#
# 功能：
#   1. 每个 PPO Step 结束时从 Rollout 数据中提取固定数量样本
#   2. 详细打印 Prompt、Model Reasoning、Tool Calls、Execution Result、Final Answer、Reward
#   3. 高亮标注异常情况（空响应、截断、语法错误等）
#   4. 追加写入独立日志文件

import os
import re
import json
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from megatron.training.global_vars import get_args, get_tokenizer

# =====================================================================================
# 配置常量
# =====================================================================================

# 监控日志文件路径 - 可通过环境变量覆盖
MONITOR_LOG_PATH = os.environ.get(
    "RETOOL_MONITOR_LOG_PATH", "./retool_output/log/retool_quality_monitor.log"
)

# 每个 PPO Step 采样的样本数量
MONITOR_SAMPLE_COUNT = int(os.environ.get("RETOOL_MONITOR_SAMPLE_COUNT", "10"))

# 是否同时打印到 stdout
MONITOR_PRINT_TO_STDOUT = os.environ.get("RETOOL_MONITOR_PRINT_STDOUT",
                                         "0").lower() in ("1", "true", "yes")

# 正则表达式
CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# =====================================================================================
# 辅助函数
# =====================================================================================


def extract_last_boxed(text: str) -> Optional[str]:
    """从文本中提取最后一个 \\boxed{...} 的内容"""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    num_open = 0
    right_idx = None

    while i < len(text):
        if text[i] == "{":
            num_open += 1
        elif text[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_idx = i
                break
        i += 1

    if right_idx is None:
        return None

    boxed_str = text[idx:right_idx + 1]
    content = boxed_str[7:-1]  # len("\\boxed{") = 7
    return content


def extract_code_blocks(text: str) -> List[str]:
    """提取所有 Python 代码块"""
    return CODE_PATTERN.findall(text)


def format_separator(char: str = "=", length: int = 80) -> str:
    """生成分隔线"""
    return char * length


def truncate_text(text: str, max_length: int = 500, suffix: str = "\n...[TRUNCATED]") -> str:
    """截断过长文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def detect_anomalies(sample_data: Dict[str, Any]) -> List[str]:
    """检测样本中的异常情况"""
    anomalies = []

    # 检查空响应
    response_text = sample_data.get("response_text", "")
    if not response_text or len(response_text.strip()) == 0:
        anomalies.append("⚠️ EMPTY_RESPONSE: 模型响应为空")

    # 检查是否被截断（基于 token 限制）
    is_truncated = sample_data.get("is_truncated", False)
    if is_truncated:
        anomalies.append("⚠️ TRUNCATED: 响应因 Token 限制被截断")

    # 检查是否有 boxed 答案
    final_answer = sample_data.get("final_answer")
    if final_answer is None or final_answer == "":
        if "\\boxed" not in response_text:
            anomalies.append("⚠️ NO_BOXED_ANSWER: 未找到 \\boxed{} 格式的最终答案")

    # 检查代码执行错误
    execution_results = sample_data.get("execution_results", [])
    for i, result in enumerate(execution_results):
        if result.get("success") is False:
            error_msg = result.get("output", "Unknown error")[:100]
            anomalies.append(f"⚠️ CODE_EXECUTION_ERROR (Turn {i+1}): {error_msg}")

    # 检查 Python 语法错误
    code_blocks = sample_data.get("code_blocks", [])
    for i, code in enumerate(code_blocks):
        if "SyntaxError" in str(sample_data.get("execution_results", [])):
            anomalies.append(f"⚠️ SYNTAX_ERROR (Code Block {i+1})")

    # 检查 reward 异常
    reward = sample_data.get("reward", None)
    if reward is not None and reward < -0.9:
        anomalies.append(f"⚠️ LOW_REWARD: 奖励分数过低 ({reward:.3f})")

    return anomalies


def extract_sample_data_from_messages(
    messages: List[Dict[str, str]],
    gt_label: Dict[str, Any],
    tokens: Optional[torch.Tensor] = None,
    prompt_length: int = 0,
    tokenizer=None,
) -> Dict[str, Any]:
    """从对话消息中提取样本数据"""
    sample_data = {
        "prompt": "",
        "tokenized_prompt": "",  # 新增：完整的tokenized prompt
        "response_text": "",
        "code_blocks": [],
        "execution_results": [],
        "final_answer": None,
        "ground_truth": "",
        "reward": None,
        "num_turns": len(messages),
        "is_truncated": False,
    }

    # 提取 prompt（用户问题）- 原始messages中的内容
    for msg in messages:
        if msg.get("role") == "user":
            sample_data["prompt"] = msg.get("content", "")
            break

    # 新增：从tokens解码完整的tokenized prompt（包含system prompt和tools）
    if tokens is not None and tokenizer is not None and prompt_length > 0:
        try:
            if isinstance(tokens, torch.Tensor):
                prompt_tokens = tokens[:prompt_length].tolist()
            else:
                prompt_tokens = tokens[:prompt_length]
            sample_data["tokenized_prompt"] = tokenizer._tokenizer.decode(
                prompt_tokens, skip_special_tokens=False
            )
        except Exception as e:
            sample_data["tokenized_prompt"] = f"[Decode Error: {e}]"

    # 提取所有 assistant 响应
    assistant_responses = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            assistant_responses.append(content)

            # 提取代码块
            code_blocks = extract_code_blocks(content)
            sample_data["code_blocks"].extend(code_blocks)

            # 检查是否有 tool_calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    if func.get("name") == "code_interpreter":
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                            code = args.get("code", "")
                            if code:
                                sample_data["code_blocks"].append(code)
                        except:
                            pass

        # 提取工具执行结果
        elif msg.get("role") == "tool":
            tool_output = msg.get("content", "")
            success = not tool_output.startswith("Error:")
            sample_data["execution_results"].append({
                "output": tool_output,
                "success": success,
            })

    sample_data["response_text"] = "\n".join(assistant_responses)

    # 提取最终答案
    if assistant_responses:
        last_response = assistant_responses[-1]
        sample_data["final_answer"] = extract_last_boxed(last_response)

    # 提取 ground truth 和 reward
    if isinstance(gt_label, dict):
        sample_data["ground_truth"] = gt_label.get(
            "ground_truth", gt_label.get("answer", str(gt_label))
        )
        sample_data["reward"] = gt_label.get("retool_reward", gt_label.get("reward"))
        sample_data["acc"] = gt_label.get("retool_acc", 0)
        sample_data["is_truncated"] = gt_label.get("train_data",
                                                   {}).get("reach_context_len_thr", False)
    else:
        sample_data["ground_truth"] = str(gt_label)

    return sample_data


def extract_sample_data_from_tokens(
    tokens: torch.Tensor,
    sequence_length: int,
    prompt_length: int,
    gt_label: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    """从 token 序列中提取样本数据（当 messages_traj 不可用时的 fallback）"""
    sample_data = {
        "prompt": "",
        "tokenized_prompt": "",  # 新增：完整的tokenized prompt
        "response_text": "",
        "code_blocks": [],
        "execution_results": [],
        "final_answer": None,
        "ground_truth": "",
        "reward": None,
        "num_turns": 0,
        "is_truncated": False,
    }

    # 解码 tokens
    if isinstance(tokens, torch.Tensor):
        tokens_list = tokens[:sequence_length].tolist()
    else:
        tokens_list = tokens[:sequence_length]

    full_text = tokenizer._tokenizer.decode(tokens_list, skip_special_tokens=False)

    # 简单分割 prompt 和 response
    # 尝试查找常见的分隔标记
    markers = ["<|im_start|>assistant", "<|assistant|>", "Assistant:", "assistant\n"]
    split_idx = -1
    for marker in markers:
        idx = full_text.find(marker)
        if idx != -1:
            split_idx = idx
            break

    if split_idx != -1:
        sample_data["prompt"] = full_text[:split_idx].strip()
        sample_data["response_text"] = full_text[split_idx:].strip()
        # tokenized_prompt 就是完整的 prompt 部分
        sample_data["tokenized_prompt"] = full_text[:split_idx].strip()
    else:
        # 基于 prompt_length 分割
        prompt_tokens = tokens_list[:prompt_length]
        response_tokens = tokens_list[prompt_length:]
        sample_data["prompt"] = tokenizer._tokenizer.decode(
            prompt_tokens, skip_special_tokens=False
        )
        sample_data["response_text"] = tokenizer._tokenizer.decode(
            response_tokens, skip_special_tokens=False
        )
        # tokenized_prompt 就是 prompt 部分
        sample_data["tokenized_prompt"] = sample_data["prompt"]

    # 提取代码块
    sample_data["code_blocks"] = extract_code_blocks(sample_data["response_text"])

    # 提取最终答案
    sample_data["final_answer"] = extract_last_boxed(sample_data["response_text"])

    # 提取 ground truth 和 reward
    if isinstance(gt_label, dict):
        sample_data["ground_truth"] = gt_label.get(
            "ground_truth", gt_label.get("answer", str(gt_label))
        )
        sample_data["reward"] = gt_label.get("retool_reward", gt_label.get("reward"))
        sample_data["acc"] = gt_label.get("retool_acc", 0)
        sample_data["is_truncated"] = gt_label.get("train_data",
                                                   {}).get("reach_context_len_thr", False)
    else:
        sample_data["ground_truth"] = str(gt_label)

    return sample_data


def format_sample_log(
    sample_idx: int,
    ppo_step: int,
    sample_data: Dict[str, Any],
    anomalies: List[str],
) -> str:
    """格式化单个样本的日志输出"""
    lines = []

    # 样本头部
    lines.append(format_separator("-", 60))
    lines.append(f"📊 Sample {sample_idx + 1} | PPO Step: {ppo_step}")
    lines.append(format_separator("-", 60))

    # 异常警告
    if anomalies:
        lines.append("\n🚨 ANOMALIES DETECTED:")
        for anomaly in anomalies:
            lines.append(f"   {anomaly}")
        lines.append("")

    # Prompt (原始用户消息)
    lines.append("📝 PROMPT (Raw User Message):")
    lines.append(format_separator("-", 40))
    prompt_text = truncate_text(sample_data.get("prompt", "N/A"), max_length=800)
    lines.append(prompt_text)
    lines.append("")

    # Tokenized Prompt (完整的tokenized prompt，包含system prompt和tools)
    tokenized_prompt = sample_data.get("tokenized_prompt", "")
    if tokenized_prompt:
        lines.append("🔧 TOKENIZED PROMPT (Full, with System Prompt & Tools):")
        lines.append(format_separator("-", 40))
        tokenized_prompt_text = truncate_text(tokenized_prompt, max_length=2000)
        lines.append(tokenized_prompt_text)
        lines.append("")

    # Model Reasoning & Response
    lines.append("🤖 MODEL RESPONSE:")
    lines.append(format_separator("-", 40))
    response_text = truncate_text(sample_data.get("response_text", "N/A"), max_length=2000)
    lines.append(response_text)
    lines.append("")

    # Code Blocks
    code_blocks = sample_data.get("code_blocks", [])
    if code_blocks:
        lines.append(f"💻 CODE BLOCKS ({len(code_blocks)} total):")
        lines.append(format_separator("-", 40))
        for i, code in enumerate(code_blocks[:3]):  # 最多显示3个代码块
            lines.append(f"--- Code Block {i + 1} ---")
            code_truncated = truncate_text(code.strip(), max_length=500)
            lines.append(f"```python\n{code_truncated}\n```")
        if len(code_blocks) > 3:
            lines.append(f"... and {len(code_blocks) - 3} more code blocks")
        lines.append("")

    # Execution Results
    execution_results = sample_data.get("execution_results", [])
    if execution_results:
        lines.append(f"⚙️ EXECUTION RESULTS ({len(execution_results)} total):")
        lines.append(format_separator("-", 40))
        for i, result in enumerate(execution_results[:3]):  # 最多显示3个结果
            status = "✅ SUCCESS" if result.get("success") else "❌ ERROR"
            output = truncate_text(result.get("output", "N/A"), max_length=300)
            lines.append(f"[Turn {i + 1}] {status}")
            lines.append(f"Output: {output}")
        if len(execution_results) > 3:
            lines.append(f"... and {len(execution_results) - 3} more execution results")
        lines.append("")

    # Final Answer
    lines.append("📌 FINAL ANSWER:")
    lines.append(format_separator("-", 40))
    final_answer = sample_data.get("final_answer")
    if final_answer:
        lines.append(f"\\boxed{{{final_answer}}}")
    else:
        lines.append("❌ No \\boxed{} answer found")
    lines.append("")

    # Ground Truth
    lines.append("✅ GROUND TRUTH:")
    lines.append(format_separator("-", 40))
    lines.append(str(sample_data.get("ground_truth", "N/A")))
    lines.append("")

    # Reward & Metrics
    lines.append("📈 REWARD & METRICS:")
    lines.append(format_separator("-", 40))
    reward = sample_data.get("reward")
    acc = sample_data.get("acc", "N/A")
    num_turns = sample_data.get("num_turns", 0)
    is_correct = "✅ CORRECT" if acc == 1 else "❌ INCORRECT"

    lines.append(f"Reward Score: {reward if reward is not None else 'N/A'}")
    lines.append(f"Accuracy: {is_correct}")
    lines.append(f"Number of Turns: {num_turns}")
    lines.append(f"Code Blocks Used: {len(code_blocks)}")
    lines.append(f"Tool Calls Made: {len(execution_results)}")
    lines.append("")

    return "\n".join(lines)


# =====================================================================================
# 主监控函数
# =====================================================================================


def log_rollout_quality_samples(
    rollout_batches: List[Dict[str, List[Any]]],
    ppo_step: int,
    is_eval: bool = False,
    num_samples: int = None,
    log_path: str = None,
) -> None:
    """
    记录 rollout 样本质量日志
    
    Parameters
    ----------
    rollout_batches : List[Dict[str, List[Any]]]
        Rollout 批次数据
    ppo_step : int
        当前 PPO 步数
    is_eval : bool
        是否为评估模式
    num_samples : int
        采样数量，默认使用 MONITOR_SAMPLE_COUNT
    log_path : str
        日志文件路径，默认使用 MONITOR_LOG_PATH
    """
    if num_samples is None:
        num_samples = MONITOR_SAMPLE_COUNT
    if log_path is None:
        log_path = MONITOR_LOG_PATH

    # 只在 rank 0 上执行
    my_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if my_rank != 0:
        return

    try:
        tokenizer = get_tokenizer()
    except:
        tokenizer = None

    # 收集所有样本
    all_samples = []
    for batch in rollout_batches:
        tokens_list = batch.get("tokens", [])
        seq_lengths = batch.get("sequence_lengths", [])
        prompt_lengths = batch.get("prompt_lengths", [])
        gt_labels = batch.get("gt_label", [])
        messages_traj = batch.get("messages_traj", [])

        batch_size = len(tokens_list)
        for i in range(batch_size):
            all_samples.append(
                {
                    "tokens":
                        tokens_list[i] if i < len(tokens_list) else None,
                    "sequence_length":
                        seq_lengths[i].item()
                        if i < len(seq_lengths) and hasattr(seq_lengths[i], 'item') else
                        (seq_lengths[i] if i < len(seq_lengths) else 0),
                    "prompt_length":
                        prompt_lengths[i].item()
                        if i < len(prompt_lengths) and hasattr(prompt_lengths[i], 'item') else
                        (prompt_lengths[i] if i < len(prompt_lengths) else 0),
                    "gt_label":
                        gt_labels[i] if i < len(gt_labels) else {},
                    "messages":
                        messages_traj[i] if i < len(messages_traj) else None,
                }
            )

    if not all_samples:
        return

    # 采样：均匀分布选取样本
    sample_indices = []
    if len(all_samples) <= num_samples:
        sample_indices = list(range(len(all_samples)))
    else:
        step = len(all_samples) / num_samples
        for i in range(num_samples):
            idx = int(i * step)
            sample_indices.append(min(idx, len(all_samples) - 1))

    # 生成日志内容
    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "EVAL" if is_eval else "TRAIN"

    # 日志头部
    log_lines.append("")
    log_lines.append(format_separator("=", 80))
    log_lines.append(f"🔍 RETOOL QUALITY MONITOR - PPO Step {ppo_step} ({mode})")
    log_lines.append(f"📅 Timestamp: {timestamp}")
    log_lines.append(f"📊 Total Samples: {len(all_samples)} | Monitoring: {len(sample_indices)}")
    log_lines.append(format_separator("=", 80))
    log_lines.append("")

    # 统计信息
    stats = {
        "total_anomalies": 0,
        "empty_responses": 0,
        "truncated": 0,
        "no_boxed": 0,
        "code_errors": 0,
        "correct": 0,
        "incorrect": 0,
        "total_reward": 0.0,
    }

    # 处理每个样本
    for sample_idx, orig_idx in enumerate(sample_indices):
        sample = all_samples[orig_idx]

        # 提取样本数据
        if sample.get("messages"):
            sample_data = extract_sample_data_from_messages(
                sample["messages"],
                sample["gt_label"],
                tokens=sample.get("tokens"),
                prompt_length=sample.get("prompt_length", 0),
                tokenizer=tokenizer,
            )
        elif sample.get("tokens") is not None and tokenizer is not None:
            sample_data = extract_sample_data_from_tokens(
                sample["tokens"],
                sample["sequence_length"],
                sample["prompt_length"],
                sample["gt_label"],
                tokenizer,
            )
        else:
            continue

        # 检测异常
        anomalies = detect_anomalies(sample_data)

        # 更新统计
        stats["total_anomalies"] += len(anomalies)
        if any("EMPTY_RESPONSE" in a for a in anomalies):
            stats["empty_responses"] += 1
        if any("TRUNCATED" in a for a in anomalies):
            stats["truncated"] += 1
        if any("NO_BOXED" in a for a in anomalies):
            stats["no_boxed"] += 1
        if any("CODE_EXECUTION_ERROR" in a or "SYNTAX_ERROR" in a for a in anomalies):
            stats["code_errors"] += 1

        if sample_data.get("acc") == 1:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

        reward = sample_data.get("reward")
        if reward is not None:
            stats["total_reward"] += reward

        # 格式化样本日志
        sample_log = format_sample_log(sample_idx, ppo_step, sample_data, anomalies)
        log_lines.append(sample_log)

    # 统计摘要
    log_lines.append(format_separator("=", 80))
    log_lines.append("📈 STEP SUMMARY")
    log_lines.append(format_separator("=", 80))
    log_lines.append(f"✅ Correct: {stats['correct']} / {len(sample_indices)}")
    log_lines.append(f"❌ Incorrect: {stats['incorrect']} / {len(sample_indices)}")
    if len(sample_indices) > 0:
        avg_reward = stats["total_reward"] / len(sample_indices)
        acc_rate = stats["correct"] / len(sample_indices) * 100
        log_lines.append(f"📊 Accuracy Rate: {acc_rate:.1f}%")
        log_lines.append(f"📊 Average Reward: {avg_reward:.4f}")
    log_lines.append(f"⚠️ Total Anomalies: {stats['total_anomalies']}")
    log_lines.append(f"   - Empty Responses: {stats['empty_responses']}")
    log_lines.append(f"   - Truncated: {stats['truncated']}")
    log_lines.append(f"   - No Boxed Answer: {stats['no_boxed']}")
    log_lines.append(f"   - Code Errors: {stats['code_errors']}")
    log_lines.append(format_separator("=", 80))
    log_lines.append("")

    # 写入日志文件
    log_content = "\n".join(log_lines)

    # 确保日志目录存在
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 追加写入
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_content)

    # 可选：同时打印到 stdout
    if MONITOR_PRINT_TO_STDOUT:
        print(log_content)

    # 简短的控制台提示
    print(
        f"[ReTool Monitor] PPO Step {ppo_step}: "
        f"Acc={stats['correct']}/{len(sample_indices)} "
        f"Anomalies={stats['total_anomalies']} "
        f"-> {log_path}"
    )
