import json
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

# 精简版 prompts - 优化推理速度
CHARACTER_OCR_SYSTEM_PROMPT = (
    "识别图片中的所有文字内容。\n"
    "要求：\n"
    "- 按从上到下、从左到右的顺序识别\n"
    "- 每处文字单独一行\n"
    "- 重复出现的文字也要分别列出\n"
    "- 如果没有文字，输出'无'\n"
    "直接输出文字内容，不要添加任何解释或格式标记。"
)


def build_character_ocr_message(image):
    user_text = CHARACTER_OCR_SYSTEM_PROMPT
    messages = [
        {
            "role":
                "user",
            "content":
                [
                    {
                        "type": "text",
                        "text": user_text
                    },
                    {
                        "type": "image",
                        "image_data": image
                    },
                ],
        },
    ]
    return messages


def preprocess_text_for_matching(text: str) -> str:
    """
    预处理文本用于匹配：去除空格、换行、标点等非字母数字字符

    Args:
        text: 原始文本

    Returns:
        处理后的文本（只保留中英文、数字、法语字符）
    """
    # 只保留中英文、数字、法语字符
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", '', text)
    return cleaned.strip()


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的Levenshtein编辑距离

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        编辑距离（整数）
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # 插入、删除、替换的代价
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_character_score(ocr_text: str, text_gt: str) -> Dict[str, Any]:
    """
    计算字符匹配分数（Python端计算，不调用VLM）

    方法：
    1. 预处理：去除空格、换行、标点，只保留中英文数字
    2. 计算编辑距离
    3. 计算分数：score = 1 - edit_distance / max(len(gt), len(ocr))
    4. 额外计算Counter匹配度（考虑重复字符）

    Args:
        ocr_text: OCR识别的文本
        text_gt: Ground truth文本

    Returns:
        {
            "ocr_text_cleaned": 清理后的OCR文本,
            "gt_text_cleaned": 清理后的GT文本,
            "edit_distance": 编辑距离,
            "score": 基于编辑距离的分数 (0-1),
            "counter_score": 基于Counter的匹配分数 (0-1),
            "final_score": 最终分数（取两者平均）,
            "reason": 评分说明
        }
    """
    # 预处理
    ocr_cleaned = preprocess_text_for_matching(ocr_text)
    gt_cleaned = preprocess_text_for_matching(text_gt)

    # 如果GT为空
    if not gt_cleaned:
        return {
            "ocr_text_cleaned": ocr_cleaned,
            "gt_text_cleaned": gt_cleaned,
            "edit_distance": len(ocr_cleaned),
            "score": 1.0 if not ocr_cleaned else 0.0,
            "counter_score": 1.0 if not ocr_cleaned else 0.0,
            "final_score": 1.0 if not ocr_cleaned else 0.0,
            "reason": "GT为空" if not ocr_cleaned else "GT为空但OCR识别出了文字"
        }

    # 计算编辑距离
    edit_distance = calculate_levenshtein_distance(ocr_cleaned, gt_cleaned)
    max_len = max(len(gt_cleaned), len(ocr_cleaned))

    # 基于编辑距离的分数
    edit_score = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
    edit_score = max(0.0, min(1.0, edit_score))  # 限制在[0, 1]

    # 基于Counter的匹配分数（考虑重复字符）
    gt_counter = Counter(gt_cleaned)
    ocr_counter = Counter(ocr_cleaned)

    # 计算交集（匹配的字符，考虑重复）
    match_counter = gt_counter & ocr_counter
    match_count = sum(match_counter.values())
    gt_total = sum(gt_counter.values())

    counter_score = match_count / gt_total if gt_total > 0 else 0.0

    # 最终分数：取两者平均（或者可以选择最大值、最小值）
    final_score = (edit_score + counter_score) / 2.0

    # 生成评分说明
    reason = (
        f"GT文本: '{gt_cleaned}' (长度{len(gt_cleaned)}), "
        f"OCR文本: '{ocr_cleaned}' (长度{len(ocr_cleaned)}), "
        f"编辑距离: {edit_distance}, "
        f"编辑距离分数: {edit_score:.4f}, "
        f"Counter匹配: {match_count}/{gt_total}, "
        f"Counter分数: {counter_score:.4f}, "
        f"最终分数: {final_score:.4f}"
    )

    return {
        "ocr_text_cleaned": ocr_cleaned,
        "gt_text_cleaned": gt_cleaned,
        "edit_distance": edit_distance,
        "score": edit_score,
        "counter_score": counter_score,
        "final_score": final_score,
        "match_count": match_count,
        "gt_total": gt_total,
        "reason": reason
    }


def extract_text_gt_from_prompt(prompt: str) -> str:
    """
    从prompt中提取文字GT（基于引号规则）
    参考ocr_reward.py的实现

    Args:
        prompt: 文生图提示

    Returns:
        提取出的文字内容，用空格连接
    """
    if not prompt:
        return ""

    extracted_texts = []

    # 所有双引号配对模式
    double_quote_patterns = [
        (r'\u201c', r'\u201d'),  # ""
        (r'"', r'"'),  # ""
        (r'\uff02', r'\uff02'),  # ＂＂
        (r'\u201d', r'\u201d'),  # ""
        (r'\u201e', r'\u201d'),  # „"
        (r'\u201f', r'\u201d'),  # ‟"
        (r'\u00ab', r'\u00bb'),  # «»
        (r'\u00bb', r'\u00ab'),  # »«
        (r'\u300c', r'\u300d'),  # 「」
        (r'\u300e', r'\u300f'),  # 『』
        (r'\u301d', r'\u301e'),  # 〝〞
        (r'\ufe41', r'\ufe42'),  # ﹁﹂
        (r'\ufe43', r'\ufe44'),  # ﹃﹄
        (r'\u275d', r'\u275e'),  # ❝❞
        (r'\u276e', r'\u276f'),  # ❮❯
    ]

    matched_ranges = []

    for left_quote, right_quote in double_quote_patterns:
        pattern_str = f'{left_quote}([^{right_quote}]*?){right_quote}'
        pattern = re.compile(pattern_str)

        for match in pattern.finditer(prompt):
            start, end = match.span()
            overlap = False
            for matched_start, matched_end in matched_ranges:
                if not (end <= matched_start or start >= matched_end):
                    overlap = True
                    break

            if not overlap:
                extracted_texts.append(match.group(1))
                matched_ranges.append((start, end))

    # 过滤空字符串并去重
    extracted_texts = [t.strip() for t in extracted_texts if t.strip()]
    seen = set()
    unique_texts = []
    for t in extracted_texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    return ' '.join(unique_texts)


def compute_character_score(ocr_text: str, prompt: str) -> dict:
    """
    在Python端计算character score（不调用VLM）

    Args:
        ocr_text: OCR识别的文本
        prompt: 文生图提示

    Returns:
        包含score和详细信息的字典
    """
    # 从prompt中提取GT文本
    text_gt = extract_text_gt_from_prompt(prompt)

    # 计算分数
    result = calculate_character_score(ocr_text, text_gt)

    return {
        "score": result["final_score"],  # 使用final_score作为主分数
        "edit_distance_score": result["score"],
        "counter_score": result["counter_score"],
        "edit_distance": result["edit_distance"],
        "ocr_text": result["ocr_text_cleaned"],
        "gt_text": result["gt_text_cleaned"],
        "match_count": result.get("match_count", 0),
        "gt_total": result.get("gt_total", 0),
        "reason": result["reason"]
    }


DSG_ENTITY_SYSTEM_PROMPT = (
    "从文生图提示中提取所有提到的文字内容，并生成验证问题。\n\n"
    "提取规则：提示中提到的任何文字都要提取（标题、招牌、路牌等），但如果文字内容没有排版，字体，大小，位置要求，则不提取。\n\n"
    "为每个文字生成4个问题，如果文字内容没有排版，字体，大小，位置要求，则不提取：\n"
    "1. 排版：文字\"XXX\"的排版是否[描述]？\n"
    "       - 排版要求描述需聚焦整体布局，例如：对齐方式、行间距、列排布、是否等宽排列等\n"
    "2. 字体：文字\"XXX\"的字体是否[描述]？\n"
    "       - 必须包含：字体类型（楷体/宋体/黑体/草书等）、粗细（粗体/常规/细体）、风格（正体/斜体/蚀刻/书法等）\n"
    "3. 大小：文字\"XXX\"的大小是否[描述]？\n"
    "       - 必须包含可验证的标准（如'占画面高度1/3''巨大''与图片尺寸比例合适'）\n\n"
    "4. 位置：文字\"XXX\"是否位于[描述]？\n\n"
    "       - 必须包含：具体位置（左上角/中央/右下角等）\n"
    "输出JSON格式：\n"
    "{\n"
    '  "questions": [\n'
    '    {"question_id": "Q1", "question": "问题内容"}\n'
    "  ]\n"
    "}\n"
    "不要添加代码块标记，直接输出JSON。"
    "若文生图提示未对在排版，字体，大小，位置四个维度提出要求，则不生成问题。"
)

DSG_SCORE_SYSTEM_PROMPT = (
    "你是严格的视觉评估专家。回答每个问题是yes还是no。\n\n"
    "评估标准：\n"
    "- 完全符合才答yes，任何不确定、模糊、部分符合都答no\n"
    "- 文字模糊、无法准确判断时答no\n\n"
    "输出JSON格式：\n"
    "{\n"
    '  "answers": [\n'
    '    {"question_id": "Q1", "answer": "yes或no"}\n'
    "  ],\n"
    '  "score": yes数量/总数\n'
    "}\n"
    "不要添加代码块标记，直接输出JSON。"
)


def build_dsg_entity_message(prompt: str) -> dict:
    """第三次VLM调用：DSG entity extraction (无需图像) - 简化版"""
    user_text = f"{DSG_ENTITY_SYSTEM_PROMPT}\n\n文生图提示：{prompt}"
    messages = [
        {
            "role": "user",
            "content": user_text
        },
    ]
    return messages


def safe_json_loads(text: str, default=None):
    """安全的JSON解析"""
    if text is None:
        return default
    text = text.strip()

    # 移除代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取JSON片段
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return default


def build_dsg_entity_score_message(image, prompt, questions):
    user_text = f"{DSG_SCORE_SYSTEM_PROMPT}\n\n原始提示：{prompt}\n\n验证问题：\n{json.dumps(questions, ensure_ascii=False, indent=2)}"
    messages = [
        {
            "role":
                "user",
            "content":
                [
                    {
                        "type": "text",
                        "text": user_text
                    },
                    {
                        "type": "image",
                        "image_data": image
                    },
                ],
        },
    ]
    return messages


# 简洁的 OCR prompt
OCR_SYSTEM_PROMPT = ("识别图片中的所有文字，只返回识别出的文字内容，不要添加任何解释。"
                     "如果有多处文字，用换行分隔。如果没有文字，返回空。")


def build_ocr_message(image):
    messages = [
        {
            "role":
                "user",
            "content":
                [
                    {
                        "type": "text",
                        "text": OCR_SYSTEM_PROMPT
                    },
                    {
                        "type": "image",
                        "image_data": image
                    },
                ],
        },
    ]
    return messages


def preprocess_string(s: str, mode: str = 'zh') -> str:
    """预处理字符串，移除特殊字符"""
    # 只保留中英文、数字、法语字符
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", '', s)
    if mode == 'en':
        normalized = re.sub(r'\s+', ' ', cleaned)
        return normalized.strip().lower()
    else:
        # 中文模式：去除所有空格
        pattern = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]")
        s = ''.join(pattern.findall(cleaned))
        return s.strip()


def calculate_match_score(text_gt: str, ocr_text: str, mode: str = 'zh') -> Tuple[float, int, int]:
    """
    计算 OCR 匹配分数

    Args:
        text_gt: Ground truth 文本
        ocr_text: OCR 识别结果
        mode: 'zh' 中文按字符匹配，'en' 英文按单词匹配

    Returns:
        (score, match_count, gt_count)
    """
    gt_processed = preprocess_string(text_gt, mode)
    ocr_processed = preprocess_string(ocr_text, mode)

    if not gt_processed:
        return 1.0 if not ocr_processed else 0.0, 0, 0

    if mode == 'en':
        gt_tokens = gt_processed.split()
        ocr_tokens = ocr_processed.split()
    else:
        gt_tokens = list(gt_processed)
        ocr_tokens = list(ocr_processed)

    gt_counter = Counter(gt_tokens)
    ocr_counter = Counter(ocr_tokens)

    # 计算交集（匹配的词/字符）
    match_counter = gt_counter & ocr_counter
    match_count = sum(match_counter.values())
    gt_count = len(gt_tokens)

    score = match_count / gt_count if gt_count > 0 else 0.0
    return score, match_count, gt_count
