# OPD 中 Teacher / Student 使用不同 Prompt 时的 Batch 格式

## 背景

`OnPolicyDistillRolloutGenerator` 支持一个 rollout batch 中同时携带 **student 侧**与
**teacher 侧**两份 prompt 的编码结果。典型场景：

- Student 与 teacher 使用不同的 tokenizer / chat template（如 qwen3.5 vs wemm3.5）；
- Student 与 teacher 使用不同的多模态处理管线，导致 `position_ids`、`vision_grid_thw`、
  `vision_data`、`image_input_mask`、`audio_feature` 等张量形状 / 数值不一致；
- Teacher 需要按自己的模板重新拼 prompt 才能正确产生教师分布。

如果直接把 student 的一份数据丢给 teacher，会因为 shape / token id 不匹配而出错。为此，
框架在 dataset → rollout_batch → teacher client 这条链路上约定了一组以 `teacher_` 为
前缀的字段，用于承载"临时构建的 teacher sample"。

对应的关键代码：

- `gcore-dev/gpatch_v4/rollout_generator/on_policy_distill_generator.py`
  - `calc_all_teacher_logps` — 组装 teacher 请求
  - `get_teacher_rollout_batch` — 从 rollout_batch 抽出 teacher 视图
  - `fillback_teacher_logps` — 把 teacher 返回的 logps 对齐回 student 长度轴
- `gcore-dev/tasks/multimodal_v4/on_policy_distill/qwen3vl_mix_dataset.py`
  - 示例 dataset：在 map function 里额外产出 teacher 侧字段

---

## Sample 顶层字段一览

DataLoader 产生的sample包含的字段.
下面表格中，`Student 独有`表示该字段永远指 student 侧编码；`Teacher 独有`表示只在
需要独立 teacher prompt 时才出现；`共享`表示 teacher 复用 student 的那份。

| 字段 | 类型 | 归属 | 说明 |
|------|------|------|------|
| unique_id: 参照例子即可
| json_data_list
| imgs_np_array_list: 如果有视觉,必须
| audios_np_array_list: 如果有音频,必须
| `tokens` | `List[Tensor]` | Student 独有 | Student 的 input_ids | 必须
| `prompt_len` | `List[Tensor]` | Student 独有 | Student prompt 长度 | 必须
| `position_ids` | `List[Tensor]` | Student 独有 | Student 的 rope index | 可选
| `vision_data` / `vision_grid_thw` / `image_input_mask` | 张量 | Student 独有 | Student 视觉侧输入 | 可选
| `input_features` / `feature_attention_mask` / `audio_feature` | 张量 | Student 独有 | Student 音频侧输入 | 可选
| `teacher_tokens` | `List[Tensor]` | **Teacher 独有** | Teacher 的 input_ids；**存在即触发 teacher 独立 sample 路径** | 
| `teacher_prompt_len` | `List[Tensor]` | Teacher 独有 | Teacher prompt 长度 |
| `teacher_res` | `Dict[str, Tensor]` | Teacher 独有 | Teacher 侧其它张量的容器（见下） |

`teacher_res` 是一个 dict，把 teacher 侧对应 student 那份的额外字段装进去，例如：

```python
teacher_res = {
    "position_ids":     ...,   # teacher 侧的 rope index
    "vision_grid_thw":  ...,   # teacher 侧的 image_grid_thw
    "image_input_mask": ...,   # teacher tokens == teacher_image_token_id
    "vision_data":      ...,   # teacher processor 产出的 pixel_values
    # 也可放 input_features / feature_attention_mask / audio_feature 等
}
```

**存在 `teacher_tokens` 即触发独立 teacher sample 路径**（判定见
`get_teacher_rollout_batch` 首行 `if "teacher_tokens" in rollout_batch`）。缺省情况
下（例如 self-distill）该字段不存在，teacher 直接复用 student 的一份数据。

---

## Dataset 侧如何构造这些字段

以 `qwen3vl_mix_dataset.py` 为参考，dataset map function 除了产出 student 一份外，
另外用 **teacher 的 processor/tokenizer** 把同一个对话再走一遍，把结果按上表命名塞
进 sample dict。示例（伪代码）：

```python
# 1) student side (existing pipeline)
stu = student_processor.apply_chat_template(msgs, ...)
stu_input_ids = pad_to(stu["input_ids"], seq_len)
stu_position_ids, _ = student_rope.get_rope_index(...)

# 2) teacher side —— 用 teacher 自己的 tokenizer/processor 再来一次
tea = teacher_processor.apply_chat_template(msgs, ...)
tea_input_ids = pad_to(tea["input_ids"], seq_len)
tea_position_ids, _ = teacher_rope.get_rope_index(...)

sample = {
    # student
    "tokens":            [stu_input_ids.squeeze(0)],
    "prompt_len":    [torch.tensor(stu_prompt_len)],
    "position_ids":      stu_position_ids,
    "vision_grid_thw":   stu["image_grid_thw"],
    "vision_data":       stu["pixel_values"],
    "image_input_mask":  stu_input_ids == student_hf_config.image_token_id,

    # teacher —— 关键三件套
    "teacher_tokens":            [tea_input_ids.squeeze(0)],
    "teacher_prompt_len":    [torch.tensor(tea_prompt_len)],
    # 其它 teacher 侧张量装进 teacher_res
    "teacher_res": {
        "position_ids":     tea_position_ids,
        "vision_grid_thw":  tea["image_grid_thw"],
        "vision_data":      tea["pixel_values"],
        "image_input_mask": tea_input_ids == teacher_hf_config.image_token_id,
    }
}
```

要点：

1. **必须一起提供** `teacher_tokens`、`teacher_prompt_len`
   缺任意一个会走到 fillback 阶段崩溃。
2. `teacher_res` 是"其它 teacher 张量的口袋"，键名与 student 侧一致。`get_teacher_rollout_batch`
   会把它铺平到 teacher_batch 里。
3. Response 长度约束：`teacher_sequence_lengths - teacher_prompt_lengths` 必须与
   `sequence_lengths - prompt_lengths` 相等（即 teacher 与 student 的**响应长度一致**，
   两者共享同一段采样生成的 response token）。断言在
   `get_teacher_rollout_batch` 中：
   ```python
   assert tea_response_len == stu_response_len
   ```

