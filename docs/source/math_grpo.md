# Train a language model on mathematical tasks using GRPO

实验步骤大体如下：

1. 数据预处理，将下载的开源数据转为 gcore 适应的格式；
2. 将 Huggingface 格式的 ckpt 转为 mlm 格式的 ckpt；
3. 对 base model 进行 finetune 训练 [非必须]；
4. 进行 GRPO 训练；
5. 将 mlm 格式的 ckpt 转回 Huggingface 格式的 ckpt。

The general steps of the experiment are as follows:

1. Data preprocessing: Convert the downloaded open-source data into a format compatible with YATT.
2. Convert the Huggingface-format checkpoint (ckpt) to the mlm-format checkpoint.
3. Finetune the base model [optional].
4. Perform GRPO training.
5. Convert the mlm-format checkpoint back to the Huggingface-format checkpoint.


## Download model, dataset, then SFT

先下载 checkpoint 与数据，然后进行 SFT。参考 <a href="/math_sft.html">Math SFT</a>。

First, download the checkpoint and data, then perform SFT (Supervised Fine-Tuning). Refer to <a href="/math_sft.html">Math SFT</a> for details.

Copy the sft checkpoint as ref model.

```
cp -r qwen_2_5_1_5b_sft qwen_2_5_1_5b
cp -r qwen_2_5_1_5b_sft qwen_2_5_1_5b_ref
```

## Convert the RM checkpoint

convert RM checkpoint

```
bash tasks/math_rl_v3/qwen/convert_ckpt_rm.sh
```

## GRPO

参考[脚本](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/grpo.sh)。

Refer to the [script](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/grpo.sh).

Megatron  的 trainer 只需要 torchrun 即可，非常简单。
我们实际使用的时候会是用内部的系统拉起作业。
对于外部用户，可以参考我们的[mpirun 的例子](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/mpirun-grpo.sh)。

Megatron's trainer can be launched simply with torchrun, making it very straightforward.
In our actual workflow, we use our internal system to start the jobs.
For external users, you can refer to our [mpirun example](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/mpirun-grpo.sh).


## Convert actor megatron checkpoint back to HF and Eval

```bash
bash tasks/math_rl_v3/qwen/convert_ckpt_actor.sh mlm_to_hf
bash tasks/math_rl_v3/qwen/eval_model.sh
```

You should be able to achieve

```
eval accuracy 0.86126
format matching degree 0.98863
```

---

250810, h20 x64, driver 570, cuda 12.8.

```
eval accuracy 0.85519
format matching degree 0.99318
```

---

250818, h20 x32, driver 570, cuda 12.8

w/odyn mbs

```
eval accuracy 0.86353
format matching degree 0.99318
```

dyn mbs

```
eval accuracy 0.85368
format matching degree 0.98939
```

---

partial rollout 1016

```
eval accuracy 0.85898
format matching degree 0.98863 
```

partial rollout 1020, `partial_rollout_gbs = 2*rollout_gbs`

```
eval accuracy 0.83548
format matching degree 0.97650 
```

---

important sampling

date: 2025/11/13

MR: `https://git.woa.com/wepsdl/gcore-dev/-/merge_requests/634`

- token level truncate important sampling
    - option: `--enable-off-policy-correction`
    ```
    eval accuracy 0.85974
    format matching degree 0.98711
    ```

- sequence level mask important sampling
    - option: `--enable-off-policy-correction` `--off-policy-correction-level sequence` `--off-policy-correction-mode mask` `--off-policy-correction-veto-threshold 1.0e-4`
    ```
    eval accuracy 0.86126
    format matching degree 0.99090
    ```

---

升级 docker 2025/12/16

jeff 记录

driver 575 + fused attn eval acc 0.855, fmt 0.985.
driver 575 + fa3 eval acc 0.848, fmt 0.982.
driver 535 + fa3 eval acc 0.855, fmt 0.986.
