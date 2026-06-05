# Train a language model on mathematical tasks using GRPO with generative RM


## Download model, dataset, then SFT

先下载 checkpoint 与数据，然后进行 SFT。参考 <a href="/math_sft.html">Math SFT</a>。

First, download the checkpoint and data, then perform SFT (Supervised Fine-Tuning). Refer to <a href="/math_sft.html">Math SFT</a> for details.

Copy the sft checkpoint as ref model.

```
cp -r qwen_2_5_1_5b_sft qwen_2_5_1_5b
cp -r qwen_2_5_1_5b_sft qwen_2_5_1_5b_ref
```

## GRPO

参考[脚本](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen-gen-rm/grpo.sh)。

Refer to the [script](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen-gen-rm/grpo.sh).

Megatron  的 trainer 只需要 torchrun 即可，非常简单。
我们实际使用的时候会是用内部的系统拉起作业。
对于外部用户，可以参考我们的[mpirun 的例子](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen-gen-rm/mpirun-grpo.sh)。

Megatron's trainer can be launched simply with torchrun, making it very straightforward.
In our actual workflow, we use our internal system to start the jobs.
For external users, you can refer to our [mpirun example](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen-gen-rm/mpirun-grpo.sh).


## Convert actor megatron checkpoint back to HF

```bash
bash tasks/math_rl_v3/qwen/convert_ckpt_actor.sh mlm_to_hf
```

You should be able to achieve

```
eval accuracy 0.80061
format matching degree 0.95148
```

250810, h20 x64, driver 570, cuda 12.8.

```
eval accuracy 0.80061
format matching degree 0.94996
```
