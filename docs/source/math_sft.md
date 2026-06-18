# Train a language model on mathematical tasks using SFT

Megatron-LM 的 Trainer 对我们来说足够用了，我们没有大幅度修改他的动机。Pretrain / SFT 我们尽可能重用 Nvidia 的代码，尽量减少 upgrade Megatron 时候适配的重复工作。

The Trainer provided by Megatron-LM is sufficient for our needs; we don’t have a strong motivation to make major modifications to it. For both pretraining and SFT (Supervised Fine-Tuning), we try to reuse Nvidia’s code as much as possible, minimizing redundant adaptation work when upgrading Megatron.


## download models and dataset

```bash
MYWD=$PWD

repo_id=AI-MO/NuminaMath-CoT
huggingface-cli download --repo-type dataset $repo_id --local-dir $MYWD/hf-hub/$repo_id

repo_id=openai/gsm8k
huggingface-cli download --repo-type dataset $repo_id --local-dir $MYWD/hf-hub/$repo_id

repo_id=Qwen/Qwen2.5-Math-1.5B
huggingface-cli download --repo-type model $repo_id --local-dir $MYWD/hf-hub/$repo_id

repo_id=Qwen/Qwen2.5-Math-RM-72B
huggingface-cli download --repo-type model $repo_id --local-dir $MYWD/hf-hub/$repo_id

repo_id=Qwen/Qwen2.5-Math-72B-Instruct
huggingface-cli download --repo-type model $repo_id --local-dir $MYWD/hf-hub/$repo_id
```

默认情况，会被下载到 **当前目录** 的 hf-hub 目录里。

By default, they will be downloaded to `$PWD/hf-hub`.

内部参考：`tasks/math_rl_v3/qwen/download-data-and-checkpoint-priv.sh` 这里配置代理。

## preprocess dataset

预处理数据

preprocess dataset

```bash
bash tasks/math_rl_v3/qwen/preprocess_data.sh
```

## convert checkpoint

任何能转换的脚本都是可以的，从 huggingface 到 megatron 格式的代码社区有很多实现，我们也提供了[一份](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/convert_ckpt_actor.sh)。

Any script that can perform the conversion is acceptable—there are many community implementations for converting from HuggingFace to Megatron format. We also provide [one such script](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/convert_ckpt_actor.sh).

```bash
bash tasks/math_rl_v3/qwen/convert_ckpt_actor.sh hf_to_mlm
```

## bash

我们参考 *Qwen2.5-Math Technical Report*，对 base model 进行了微调, 参考[脚本](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/sft.sh)。

We followed the *Qwen2.5-Math Technical Report* to fine-tune the base model, referring to this [script](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/sft.sh) as a reference.

和 megatron lm trainer 没太大区别，但我们自己做了一层模型的封装，方便对比切换模型，同理你可以切到起他尺寸或者起他 llama3 之类的模型。

It’s not much different from the Megatron-LM trainer, but we’ve added our own layer of model encapsulation to make it easier to switch and compare models. Similarly, you can switch to other model sizes or to models like Llama 3 as needed.

```bash
--cli-arg-yaml-cfgs gpatch/model_yamls/qwen2.5-math-1.5b.yaml \
```

对应的代码在[这里](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/sft.py)，具体根据细节请看 <a href="/trainer.html">trainer</a> 部分的文档。

The corresponding code can be found [here](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/sft.py). For specific details, please refer to the documentation in the <a href="/trainer.html">trainer</a> section.

## launcher

Megatron  的 trainer 只需要 torchrun 即可，非常简单。
我们实际使用的时候会是用内部的系统拉起作业。
对于外部用户，可以参考我们的[mpirun 的例子](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/mpirun-sft.sh)。

Megatron's trainer can be launched simply with torchrun, making it very straightforward.
In our actual workflow, we use our internal system to start the jobs.
For external users, you can refer to our [mpirun example](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/qwen/mpirun-sft.sh).

```bash
export _MASTER_ADDR=$YOUR_IP

mpirun -v --allow-run-as-root \
  --bind-to none --map-by slot --hostfile $hostfile \
  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH -x _MASTER_ADDR \
  bash tasks/math_rl_v3/qwen/sft.sh
```

mpi 虽然比较复古，但对于拉起作业是够用的。
其他 pdsh / slurm 之类能 launch 的方式应该也可以的。
如果你节点不多，甚至懒得去搞 launcher，手动在多个节点执行 torchrun，也是可以的。

Although MPI is somewhat old-fashioned, it is sufficient for launching jobs.
Other methods capable of launching jobs, such as pdsh or slurm, should also work.
If you don't have many nodes or don't want to bother with a launcher, you can even manually run torchrun on multiple nodes.

## Eval

You should be able to achieve

```
eval accuracy 0.58150
format matching degree 0.69901
```

250810, h20 x64, driver 570, cuda 12.8.

```
eval accuracy 0.57998
format matching degree 0.69826
```

mbridge lora sft(lora 好像对初始化比较敏感):
```
eval accuracy 0.56
format matching degree 0.70
```

mbridge canonical_lora sft（lora 好像对初始化比较敏感）:
```
eval accuracy 0.55
format matching degree 0.684
```