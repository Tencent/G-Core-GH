# Trainer

关于 gcore 的一些设计可以参考我们的 arxiv，后续其他 ai 算法部门同事相关论文也会放出，敬请期待。
本文按照 math 的例子代码使用参考。

Some design aspects of gcore can be referenced in our arXiv paper. In the future, related papers from colleagues in other AI algorithm departments will also be released—please stay tuned.
This document uses the math example code as a reference.

## SFT

SFT 的 trainer 就是 Megatron LM 的 trainer，通过 patch 稍微改动。Megatron LM 的 trainer
谈不上完美，但也能用，暂时没有必要完全重做。
例子请参考 <a href='/math_sft.html'>Math SFT</a>。

The SFT trainer is essentially the Megatron LM trainer, with minor modifications applied via patches. While the Megatron LM trainer is not perfect, it is functional, and there is currently no need to completely rewrite it.
For examples, please refer to <a href='/math_sft.html'>Math SFT</a>.

## GRPO

例子请参考 <a href='/math_grpo.html'>Math GRPO</a>。

gcore launcher 和 megatron 一样用 torchrun。多个节点拉起可以手动拉起，也可以通过一些内部系统启动，或者用 mpirun 启动也可以。

For examples, please refer to <a href='/math_grpo.html'>Math GRPO</a>.

The gcore launcher, like Megatron, uses torchrun. For multi-node setups, you can launch nodes manually, use some internal systems to start them, or use mpirun as well.

### Auto Config

在 wxg 内部 gemini，我们会先通过 `tools/auto_place.py` 创建一个初始的配置文件，由于这些配置比较繁琐，所以我们没在
trainer 内处理，而是通过一个程序输出一个配置文件，这个配置文件包含了 gpu
集群的信息以及初始的摆放方案。需要输入 actor、critic、sampler 的并行配置。

注意，这里假定 mpirun 在每个节点的 slot 是 1。如果在 gemini 批量执行或者命令入口也可以。

默认的 interface name 数据端口的虚拟设备是 bond1（腾讯的设备），在外部一般是 eth0 或者 eth1。

In the internal Gemini system at wxg, we first use `tools/auto_place.py` to create an initial configuration file. Since these configurations are quite complex, we do not handle them inside the trainer. Instead, we use this program to generate a configuration file that contains information about the GPU cluster and the initial placement scheme. You need to provide the parallel configuration for the actor, critic, and sampler.

Note: Here, we assume that the slot for mpirun on each node is 1. This also applies if you use batch execution or command entry in Gemini.

By default, the interface name for the data port is bond1 (Tencent’s device). In external environments, it is usually eth0 or eth1.

```bash
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile /root/hostfile \
      python tools/auto_place.py \
      --fn gen --config-folder $PLACE_CFG_FOLDER \
      --interface-name bond1 \
      --sampler-tp-size 2 --sampler-pp-size 1 \
      --critic-tp-size 2 --critic-pp-size 2 \
      --actor-tp-size 2 --actor-pp-size 2 --actor-cp-size 2 \
```

默认情况会在 `place-config/config.json` 生成一个配置：

By default, a configuration file will be generated at `place-config/config.json`.

```
{
            ...
            },
            {
                "dp_rank": 3,
                "ip": "28.6.11.6",
                "port": 61503
            }
        ]
    },
    "ray": {
        "port": 6379
    }
}
```

### Sampler

gcore 的 sampler 是独立的进程，与 actor 分离，这样代码比较好处理，也可以视为普通的推理服务，而不需要感知 training backend 的拓扑。

The sampler in gcore is an independent process, separated from the actor. This makes the code easier to manage, and the sampler can be treated as a regular inference service without needing to be aware of the topology of the training backend.

```bash
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/sampler.hostfile \
      bash tasks/math_rl_v3/qwen/grpo.sh $PLACE_CFG_FOLDER sampler >$LOG_DIR/sampler$PART.log 2>&1 &
```

参考[代码](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/train_ppo_sampler.py)

Refer to the [code](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/train_ppo_sampler.py) for more details.

大部分的时候，model provider 用 `default_sampler_model_provider` 默认即可，trainer 也是用 `GrpoSamplerV3` 默认即可。
你只需要根据你的采样规则修改 generation function。这里的代码很简单很开放，
所以也可以 call tool 或者是外部的 web api。

Most of the time, you can just use the default `default_sampler_model_provider` for the model provider, and the default `GrpoSamplerV3` for the trainer.
You only need to modify the generation function according to your sampling rules.
The code here is very simple and open, so you can also call tools or external web APIs.

```python
    grpo_sampler = GrpoSamplerV3()
    run_grpo_sampler_v3(
        grpo_sampler,
        model_provider,
        gen_rollouts,
        extra_args_provider=get_tasks_args,
    )
```

```python
@torch.no_grad()
async def gen_rollouts(engine, batch):
    # batch 就是 actor 的 rollout_get_batch 的返回值，prompt_token_ids 是 list of dict，lpad_lens 和
    # gt_label 是 [b,] 的 tensor。
    args = get_args()
    prompt_token_ids, lpad_lens, gt_label = batch["prompt_token_ids"], batch["lpad_lens"], batch["gt_label"]

    sampling_params = get_sampling_params(engine)
    gens = []
    for i in range(len(prompt_token_ids)):
        for j in range(args.ppo_sampling_repeat):
            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += i * args.ppo_sampling_repeat + j
            gen = engine.async_generate(prompt_token_ids[i], tmp_sampling_params, str(uuid.uuid4().hex))
            gens.append(gen)
    
    # ...
```

### RM

gcore 的 rm 是独立的进程，与 actor 分离，一方面避免了一些全局状态（例如 MPU）互相干扰，一方面也方便做 Gen-RM。

The RM (Reward Model) in gcore runs as an independent process, separate from the actor. On one hand, this avoids interference between global states (such as MPU), and on the other hand, it also makes it easier to implement Gen-RM.

参考[代码](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/train_ppo_critic.py)

Refer to the [code](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/train_ppo_critic.py) for more details.

```bash
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/critic.hostfile \
      bash tasks/math_rl_v3/qwen/grpo.sh $PLACE_CFG_FOLDER critic >$LOG_DIR/critic$PART.log 2>&1 &
```

大部分的时候，model provider 用 `default_reward_model_provider` 默认即可，trainer 也是用 `GrpoRmTrainerV3` 默认即可。

Most of the time, you can just use the default `default_reward_model_provider` for the model provider, and the default `GrpoRmTrainerV3` for the trainer.

```python
    trainer = GrpoRmTrainerV3()
    run_grpo_rm_v3(
        trainer,
        model_provider,
        critic_provider,
        ModelType.encoder_or_decoder,
        extra_args_provider=get_tasks_args,
    )
```

### Actor

参考[代码](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/train_ppo_actor.py)

Refer to the [code](https://github.com/Tencent/Wechat-YATT/blob/public/tasks/math_rl_v3/train_ppo_actor.py) for more details.

```bash
mpirun -v --allow-run-as-root \
      --bind-to none --map-by slot --hostfile $PLACE_CFG_FOLDER/actor.hostfile \
      bash tasks/math_rl_v3/qwen/grpo.sh $PLACE_CFG_FOLDER actor >$LOG_DIR/actor$PART.log 2>&1 &
```

```python
    trainer = MathRLActorTrainer(extra_metric_info=extra_metric_info_provider)
    train_ppo_actor_v3(trainer,
                       model_provider,
                       actor_provider,
                       sampler_client_provider,
                       rm_critic_client_provider,
                       gen_rm_client_provider,
                       train_valid_test_datasets_provider,
                       rollout_get_batch,
                       filter_samplings,
                       ModelType.encoder_or_decoder,
                       extra_args_provider=get_tasks_args)
```

大部分都是默认即可，只有 `rollout_get_batch` 获取数据需要根据实际输出处理。
在这里例子中，trainer 是继承的，因为有一些特殊的重放规则，实际上大部分时间并不必须。

Most of the settings can be left as default; only the `rollout_get_batch` function, which retrieves data, needs to be handled according to the actual output.
In this example, the trainer is subclassed (inherited) because there are some special replay rules. In practice, this is not required most of the time.

```python
def rollout_get_batch(data_iterator):
    args = get_args()
    assert data_iterator is not None
    data = next(data_iterator)

    tokens = data['input_ids']
    lpad_lens = data['lpad_lens']
    gt_label = data['gt_label']

    lpad_lens_list = lpad_lens.tolist()
    prompt_token_ids = []
    for i in range(len(tokens)):
        prompt_token_ids.append({
            'prompt_token_ids': tokens[i][:lpad_lens_list[i]].tolist(),
        })

    batch_data = {
        "prompt_token_ids": prompt_token_ids,
        "lpad_lens": lpad_lens,
        "gt_label": gt_label,
    }
    return batch_data
```
