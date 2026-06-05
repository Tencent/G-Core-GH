# GCORE ARGS 说明
关于gcore常用的参数按照分类进行简单介绍。
## 1、DISTRIBUTED_ARGS
DISTRIBUTED_ARGS主要用于torchrun建立网络连接所用，假设共有N个节点，每个节点有8张卡，则共有8*N张卡。
```
--nproc_per_node         # 每个节点有多少张GPU卡，通常是8张
--nnodes                 # 共有多少个节点，假设为N
--node_rank              # 当前节点在所有节点中的rank，取值为[0~N-1]
--master_addr            # master节点的IP
--master_port            # master节点的端口号
```

## 2、MP_ARGS
MP_ARGS是大模型分布式并行策略相关的参数。
```
--tensor-model-parallel-size         # TP并行的规模
--pipeline-model-parallel-size       # PP并行的规模
--sequence-parallel                  # 采用SP并行优化
--context-parallel-size              # CP并行的规模
--use-distributed-optimizer          # 开启分布式优化器
--use-tp-pp-dp-mapping               # 如果设置了这个参数，则会按照tp-cp-ep-pp-dp的顺序进行分布式并行初始化，
                                     # 否则按照tp-cp-ep-dp-pp的顺序进行分布式并行初始化
```

## 3、TRAINING_ARGS
TRAINING_ARGS是seq-length、batch-size、lr、optimizer等相关的参数。
```
--use-mcore-models                    # 使用Megatron-Core的权重格式
--seq-length                          # 处理的文本的最大长度
--seed                                # python, pytorch, numpy 和 cuda的随机种子
--no-check-for-nan-in-loss-and-grad   # 不检查loss和gradient是否是NaN 
--eod-mask-loss                       # 不进行eod token的loss计算
--micro-batch-size                    # micro batch size
--global-batch-size                   # global batch size必须是micro-batch-size*dp的倍数
--train-iters                         # 训练过程共运行多少个iterations
--lr                                  # 学习率
--min-lr                              # 学习率最小值 
--lr-warmup-iters                     # 使用多少个iterations线性化预热学习率
--lr-decay-style                      # 学习率衰减函数
--optimizer                           # 选用哪种优化器
--weight-decay                        # 权重衰减系数
--clip-grad                           # 梯度裁剪
--adam-beta1                          # adam优化器的beta1参数
--adam-beta2                          # adam优化器的beta2参数
--adam-eps                            # Adam 优化器中的epsilon（防止除零）
--attention-backend                   # Attention backend，默认值为auto，可选值为(flash,fused,unfused,local,auto)
--use-flash-attn                      # 使用FlashAttention
```

## 4、DATA_ARGS
DATA_ARGS是与tokenizer、dataset等数据处理相关的参数。
```
--tokenizer-type                    # 使用的tokenizer
--tokenizer-model                   # tokenizer model的目录
--actor-tokenizer-model             # actor tokenizer model的目录
--rm-tokenizer-models               # rm tokenizer model的目录
--dataloader-type                   # dataloader的类型，可选值为（single, cyclic, external）
--vocab-file                        # 词表文件的路径
--merge-file                        # BPE合并规则文件的路径
--num-workers                       # 设置dataloader的worker数
--gdatasetv4-train-metadata-file    # 训练dataset的metadata文件，文件格式为json文件，训练dataset是gcore的GDatasetV4
--gdatasetv4-eval-metadata-file     # 评估dataset的metadata文件，文件格式为json文件，评估dataset是gcore的GDatasetV4
--px-shuffle-data                   # 对数据进行局部shuffle
--px-shuffle-buffer-size            # 局部shuffle的buffer的size
```

## 5、OUTPUT_ARGS
OUTPUT_ARGS是与保存checkpoint等输出相关的参数。
```
--log-interval                    # 输出log和timing的间隔
--save-interval                   # 每隔多少个iterations保存一次持久化checkpoint
--tensorboard-dir                 # tensorboard的保存目录
--tensorboard-log-interval        # 每隔多少个iterations保存一次tensorboard log 
--eval-interval                   # 每经过多少个训练iterations进行一次评估
--eval-iters                      # 共进行多少次评估iterations
--wandb-project                   # wandb的project名字
--wandb-exp-name                  # wandb的实验名字
--wandb-save-dir                  # 保存wandb文件的本地目录
```

## 6、EVAL_ARGS
EVAL_ARGS是与评估相关的参数。
```
--ppo-step-eval-interval                # 每隔多少个ppo_step进入一次eval_loop，当该值<=0时不进行eval
--ppo-eval-steps                        # 一次eval_loop有多少个mini batch需要进行评估
--ppo-eval-rollout-global-batch-size    # ppo eval的global batch size
--ppo-eval-rollout-micro-batch-size     # ppo eval的micro batch size
--ppo-eval-sampling-repeat              # ppo eval的sample重复次数
```

## 7、RL_ARGS
RL_ARGS是与强化学习相关的参数。
```
--ppo-early-swap-model                      # 在model update之前是否swap model
--infer-engine-impl                         # ppo sampler的infer engine backend
--ppo-auto-calc-args                        # 自动计算actor总共运行多少个iterations以及每个epoch有多个ppo step
--distributed-timeout-minutes               # torch.distributed的超时时间，单位是分钟
--ppo-display-rollout-generation            # actor中打印rollout generation的log
--ppo-disable-tqdm                          # 禁用tqdm的进度条
--ppo-standalone-sampler                    # 使用独立的sampler server
--hf-config-json-path                       # hf模型的config.json的路径
--ppo-actor-node-ips                        # actor所有节点的ips
--ppo-actor-data-parallel-size              # ppo actor的DP数
--ppo-actor-pipeline-model-parallel-size    # ppo actor的PP数
--ppo-critic-ips                            # ppo critic所有节点的ips
--ppo-critic-ports                          # ppo critic所有节点的端口号、
--ppo-critic-pipeline-model-parallel-size   # ppo critic的PP数
--ppo-critic-tensor-model-parallel-size     # ppo critic的TP数
--ppo-critic-data-parallel-size             # ppo critic的DP数
--ppo-sampler-ips                           # ppo sampler所有节点的ips
--ppo-sampler-ports                         # ppo sampler所有节点的端口号
--sampler-dist-init-addrs                   # sampler所有进程的ip和ports
--ppo-sampler-tensor-model-parallel-size    # ppo sampler的TP数
--ppo-sampler-pipeline-model-parallel-size  # ppo sampler的PP数
--ppo-sampler-data-parallel-size            # ppo sampler的DP数
--ppo-step-update-sampler-interval          # 每隔多少个ppo step更新ppo sampler
--ppo-max-epochs                            # 最多执行多少个epochs
--ppo-max-epochs-2                          # 单个ppo step的数据的epoch
--ppo-step-save-interval                    # 每隔多少个ppo step保存checkpoint
--ppo-step-per-epoch                        # 每个epoch有多少个ppo step
--ppo-rollout-micro-batch-size              # ppo rollout的micro batch size
--ppo-rollout-global-batch-size             # ppo rollout的global batch size
--ppo-resp-seq-len                          # ppo resp的sequence length
--ppo-rollout-pad-to-multiple-of            # ppo rollout的padding
--ppo-logps-fwd-micro-batch-size            # ppo logps forward的micro batch size
--combine-rm-and-critic-server              # ppo合并rm和critic server
--ppo-rollout-top-p                         # ppo actor rollout从概率最高的token开始累加，直到它们的总概率达到P，
                                            # 然后从这个集合中进行采样
--ppo-rollout-top-k                         # ppo actor rollout只考虑概率最高的K个token进行采样
--ppo-rollout-temperature                   # ppo actor rollout的temperature，值越高，概率分布越平滑，生成结果更随机；
                                            # 值越低，分布越尖锐，生成结果更倾向于高概率词元，更确定、更保守
--ppo-ratio-eps                             # ppo clip的ratio eps
--ppo-clip-ratio-low                        # dapo clip-higher的low ratio，该值设置之后将会覆盖--ppo-ratio-eps
--ppo-clip-ratio-high                       # dapo clip-higher的high ratio，该值设置之后将会覆盖--ppo-ratio-eps
--ppo-rm-mask-prompt                        # 当计算loss的时候，ppo acotr reward model需要mask prompt
--rm-output-scalar                          # 输出sequence，表示per token reward
--rm-output-sequence                        # 输出scalar，表示sequence 的 reward
--ppo-sampling-keeping-strategy             # ppo 多重采样保留策略，可选值为['best-and-worst', 'test', 'all']
--ppo-sampling-repeat                       # ppo prompt 重复采样次数
--ppo-sampling-keep                         # ppo prompt 重复采样后保留个数
--ppo-use-absolute-kl                       # 使用KL散度的绝对值
--use-grpo                                  # 使用grpo
--use-gspo-loss                             # 使用gspo loss
--grpo-advantage-epsilon                    # grpo adavantage函数的epsilon
--grpo-kl-loss-beta                         # grpo loss的系数
--rm-head-arch                              # rm head arch，可选值为['single_layer', 'multi_layers']
--ppo-save-first-rollout-data               # 保存第一个rollout数据，以便于进行调试
--ppo-grpo-reward-type                      # grpo reward type，可选值为["rm_only", "rule_only", "rm_with_rule"]
--gen-term-at-nan                           # 当logits的softmax出现Nan时中断
--ppo-rm-reward-alpha                       # rm reward alpha，只有当type为rm_with_rule起作用
--ppo-rule-reward-beta                      # rule reward beta，只有当type为rm_with_rule起作用
--grpo-prefetch-samplings                   # grpo提前预取samples
--ppo-dynamic-sampling-max-replay           # ppo 最大sample次数，只有当设置了--ppo-dynamic-sampling有效
--update-weight-max-size-mb                 # 当更细权重的时候每次最大的size
--sampler-gpu-memory-utilization            # sampler的gpu显存使用的比例
--gen-rm-gpu-memory-utilization             # gen-rm的gpu显存使用比例
--ppo-smart-pad-infer                       # 使用ppo推理的smart_pad
--ppo-smart-pad-train                       # 使用ppo训练的smart_pad
--ppo-train-dynamic-mbs-target-seq          # ppo train动态target sequence的size（单位是mb）
--ppo-train-dynamic-mbs-limit               # ppo train动态microbatch限制（单位是mb）
--ppo-partial-rollout-global-batch-size     # ppo partial rollout的global batch size
--no-fused-kernel                           # 不使用fused kernel
--dapo-overlong-penalty                     # 使用dapo overlong penalty
--dapo-overlong-buffer-len                  # dapo overlong penalty的buffer长度
--dapo-overlong-penalty-factor              # dapo overlong penalty factor
--ppo-dual-clip-ratio-c                     # dual-clip ppo的ratio
--ppo-actor-freeze-ppo-steps                # ppo actor不更新的step
```

## 8、FINETUNE_ARGS
FINETUNE_ARGS是用来设置是否是断点续训的。
```
--finetune                            # 非断点续训，断点续训是需要去掉这个参数
--no-load-optim                       # 非断点续训，断点续训是需要去掉这个参数
--no-load-rng                         # 非断点续训，断点续训是需要去掉这个参数
```

## 9、MONITOR_ARGS
MONITOR_ARGS是与monitor相关的参数。
```
--do-monitor                            # 打开monitor
--monitor-server-ip                     # monitor server的ip
--monitor-port                          # monitor server的端口号
--auto-set-finetune-arg                 # 允许自动重新设置例如checkpoint目录等参数
```

