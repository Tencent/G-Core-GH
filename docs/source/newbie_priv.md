# 新人阅读

## 腾讯特定

1. WXG devcloud 初始化 https://iwiki.woa.com/p/928878373
2. 二组的新人指引：https://git.woa.com/wxg-infra/plt2/issues/22
3. cephfs
  - https://iwiki.woa.com/p/1356046360
  - https://iwiki.woa.com/p/4015254477
  - https://iwiki.woa.com/p/4013358526

我们这里的开发模式是，代码写在 devcloud，然后通过 rsync / lsyncd / i-FT 同步到 devcloud 上挂载的 cephfs，
然后 gpu 机器也 mount 这个 cephfs，直接跑 devcloud 同步过去的代码。

## Demo

- <a href="/math_sft.html">Math SFT</a>
- <a href="/math_grpo.html">Math GRPO</a>
- <a href="/math_grpo_gen_rm.html">Math GRPO Gen-RM</a>

## Paper

算法论文：
1. transformers
  1. attn is all u need
  2. gpt tech reports
  3. llama tech reports
  4. Qwen tech reports
2. resnet https://arxiv.org/abs/1512.03385
3. t2i
  1. sd / sd2 / sd3
  2. flux
3. rl
  1. Dseek v3/r1
  2. ppo, grpo, gspo
  2. ddpo, flowgrpo, dancegrpo, mixgrpo, refl

工程论文：
1. tp pp sp
  1. tp https://arxiv.org/pdf/1909.08053
  2. pp https://arxiv.org/pdf/2104.04473
  3. sp https://arxiv.org/pdf/2205.05198
  4. fsdp https://arxiv.org/abs/2304.11277
  5. z1 https://arxiv.org/abs/2304.11277
  6. z3 https://arxiv.org/abs/1910.02054
2. cp
  1. ring cp https://arxiv.org/abs/2310.01889
  2. uly cp https://arxiv.org/abs/2309.14509
  3. ag cp https://arxiv.org/abs/2407.21783
  4. dyn cp https://arxiv.org/abs/2402.15627
3. moe
  1. mlm ep https://arxiv.org/abs/2504.14960
  2. https://arxiv.org/abs/2303.06318
  3. deepspeed moe https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf
  4. megablock https://arxiv.org/abs/2211.15841
4. rl
  1. gcore https://arxiv.org/abs/2508.07970
  2. openrlhf https://arxiv.org/abs/2405.11143
  3. verl https://arxiv.org/html/2409.19256v1
  4. areal https://arxiv.org/abs/2505.24298
  5. Longcat-Flash-Thinking https://arxiv.org/abs/2509.18883
  6. APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation
  7. RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training
  8. r3 https://arxiv.org/abs/2505.13388
5. 其他
  1. wlb llm: https://arxiv.org/abs/2503.17924

排查莫名的算法问题：
1. [cx 典范](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)


## 集群挂载
访问A100集群需要挂载 `ceph-nj2-csp` 