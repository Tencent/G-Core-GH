# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

## dataset

Paper use DAPO-MATH dataset. You can download processed data file from https://mirrors.tencent.com/repository/generic/wepsdl/data/bytedtinghua-sia/dapo-math-17k-v2.jsonl and put it to `./data/DAPO-Math-17k-jsonl/train/dapo-math-17k-v2.jsonl`

## train

1. prepare Qwen2.5-7B-instruct mlm model
2. experiments (hyper params are aligned with verl):
```shell
# baseline 
bash tasks/entropy/mpirun-grpo-baseline.sh
# kl-cov
bash tasks/entropy/mpirun-grpo-kl-cov.sh
# baseline 
bash tasks/entropy/mpirun-grpo-clip-cov.sh
```