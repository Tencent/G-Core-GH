## 背景

aigc 训练 trainloop 不像 llm, vlm 那么规范， 每个任务的 train loop 都不太一样， 把那些任务的 train loop 纳入到 gcore 似乎不太现实。但不同的任务也有一些可以复用的部分。gcore light 抽象出分布式策略，ep dispatch， dcp 等通用的部分作为轻量级的**分布式训练脚手架 （scaffold）**， 让不同的 aigc 任务可以迅速地构建自己的 trainloop。gcore light 是 gcore 里自包含的一个子模块，可以直接导入，也可以直接 copy 到业务项目。


## 核心 feature


### 一、并行策略

支持以下并行策略

#### 1、 hybrid fsdp, cp

* 基础的分布式 device mesh 管理
* ac，all-gather prefetch, fully_shard
* grad clip

#### 2、ep

以不侵入主网的方式支持 ep

### 二、dcp


dcp 支持 fsdp 与 ep




### 三、国产卡适配

目前支持 npu, mlu


### 四、底层库的 patch

对  torch， torch_mlu,  torch_npu 的修复与优化通过 patch 的方式引入


### 五、算子优化

TODO:  长视频可能需要sparse attention


## 业务接入

* wegen omni
* wegen video
* hunyun image 3.0 sft 任务
