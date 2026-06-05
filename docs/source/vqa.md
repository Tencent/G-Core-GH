# Train a VQA model using SFT/LoRA/DPO/GRPO

目前， gcore 平台在多模态训练方面已全面支持 Qwen2.5-VL/Qwen2-VL和Gemma3模型，提供包括SFT（监督微调）、LoRA（低秩适应）、DPO（直接偏好优化）和GRPO在内的多种训练方法。这些功能已在微信内部多个业务场景中落地应用，为业务效果提升提供了有力支持。为了帮助大家更好地理解 gcore 在多模态训练中的应用，下面介绍从训练 Qwen2.5-VL/Qwen2-VL和Gemma3 的方法。

Currently, the gcore platform fully supports multimodal training for the Qwen2.5-VL/Qwen2-VL and Gemma3 models, offering various training methods including SFT (Supervised Fine-Tuning), LoRA (Low-Rank Adaptation), DPO (Direct Preference Optimization), and GRPO. These features have been implemented in multiple business scenarios within WeChat, providing strong support for enhancing business outcomes. To help everyone better understand the application of gcore in multimodal training, the following introduces the methods for training Qwen2.5-VL/Qwen2-VL and Gemma3.

在使用下面方法之前，先按要求准备好 gcore 的相关环境，包括 Megatron-LM

Before using the methods below, ensure that the relevant gcore environment is prepared, including Megatron-LM.


## Download models and dataset/下载模型及数据集

For Qwen2.5-VL/Qwen2-VL
```bash
set -ex

MYWD=$PWD

repo_id=Qwen/Qwen2.5-VL-3B-Instruct
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type model $repo_id --local-dir $save_dir

repo_id=Qwen/Qwen2.5-VL-72B-Instruct
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type model $repo_id --local-dir $save_dir

repo_id=Qwen/Qwen2-VL-2B-Instruct
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type model $repo_id --local-dir $save_dir

# sft/lora
repo_id=RadGenome/PMC-VQA
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type dataset $repo_id --local-dir $save_dir

# dpo
repo_id=llamafactory/RLHF-V
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type dataset $repo_id --local-dir $save_dir

# grpo
repo_id=hiyouga/geometry3k
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type dataset $repo_id --local-dir $save_dir
```


For Gemma3
```bash
set -ex

MYWD=$PWD

repo_id=google/gemma-3-4b-it
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type model $repo_id --local-dir $save_dir

# sft
repo_id=BUAADreamer/llava-en-zh-300k
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type dataset $repo_id --local-dir $save_dir

# grpo
repo_id=yusuf802/captcha_dataset
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type dataset $repo_id --local-dir $save_dir

# dpo
repo_id=llamafactory/RLHF-V
save_dir=$MYWD/hf-hub/$repo_id
huggingface-cli download --repo-type dataset $repo_id --local-dir $save_dir
```

默认情况，会被下载到 **当前目录** 的 hf-hub 目录里。

By default, they will be downloaded to `$PWD/hf-hub`.

## preprocess dataset/预处理数据

在多模态训练中，我们通常使用 datasetv3 和 `GDatasetV4`。两者的主要区别在于，`GDatasetV4`支持智能重排序功能和动态长度（smart pad），但要求训练时每个样本都是合法的，这意味着在使用前需要对`GDatasetV4`进行过滤。相比之下，datasetv3 不支持智能重排序和动态长度，但可以在训练过程中动态丢弃不合法样本。

In multimodal training, we typically use datasetv3 and `GDatasetV4`. The main difference between the two is that `GDatasetV4` supports intelligent reordering and dynamic length(named: smart pad), but it requires that each sample be valid during training, necessitating pre-use filtering. In contrast, datasetv3 does not support intelligent reordering or dynamic length, but it can dynamically discard invalid samples during training.


此外， gcore 中的多模态数据集会将文本信息与多媒体信息分开存储。文本信息最好包含足够的内容，以便`GDatasetV4`能够过滤出合法样本。多媒体信息通常存储在键值数据库中（例如cos/lmdb等）。

Additionally, in gcore, multimodal datasets store text information separately from multimedia information. It is advisable for the text to contain sufficient information to allow `GDatasetV4` to filter out valid samples. Multimedia information is generally stored in key-value databases (such as cos/lmdb, etc.).


For Qwen2.5-VL/Qwen2-VL
```bash
bash tasks/qwen2vl/sh/preprocess_data.sh
```

For Gemma3
```bash
bash tasks/gemma3/sh/preprocess_data.sh
```

## convert checkpoint

在这里，我们为 Qwen2.5-VL/Qwen2-VL 和 Gemma3 准备了一个脚本，它们的使用方法是相同的：
- 第一个参数是转换类型，选项包括：hf_to_mlm 或 mlm_to_hf。
- 第二个参数是训练模型类型，选项包括：sft、dpo 或 lora，其中 sft 也适用于 grpo。

Here, we have prepared a script for both Qwen2.5-VL/Qwen2-VL and Gemma3, and their usage is identical:
- The first parameter is the conversion type, with options: hf_to_mlm or mlm_to_hf.
- The second parameter is the training model type, with options: sft, dpo, or lora, where sft is also applicable to grpo.


示例如下：

Example:


Qwen2.5-VL SFT/GRPO: huggingface format to megatron-lm format：
```bash
bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh hf_to_mlm sft
```

Qwen2-VL LoRA: huggingface format to megatron-lm format：
```bash
bash tasks/qwen2vl/sh/convert_qwen2vl.sh hf_to_mlm sft
```

Gemma3 DPO: huggingface format to megatron-lm format：
```bash
bash tasks/gemma3/sh/convert_gemma3.sh hf_to_mlm dpo
```

Gemma3 现在暂不支持 LoRA

Gemma3 currently does not support LoRA.


## SFT

本节以 Qwen/Qwen2.5-VL-3B-Instruct 的模型 SFT 为例：

This section uses the Qwen/Qwen2.5-VL-3B-Instruct model SFT as an example:

1. convert huggingface format to megatron-lm format, if not converted
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh hf_to_mlm sft
    ```

2. train
    ```bash
    bash tasks/qwen2vl/sh/train_qwen2p5vl.sh sft
    ```

3. convert the ouput checkpoint to huggingface format, You may need to modify the path of `MLM_INPUT_DIR`
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh mlm_to_hf sft
    ```


对于`GDatasetV4`，以下三个参数需要与预处理数据脚本`tasks/qwen2vl/sh/preprocess_data.sh`中过滤数据集的参数保持一致。如果不一致，则需要运行`tools/filter/qwen2vl_filter.py`来进行数据过滤: 

For `GDatasetV4`, the following three parameters need to be consistent with the dataset filtering parameters in the preprocessing script `tasks/qwen2vl/sh/preprocess_data.sh`. If they are not consistent, you will need to run `tools/filter/qwen2vl_filter.py` to filter the data:

  - `--seq-length`
  - `--max-pixels-num`
  - `--min-pixels-num`
<br>

`GDatasetV4`支持 smart pad，可以通过下面几个选项打开：

`GDatasetV4` supports smart pad, which can be enabled through the following options:
- `--px-shuffle-buffer-size 102400`
- `--px-smart-padding-buffer-size 256`
- `--px-inputs-pad-to-longest`
- `--px-pad-to-multiple-of 128`
<br>

如果没有使用 wandb ，可以删除 wandb 相关选项，或者直接使用命令 `wandb offline`

If you are not using wandb, you can remove wandb-related options or simply use the command wandb offline.

如果要调其它参数量更大的模型，就要先下载好模型以及修改`convert_qwen2p5vl.sh`及`train_qwen2p5vl.sh`内模型的路径，Qwen2Vl 与 Gemma3的使用方法也类似于上面。

If you want to adjust models with larger parameter sizes, you need to download the model first and modify the model paths in convert_qwen2p5vl.sh and train_qwen2p5vl.sh. The usage of Qwen2Vl and Gemma3 is similar to the steps mentioned above.


## LoRA

1. convert huggingface format to megatron-lm format, if not converted. You may need to modify the option `--lora_r`
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh hf_to_mlm lora
    ```

2. train
    ```bash
    bash tasks/qwen2vl/sh/train_qwen2p5vl.sh lora
    ```

3. convert the ouput checkpoint to huggingface format. You may need to modify the path of `MLM_INPUT_DIR`
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh mlm_to_hf lora
    ```

## DPO

1. convert huggingface format to megatron-lm format, if not converted
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh hf_to_mlm dpo
    ```

2. train
    ```bash
    bash tasks/qwen2vl/sh/train_qwen2p5vl.sh dpo
    ```

3. convert the ouput checkpoint to huggingface format. You may need to modify the path of `MLM_INPUT_DIR` and the select model option `--qwen2vl_dpo_choice_model` 
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh mlm_to_hf dpo
    ```


## GRPO

由于 GRPO 有三个角色（actor、sampler、critic 或 gen-rm）,且我们的框架会在同一个 GPU 上共享这些角色的进程，这里是使用 mpirun 程序来拉超来的，如果一些机器上没有 hostfile，单台的情况下可以先查看命令 `hostname`，再新建一个 hostfile 文件，内容如下：

Since GRPO has three roles (actor, sampler, critic or gen-rm), and our framework will share these roles' processes on the same GPU, we use the mpirun program to launch these processes. If some machines do not have a hostfile, in a single-machine setup, you can first use the command hostname to check the hostname, and then create a new hostfile with the following content:

```text
cmd_hostname_output slots=1
```

### Rule only train

1. convert huggingface format to megatron-lm format, if not converted.
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh hf_to_mlm sft
    ```

2. train, all logs are in `LOG_DIR`
    ```bash
    bash tasks/qwen2vl/grpo/mpirun-qwen2p5vl-grpo.sh
    ```

3. convert the ouput checkpoint to huggingface format. You may need to modify the path of `MLM_INPUT_DIR`
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh mlm_to_hf sft
    ```

### Gen RM train

1. convert huggingface format to megatron-lm format, if not converted.
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh hf_to_mlm sft
    ```

2. train, all logs are in `LOG_DIR`
    ```bash
    bash tasks/qwen2vl/grpo/gen_rm/mpirun_gen_rm.sh
    ```

3. convert the ouput checkpoint to huggingface format. You may need to modify the path of `MLM_INPUT_DIR`
    ```bash
    bash tasks/qwen2vl/sh/convert_qwen2p5vl.sh mlm_to_hf sft
    ```

## Q & A
Q: Qwen2VL/Qwen2.5VL/Qwen3VL 中的 `--max-pixels-num` 与 `--min-pixels-num` 是干什么用的：

A: 是将图片的像素点个数限制在 [`--min-pixels-num`, `--max-pixels-num`]

Q: freeze 不同部分的选项有哪些？

A: `--mm-freeze-llm` `--mm-freeze-vision-encoder` `--mm-freeze-projector`

Q: 有没有`warmup_ratio`?

A: 有一个`--lr-warmup-fraction`，也有一个`--lr-warmup-iters `，需要自己计算出来

Q: 为什么要自己计算`--train-iters`?

A: 我们不推荐使用 map dataset，因为这样的数据预处理时间太长了，导致启动训练很慢。使用 iter dataset 的话，在 packed 与丢样本的情况下程序是没有办法知道有多少样本的，所以只能让用户来自己计算。也就是说 gcore 没有 epoch 的概念，要自己通过设置`--train-iters`来得到。上面这些操作都引用自 megatron-lm 的，我们并没有对修改。

