### DPO demo


## 关于数据集

数据集同样使用 indexed_dataset, 需要进行预处理，参考 `tasks/math_rl_v3/README.md` 的说明，这里 dpo 的原始 jsonl 文件的每一条遵循下面格式

```
{
  "instruction": "aaa",
  "chosen": "bbb",
  "rejected": "xxx"
}
```

## 关于 dpo-model-using 使用的几种策略
1. `both`: 同时加载 `policy model` 和 `ref model`，这种方式使用简单，但是在model比较大的时候，存在显存问题;
2. `ref`: 只加载 `ref model`，只在一种场景下使用，即离线计算 `ref model` 的 logps;
3. `policy`: 只加载 `policy model`, 通常在训练中使用, 配合离线计算 `ref model` 使用;


以 qwen3-moe 235B 为例子
## 使用步骤

1. 第一次数据预处理。首先，将原始数据进行预处理。预处理脚本 `tasks/dpo/qwen_demo/qwen3_moe_235B/preprocess_data.sh`，注意修改 `data_path` 到你要处理的数据文件所在目录上。

2. 将 hf ckpt 转为 mlm sft 的 ckpt 格式。可以参考 `tasks/dpo/qwen_demo/qwen3_moe_235B/convert_sft_ckpt.sh`，走 `hf_to_mlm` 分支，注意修改 load ckpt 和 save ckpt 的路径；

3. 将 mlm sft ckpt 转为 mlm dpo ckpt 格式。参考 `tasks/dpo/qwen_demo/qwen3_moe_235B/convert-sft-to-dpo.sh`。如果处理 235B 权重，可能存在内存 oom 问题，加大 topo 并加机器即可。
通过下面方式用双机转
```
bash tasks/dpo/qwen_demo/qwen3_dense/mpi_train.sh tasks/dpo/qwen_demo/qwen3_moe_235B/convert-sft-to-dpo.sh 2
```
对于比较小的 model，单独运行 `tasks/dpo/qwen_demo/qwen3_moe_235B/convert-sft-to-dpo.sh`即可


4. 离线计算 ref model 的 logps. 如果 model 比较小，可以使用 `dpo-model-using = both`，那么可跳过 4 和 5 步。
4.1 将第一步预处理的数据路径填到 `tasks/dpo/qwen_demo/qwen3_moe_235B/infer_data.json` 的 `eval_data_infos` 的 path 字段中；
4.2 编辑 `tasks/dpo/qwen_demo/qwen3_moe_235B/infer_dpo.sh` 的 `DATA_CONFIG` 成 4.1 中的 infer_data.json，设置 `SAVE_MARGIN_DIR` 为离线 logps 数据的的保存路径；
4.3 然后修改 infer_dpo.sh 的 `LOAD_CHECKPOINT_DIR` 为第 3 步的ckpt路径。执行 infer_dpo.sh 做离线处理。多机运行同样用 `tasks/dpo/qwen_demo/qwen3_dense/mpi_train.sh` 驱动。

需要注意的是离线处理需要满足 `tp_size * pp_size * cp_size == world_size`, 即强制 `dp_size = 1`。

5. 第二次数据预处理。将第4步离线计算得到数据，即上面 `SAVE_MARGIN_DIR` 保存下来的数据再次做预处理。注意修改 `preprocess_data.sh` 的 data_path. 

6. dpo 训练。
6.1 将第 1 （如果用 both 的话）/ 5 步处理得到的数据填入 `tasks/dpo/qwen_demo/qwen3_moe_235B/dpo_data.json` 的 `train_data_infos` 的 path 字段；
6.2 参考 `tasks/dpo/qwen_demo/qwen3_moe_235B/train_dpo.sh` 脚本。其中 `LOAD_CHECKPOINT_DIR` 设置为上面第 3 点转完的 dpo 模型的路径。

  注意下面两点：
  a. `MICRO_BATCH_SIZE` 必须是 2的倍数，因为每一条数据实际上有 chosen 和 rejected 部分，需要成对输入；
  b. 如果 `--dpo-model-using` 设置 `ref` 或 `policy`，则 `--dpo-policy-ref-model-cnt 1`, 否则设置 `--dpo-policy-ref-model-cnt 2`.

7. dpo ckpt 转为 mlm sft ckpt. 参考 `tasks/dpo/qwen_demo/qwen3_moe_235B/convert-dpo-to-sft.sh`;
8. mlm sft ckpt 转为 hf ckpt. 参考 `tasks/dpo/qwen_demo/qwen3_moe_235B/convert_sft_ckpt.sh`, 走 `mlm_to_hf` 分支。


