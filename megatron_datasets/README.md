[TOC]

# MEGATRON-DATASETS

# Indexed JsonL Dataset

之前数据集有两种：

1. 一种是直接使用 jsonl，由于数据集太大了，无法使用 map dataset（完全在内存），只能使用 iterable dataset（用时读文件）。
然而 jsonl 无法确定 offset，断点重启需要跳过样本 + tokenize，效率低下，耗时超过 1 天。
2. 另一种是 megatron-lm 预处理提前 tokenize 的数据集，可以启动时快速定位（或者快速跳过，因为不需要 tokenize）。
之前 pretrain 便是这中方式，但是在 gemini 即使多进程处理也很慢，需要额外用 spark cpu 集群处理。
另外，也非常不灵活，不能增加一些不定长的字段。

所以，我们搞了第三种介于两者之间的数据集格式。数据文件主体仍然是 jsonl，保持了数据灵活。
为了快速热启动，我们提前进行轻量的预处理，为每个条目创建索引，记录 offset 跳转位置（处理 1B jsonl 只需要 50 sec with 224 GPU）。
训练时，会记录消费进度，热启动后只读取索引快速跳过，到合适位置才开始 tokenize 取数据。

## API

创建数据集：
```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    tokenizer = get_tokenizer()

    from megatron_datasets.indexed_jsonl_pretrain_dataset import build_train_valid_test_datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        rank=torch.distributed.get_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size())
```

训练时记录消费进度：
```python
def get_batch(data_iterator):
    args = get_args()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    update_epoch_and_line(args.train_data_consuming_progresses, torch.distributed.get_rank(), data)
```

## 预处理数据

```bash
# 224 张卡，28 台机器，每张卡起 7 个线程。
# 不受 gpu 数量限制，只要 cpu 能扛住，大小随便开。
# 目前并发的处理粒度是文件粒度。
readonly NP=$((28 * 7))

# 在 gemini 运行
# `--data_folder` 就是数据目录，下面放着 `--data_file_postfix` 结尾的数据文件。
# `--ensure_each_line_forms_valid_json` 是非必需的，但他会检查是否有 invalid json。
# 如果你的数据有压缩过，可以通过 `--decompress`、`--decompress_postfix 'gz'` 先解压缩。
# 目前只支持 gzip（BoG 数据集）。
mpirun -v --allow-run-as-root \
    --hostfile /etc/mpi/hostfile --oversubscribe -np $NP \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 \
    python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
    --data_folder 数据目录 \
    --data_file_postfix 'json' \
    --ensure_each_line_forms_valid_json \
    --decompress \
    --decompress_postfix 'gz' \

# 测试机运行同理，但 mpi 命令略有不同。
mpirun -v --allow-run-as-root \
    --oversubscribe -np $NP \
    -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH \
    python3 megatron_datasets/preprocess_indexed_jsonl_dataset.py \
```

## 增加业务自定义数据集

在 `megatron_datasets/tasks/` 目录下添加任务代码。

数据集实现代码参考：
- [预训练数据集](https://git.xxx.com/pretrainx-team/Megatron-LM/blob/pxmain-core_r0.5.0/megatron_datasets/indexed_jsonl_pretrain_dataset.py)
- [SFT 数据集（带样本拼接）](https://git.xxx.com/pretrainx-team/Megatron-LM/blob/pxmain-core_r0.5.0/megatron_datasets/tasks/packed_sft_dataset.py)

数据集使用代码参考：
- [相关性 post train 任务](https://git.xxx.com/pretrainx-team/Megatron-LM/blob/pxmain-core_r0.5.0/tasks/relevance_post_train/train.sh)


## 多模态数据处理

1. 多模态的文本数据全部写入到jsonl文件中，这些jsonl文件构建索引和上面的流程一样
2. 图片与视频等数据可以使用绝对路径来引用，全部写入到lmdb中，这里推荐使用lmdb，因为最多每台机器起一个lmdb server进程就好，可以节省大量的显存。另外，tar包的格式因为速度问题，已经被抛弃了。
3. qwen2vl的数据格式及注意事项如下：
```json
{
    "conversations": [
        {
            "role": "user",
            "content": "挂在交通灯杆上的是什么？<image>"
        },
        {
            "role": "assistant",
            "content": "一个绿色的街牌挂在交通灯杆上。"
        }
    ],
    "images": [
        {
            "image_path": "7_0.png",
        }
    ]
}
```
qwen2vl要求：
1. 必须要有一张图片
2. images也可以只写image_path绝对路径，这样就可以直接读。使用lmdb时，相对路径也能读
3. 每一条样本都要是有效样本

由于每个业务原来的数据格式可能都不太一样，这里的写了一个mpi可以直接跑的脚本，用于转换数据，样例如下：
```bash
mpirun -v --allow-run-as-root \
    --oversubscribe -np 32 \
    --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 \
    python3 megatron_datasets/tools/mm_convert.py \
    --jsonl_filepath your_jsonl_path.jsonl \
    --save_dir your_save_path \
    --images_dir your_images_dir
```
- 上面的`--images_dir`如果没有填写，那么就按如下规则来：
    ```python
    root_dir = os.path.dirname(jsonl_filepath) if os.path.isabs(jsonl_filepath) else os.getcwd()
    ```
