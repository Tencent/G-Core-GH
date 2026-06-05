G-Dataset
=========

g-dataset-v4 是项目下数据集支持。g-dataset 强调简单快速。

g-dataset 可以作为组件独立工作，被 import 到 transformers / diffusers 的项目中。

## Why g-dataset?

### Memory Consumption

pytorch 的 map dataset 启动缓慢，容易 oom；pytorch 的 iterable dataset 提供能力很少（shuffle，length，dp sharding）。

g-dataset 是 iterable dataset，并且提供了 shuffle、length、map、dp sharding 等能力。

### Multi-Modal data

目前的 datasets 都是闭环的，采用类似 tar / parquet 的结构存储。不支持从后端存储读取大块数据（image、video）。

显然 naive 的想法是 image / video 文件存储在 dfs，使用上及其方便。分布式存储（如 cephfs）由于 name node 等原因，往往对文件数量有比较严格的 quota 限制；例如腾讯的 cephfs 默认只会分配 100 万的 file num quota；所以这显然是不可行的。

如果模型开发者需要频繁调整数据，使用 tar（特别是在 dfs 上的 tar），频繁 unpack 再 pack，会比较低效且痛苦。

g-dataset-v4 的数据存储在 jsonl，但其中的图像等数据存储在远端的 kv / 对象存储，运行时惰性加载。

### Resuming (and scalable)

huggingface accelerate + datasets 对 iterable dataset 的 resuming 支持很差，重启可能耗费数小时。

支持 scalable gpu cluster（variable data parallel size）的 iterable dataset。

对于这两个需求，g-dataset-v4 提供了简单的实现。

## Installation

```bash
# 从代码安装
git clone https://github.com/Tencent/Wechat-YATT
pip3 install Wechat-YATT

# 或者如果在内网
pip3 install gcore --index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```

## API

`gdataset.GDatasetV4`

数据集基础类型，包括了 iterable、lazy loading、shuffling、resuming、scalable dp 的实现。

```python
from torch.utils.data import IterableDataset as TorchIterableDataset

class GDatasetV4(TorchIterableDataset):

    def __init__(
        self,
        metadata_file,           # metadata file 路径
        dp_rank=-1,              # data parallel rank
        dp_size=-1,              # data parallel size
        gbs=1,                   # global batch size = micro batch size * gradient accumulation step * dp size
        shuffling_buffer_size=0, # iterable dataset 只支持局部 shuffle
        seed=0,                  # 随机数种子
        consumed=0,              # 消费掉的数据数量 = train_iters * gbs
        feats=None,              # 复杂数据类型，见 `gdataset.Feat`
    ):
```

`gdataset.feat.base.Feat`

大多数的简单字段可以被 json 自动确定类型，复杂的 image / video / audio 需要确定类型和 remote 路径等才能转换。

```python
class Feat(ABC):
```

如果你不理解，可以参考下 `huggingface.datasets` 中关于 `Features` 的[文档](https://huggingface.co/docs/datasets/about_dataset_features)。

```python
dataset = load_dataset("AI-Lab-Makerere/beans", split="train").cast_column("image", Image(decode=False))
```

`gdataset` 的 `Feat` 提供了类似的功能，同时还封装了 remote storage 读取的能力。具体可参考 `gdataset.PilImageFeat`。

`gdataset.PilImageFeat`

```python
class PilImageFeat(Feat):

    def __init__(
        self,
        cos=False, # 是否存储在 cos，否则存储在 LFS
        field_value_is_json_str=False, # 如果输入数据是 string，默认是 cos 路径，也可能 parse 之后是 json
    ):
```

`gdataset.JsonFeat`

将 uniondb 存的 string 转换成 json obj。

```
class JsonFeat(Feat):

    def __init__(
        self,
        nest=True,
    ):
```

`gdataset.default_collate`

一个简单的 `torch.data.default_collate` 的包裹，除了 `int`、`float`、`str` 和 `tensor`，还能处理 PIL image 等，如果不喜欢可以自己写。

## 数据格式

`gdataset` 是 jsonl，但对数据的格式有点小小的要求。为了方便从 remote storage 获取文件，字段会用 dict 来表示，而不是一个简单的字符串。

例如，一个数据文件会表示成

```json
  {
      "fp": "/data1/name/xxx.jsonl"
  }
```
```json
  {
      "uniondb": {
          "columns": [
              "CF1_bucket_res",
              "CF1_caption",
              "CF3_cos_url"
          ]
      },
      "udb_key_file": "/mnt/shangcephfs/xxx/union_id_list_110w_0617.txt"
  }
```

而不是 `"xxx.jsonl"`（无法表示 cos bucket、uniondb table 等信息）。

### metadata 文件（类似目录）

1. 原始方式

```json
{
    "name": "demo",
    "description": "这是一个测试的数据 dataset，bla",
    "cos_region": "ap-shanghai",
    "cos_bucket_name": "cos 桶名，如果数据里没有写桶名，就默认这个",
    "cos_secret_id": "see https://cloud.tencent.com/document/product/436/40762",
    "cos_secret_key": "see https://cloud.tencent.com/document/product/436/40762",
    "udb_endpoint": "http://mmpltuniondbapi.polaris:29928 ，咨询 lucienxian",
    "udb_table": "udb table name，咨询 lucienxian",
    "udb_token_key": "udb token，咨询 lucienxian",
    "data_files": [
      	{
            "fp": "/data1/name/xxx.jsonl"
        },
        {
            "uniondb": {
                "columns": [
                    "CF1_bucket_res",
                    "CF1_caption",
                    "CF3_cos_url"
                ]
            },
            "udb_key_file": "/mnt/shangcephfs/xxx/union_id_list_110w_0617.txt"
        }
    ],
    "data_file_num_lines": [
        999999,
        999999,
    ]
}
```

2. 通过北极星sdk访问的方式

安装：pip install polaris-cpp-py --index-url https://mirrors.cloud.tencent.com/pypi/simple/

```json
{
    "name": "demo",
    "description": "这是一个测试的数据 dataset，bla",
    "cos_region": "ap-shanghai",
    "cos_bucket_name": "cos 桶名，如果数据里没有写桶名，就默认这个",
    "cos_secret_id": "see https://cloud.tencent.com/document/product/436/40762",
    "cos_secret_key": "see https://cloud.tencent.com/document/product/436/40762",
    "cos_service": "see https://iwiki.woa.com/p/1490527393", 
    "udb_namespace": "Production",
    "udb_service": "udb的北极星service ，咨询 lucienxian",
    "udb_table": "udb table name，咨询 lucienxian",
    "udb_token_key": "udb token，咨询 lucienxian",
    "data_files": [
        {
            "uniondb": {
                "columns": [
                    "CF1_bucket_res",
                    "CF1_caption",
                    "CF3_cos_url"
                ]
            },
            "udb_key_file": "/mnt/shangcephfs/xxx/union_id_list_110w_0617.txt"
        }
    ],
    "data_file_num_lines": [
        999999,
        999999,
    ]
}
```

3. 通过 aksk托管平台获取敏感数据凭证的方式（Gemini 平台）

当不希望在配置文件中明文存储 `cos_secret_id` 和 `cos_secret_key` 时，可以使用 STS 托管模式。该模式通过内部aksk托管平台动态获取临时凭证。
参考文档：https://iwiki.woa.com/p/4008321299

**前置条件**：
- 环境变量 `YARD_APP_SERVICE_TICKET` 已设置（ac票据，用于访问托管平台，Gemini上的环境变量自带）
- 确保票据具备访问aksk的权限

```json
{
    "name": "demo",
    "description": "使用 aksk托管平台 托管模式获取 COS 凭证",
    "cos_region": "ap-shanghai",
    "cos_bucket_name": "cos 桶名，如果数据里没有写桶名，就默认这个",
    "cos_credential_type": "aksk",
    "cos_aksk_config": {
        "asset_name": "资产名称，如 s_wxg_aigcdata_xxx",
        "access_point": "访问点，如 ap-shanghai",
        "application_name": "应用名称，如 p_mmvisionaigcdata"
    },
    "cos_service": "see https://iwiki.woa.com/p/1490527393", 
    "udb_namespace": "Production",
    "udb_service": "udb的北极星service ，咨询 lucienxian",
    "udb_table": "udb table name，咨询 lucienxian",
    "udb_token_key": "udb token，咨询 lucienxian",
    "data_files": [
      	{
            "uniondb": {
                "columns": [
                    "CF1_bucket_res",
                    "CF1_caption",
                    "CF3_cos_url"
                ]
            },
            "udb_key_file": "/mnt/shangcephfs/xxx/union_id_list_110w_0617.txt"
        }
    ],
    "data_file_num_lines": [
        999999
    ]
}
```

**字段说明**：
- `cos_credential_type`: 凭证类型，设置为 `"aksk"` 表示使用 aksk 托管模式（默认为前两种的明文模式）
- `cos_aksk_config.asset_name`: 资产名称，从 aksk 托管平台获取
- `cos_aksk_config.access_point`: 访问点，如 ap-shanghai
- `cos_aksk_config.application_name`: 应用名称，从 aksk 托管平台获取

**注意**：使用 aksk 模式时，无需配置 `cos_secret_id` 和 `cos_secret_key` 字段；不支持多线程使用

4. 通过 Taiji 证书获取敏感数据凭证的方式（Taiji 平台）

当使用 Taiji 平台时，可以通过本地证书文件获取临时凭证。该模式通过证书向 Taiji STS 服务请求临时凭证。

**前置条件**：
- 拥有有效的证书文件（`*.crt`）和私钥文件（`*.key`）
- 证书文件可访问（建议使用绝对路径）

```json
{
    "name": "demo",
    "description": "使用 Taiji 证书模式获取 COS 凭证",
    "cos_region": "ap-shanghai",
    "cos_bucket_name": "cos 桶名，如果数据里没有写桶名，就默认这个",
    "cos_credential_type": "taiji_certfile",
    "cos_taiji_certfile_config": {
        "asset_name": "资产名称，如 s_wxg_aigcdata_xxx",
        "access_point": "访问点，如 ap-shanghai",
        "application_name": "应用名称，如 p_mmvisionaigcdata",
        "certfile": "/path/to/aigc_client.crt",
        "keyfile": "/path/to/aigc_private.key"
    },
    "cos_service": "see https://iwiki.woa.com/p/1490527393", 
    "udb_namespace": "Production",
    "udb_service": "udb的北极星service ，咨询 lucienxian",
    "udb_table": "udb table name，咨询 lucienxian",
    "udb_token_key": "udb token，咨询 lucienxian",
    "data_files": [
      	{
            "uniondb": {
                "columns": [
                    "CF1_bucket_res",
                    "CF1_caption",
                    "CF3_cos_url"
                ]
            },
            "udb_key_file": "/mnt/shangcephfs/xxx/union_id_list_110w_0617.txt"
        }
    ],
    "data_file_num_lines": [
        999999
    ]
}
```

**字段说明**：
- `cos_credential_type`: 凭证类型，设置为 `"taiji_certfile"` 表示使用 Taiji 证书模式
- `cos_taiji_certfile_config.asset_name`: 资产名称
- `cos_taiji_certfile_config.access_point`: 访问点，如 ap-shanghai
- `cos_taiji_certfile_config.application_name`: 应用名称
- `cos_taiji_certfile_config.certfile`: 证书文件路径（建议使用绝对路径）
- `cos_taiji_certfile_config.keyfile`: 私钥文件路径（建议使用绝对路径）

**注意**：
- 使用 taiji_certfile 模式时，无需配置 `cos_secret_id` 和 `cos_secret_key` 字段
- 确保证书文件和私钥文件存在且有读取权限

### 数据文件

数据文件是 **jsonl**，**一行一条数据**。这里为了方便阅读用 `jq` 格式化了，并且省略了大部分字段。

cos 数据需要包含 `cos`、`cos_url`，与 `cos_bucket_name` 字段。

```json
{
  "height": 1280,
  "width": 1282,
  "bucket_res": "1024_1024",
  "aes_pcg_color": "1.688",
  "aes_pcg_comp": "1.744",
  "aes_pcg_illum": "1.491",
  "face_info": "{}",
  "MD5": "1864654ebd1073c463c61381f734355d",
  "image": {
    "cos": true,
    "cos_url": "xxx/yyy/zzz.jpg",
    "cos_bucket_name": "wxg-processdata-1258344707"
  }
}
```

## Example

**注意不要使用 sampler（dp sharding 和 shuffling 我们都处理了）**。


