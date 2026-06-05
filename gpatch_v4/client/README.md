# client — RPC 客户端

封装与远程推理 / RM 引擎的通信逻辑。Trainer Actor 通过 Client 向 Sampler、RM、Teacher 等 Actor 发送请求。

## 架构

```
Trainer Actor
  └─ SamplerClient   ──→  Sampler Actor (vLLM/SGLang)
  └─ GenRmClient      ──→  Gen RM Actor
  └─ BtRmClient       ──→  BT RM Actor
  └─ TeacherClient    ──→  Teacher Actor
  └─ KvStoreClient    ──→  KV Store Actor
```

底层 RPC 通道由 `rpc_client/RpcClientFactory` 创建（HTTP / ZeroMQ / Ray）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `base_client.py` | 抽象基类 `BaseClientAbc` 及 `SamplerClientMixin`、`RmClientMixin`、`TeacherClientMixin` |
| `mixin.py` | 请求 / 响应处理的公共 mixin |
| `rpc_req_and_rep.py` | 请求 / 响应数据结构 |
| `sampler_client.py` | `SamplerClient` — 权重更新、采样生成、abort 等 |
| `gen_rm_client.py` | `GenRmClient` — Generative RM 打分（文本） |
| `gen_rm_client_t2i.py` | `T2iGenRmClient` — Generative RM 打分（T2I） |
| `bt_rm_client.py` | `BtRmClient` — Batch RM 打分（文本） |
| `bt_rm_client_t2i.py` | `T2iBtRmClient` — Batch RM 打分（T2I） |
| `teacher_client.py` | `TeacherClient` — 蒸馏场景教师端通信 |
| `kv_store_client.py` | `KvStoreClient` — 分布式 KV 读写 |
| `infer_client.py` | `InferClient` — 纯推理客户端 |
