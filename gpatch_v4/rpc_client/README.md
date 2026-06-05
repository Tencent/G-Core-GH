# rpc_client — 底层 RPC 通信

底层 RPC 客户端实现，供 `client/` 模块使用。`RpcClientFactory` 根据 `rpc_type` 选择通信后端。

## 支持的 RPC 类型

| rpc_type | 实现类 | 说明 |
|----------|--------|------|
| `http` | `HttpRpcClient` | HTTP 请求 |
| `zeromq` | `ZeroMqRpcClient` | ZeroMQ 异步通信 |
| `ray` | `RayRpcClient` | Ray actor 调用 |
| `ray` (multi_cast) | `MultiCastRayRpcClient` | Ray 多播调用 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `base_rpc_client.py` | `RpcClient` 基类、`HttpRpcClient`、`ZeroMqRpcClient` |
| `ray_rpc_client.py` | `RayRpcClient` — 通过 `ray.get_actor` 直接调用 |
| `multi_cast_ray_rpc_client.py` | `MultiCastRayRpcClient` — 向多个 Ray actor 广播 |
