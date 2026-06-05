# GPatch v4 日志系统设计文档

## 1. 日志目录结构

每次训练任务启动时，会自动创建一个 task-wise 日志目录，目录名包含实验名和时间戳：
注意，日志也会打印到 console 上
```
<base_dir>/<run_name>_YYYYmmdd_HHMMSS/
├── training.log                        # driver/train_main 单写的聚合训练日志 (INFO 级别，可选包含 DEBUG)
├── debug_log/                          # DEBUG-only 分片目录 (仅 debug_log_to_file=true 时生成)
│   ├── debug_distill_student_actor_rank0_pid123.log
│   ├── debug_distill_teacher_actor_rank0_pid456.log
│   └── ...
└── infer_engine_log/                   # 推理引擎 stdout/stderr 分片目录
    ├── sglang_sampler_engine0_rank0.log
    ├── sglang_sampler_engine1_rank8.log
    ├── vllm_gen_rm_engine0_rank0.log
    └── ...
```

`<base_dir>` 的优先级：
1. `report.log_dir` 如果设置了，直接作为 base → `<report.log_dir>/<run_name>_YYYYmmdd_HHMMSS/`
2. 否则 fallback 到 `checkpoint.save_ckpt_path/logs/` → `<save_ckpt_path>/logs/<run_name>_YYYYmmdd_HHMMSS/`
3. 再依次 fallback 到 `training.checkpoint_dir/logs/`、`checkpoint_dir/logs/`
4. 如果以上路径都没有配置，则使用当前工作目录下的 `logs/<run_name>_YYYYmmdd_HHMMSS/`

任务启动后会在 stdout 打印一行：
```
[GCore logging] Redirect logs to path /absolute/path/to/log_dir
```

## 2. 日志文件说明

| 文件 | 内容 | 写入方式 | 何时产生 |
|------|------|----------|----------|
| `training.log` | driver 聚合后的训练日志；包含 `train_main` 结构化日志、Ray 转发的 actor GPatch structured logging、actor stdout/stderr、mbridge/Megatron-LM 等第三方 logging/print；`log_level=debug` 时包含 `log_debug()` | `train_main` direct file handler + Ray `log_to_driver` → driver `TeeStream` → driver 单进程 append | 始终 |
| `debug_log/debug_<role>_rank<rank>_pid<pid>.log` | `log_debug()` 输出的 DEBUG-only 调试信息 (shape, dtype 等)，文件名带 `pid` 避免多进程共享同一 debug 文件 | 各 rank/role/pid 自己写自己的 DEBUG-only 分片文件 | 仅 `debug_log_to_file=true` |
| `infer_engine_log/<backend>_<role>_engine<idx>_rank<rank>.log` | sglang/vLLM engine 进程的原始 stdout/stderr、vLLM Python logger 输出、`ppo_step` 分隔标记等；不是主训练结构化日志 | `os.dup2` fd 重定向 (engine 创建期间) + Python logging FileHandler (vLLM throughput) + marker append | 仅 `capture_infer_engine_log=true` (默认开) |

### 日志行格式

`training.log` 和 `debug_log/` 中每条记录格式为：
```
2026-05-08 20:30:01,234 - INFO - role=distill_student_actor - rank=0 - pid=12345 - node=10.0.0.1 - [RANK 0   ] some message
```

字段：`time` `level` `role` `rank` `pid` `node` `message`

## 3. 配置开关

所有开关均在 `report` 配置段下 (`ReportConfig`)：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `report.log_dir` | `str \| null` | `null` | 自定义日志根目录。不设置时按上文 `<base_dir>` 优先级 fallback |
| `report.log_level` | `str` | `"info"` | `"info"`: `log_debug()` 不输出到 `training.log`；`"debug"`: `log_debug()` 内容也写入 `training.log` |
| `report.debug_log_to_file` | `bool` | `false` | `true` 时，`log_debug()` 内容写入 `debug_log/` 下的 DEBUG-only 分片文件 (即使 `log_level=info` 也生效) |
| `report.capture_infer_engine_log` | `bool` | `true` | `true` 时，sglang/vLLM engine 的 stdout/stderr 重定向到 `infer_engine_log/` 分片文件 |
| `report.log_to_driver` | `bool` | `true` | 低层开关。标准 `gpatch_v4` 训练入口会强制启用 Ray `log_to_driver=True`，actor 不直接写 `training.log`；无需手动设置 |

YAML 示例：
```yaml
report:
  log_dir: null                     # 使用默认路径
  log_level: info                   # 只打印 INFO 以上
  debug_log_to_file: false          # 不生成 DEBUG-only 分片日志
  capture_infer_engine_log: true    # 捕获推理引擎日志
  log_to_driver: true               # 标准训练入口会覆盖为 driver 聚合模式
```

### log_level 与 debug_log_to_file 组合矩阵

| log_level | debug_log_to_file | training.log 含 DEBUG? | 生成 DEBUG-only 分片? |
|-----------|-------------------|------------------------|-----------------|
| `info`    | `false`           | 否                     | 否              |
| `info`    | `true`            | 否                     | 是              |
| `debug`   | `false`           | 是                     | 否              |
| `debug`   | `true`            | 是                     | 是              |

## 4. 日志链路设计

### 4.1 整体流程

```
任务启动 (grpo.sh)
  │
  ▼
orches.init(config)
  ├─ setup_gpatch_logging(role="train_main", rank=0)
  │    ├─ 创建 task log dir
  │    ├─ 安装 train_main direct file handler → training.log
  │    ├─ 安装 TeeStream 捕获 Ray 转发日志和 driver stderr 错误 → training.log
  │    └─ 设置环境变量 (GPATCH_TASK_LOG_DIR, GPATCH_LOG_LEVEL, ...)
  │
  └─ ray.init(log_to_driver=True, runtime_env={"env_vars": PROPAGATE_ENV_KEYS})
       │  ← 环境变量传播到所有 Ray worker
       │
       ▼
  Ray Actor 初始化
  ├─ train_actor.init(config)
  │    └─ setup_gpatch_logging(role="<actor_class_name_in_snake_case>", rank=N)
  │         ├─ 从 env 读取 GPATCH_TASK_LOG_DIR → 复用同一日志目录
  │         ├─ 安装 console handler，不直接打开 training.log
  │         │    └─ GPatch structured logging → Ray log_to_driver → driver
  │         ├─ 安装 root console handler，捕获 mbridge/Megatron-LM 等第三方 logging
  │         └─ 如果 debug_log_to_file=true，安装 DEBUG-only 分片 handler
  │
  └─ InferEngine.from_engine_args(...)
       ├─ configure_third_party_logging(role, backend, engine_idx, rank)
       │    ├─ 在 driver 聚合模式下重新安装 console-only handlers (防止 vLLM dictConfig 清空)
       │    ├─ 如果 capture_infer_engine_log=true:
       │    │    └─ 计算分片路径 infer_engine_log/<backend>_<role>_engine<idx>_rank<rank>.log
       │    └─ 如果 backend=vllm:
       │         └─ 给 vllm logger 追加 FileHandler → 同一分片文件 (捕获 throughput 等)
       │
       └─ redirect_stdio_fds_to_file(infer_engine_log_path)
            ├─ os.dup2(target_fd, 1)  # stdout → 分片文件
            ├─ os.dup2(target_fd, 2)  # stderr → 分片文件
            ├─ Engine 创建 (sgl.Engine / AsyncLLM.from_engine_args)
            │    └─ 子进程继承 fd → 引擎原始输出写入分片文件
            └─ 恢复原始 fd
```

### 4.2 training.log 单写聚合

标准训练入口使用 driver 聚合模式：`training.log` 只由 driver/train_main 进程直接写。actor 不打开 `training.log`，而是把 Python logging 和 stdout/stderr 输出到 console，由 Ray `log_to_driver` 转发给 driver：

```
actor logger.info("msg")
  → actor console handler
  → Ray log_to_driver
  → driver stdout/stderr
  → TeeStream
  → training.log
```

Ray 转发来的 GPatch structured record 会去掉 Ray 前缀后保留原始记录，因此保留原始 `role` / `rank` 字段。Ray actor 的 `role` 来自 actor class name 的 snake_case 形式，例如 `DistillStudentActor` → `distill_student_actor`、`DistillTeacherActor` → `distill_teacher_actor`。Ray 转发来的普通第三方输出（例如 mbridge/Megatron-LM 的 `print`）会包装成 GPatch-style record 后写入 `training.log`，避免混入 raw 行；这类非结构化输出使用 `role=ray_actor`、`rank=none`，原始 Ray actor 前缀会保留在 message 中用于追来源。

如果绕过 `orches.init()` 直接调用 `setup_gpatch_logging(..., log_to_driver=False)`，当前进程会安装 direct `training.log` file handler。这是工具级兼容路径；标准 `gpatch_v4` 训练中只有 `train_main` 走 direct file handler，Ray actors 走 driver 聚合路径。

在 `log_to_driver=True` 的 actor 路径中，console handler 优先写当前 `sys.stderr`，以便 WandB 等 wrapper 继续捕获 UI Logs；如果当前 `sys.stderr` 已经是 driver 侧的 `TeeStream`，则自动回退到 `sys.__stderr__`，避免日志写回 TeeStream 造成重复或递归。`train_main` 的 console handler 仍使用原始 stderr。

### 4.3 stderr 错误捕获

driver 侧 `TeeStream` 替换 `sys.stderr`，当检测到非 Ray 转发的错误模式 (`Traceback`, `CUDA error`, `RuntimeError` 等) 时，仍会自动将错误内容格式化后写入 `training.log`：

```
sys.stderr (Python 层)
  → TeeStream.write(data)
      ├─ 原始输出转发到 console (sys.__stderr__)
      └─ 如果 STDERR_ERROR_PATTERNS 匹配:
           └─ _write_to_training_log(data)
                → append(training_log_fd, formatted_blob)
```

### 4.4 推理引擎日志分片

每个 sglang/vLLM engine 实例的 stdout/stderr 通过 `os.dup2` 重定向到独立文件：

```
redirect_stdio_fds_to_file(path)
  ├─ os.open(path, O_WRONLY | O_CREAT | O_APPEND)
  ├─ os.dup2(target_fd, 1)   # fd 1 (stdout) → 分片文件
  ├─ os.dup2(target_fd, 2)   # fd 2 (stderr) → 分片文件
  ├─ sgl.Engine() / AsyncLLM.from_engine_args()
  │    └─ engine 内部 fork 的子进程继承这些 fd
  └─ finally: 恢复原始 stdout/stderr fd
```

文件命名规则：`<backend>_<role>_engine<idx>_rank<rank>.log`

- `backend`: `sglang` 或 `vllm`
- `role`: `sampler`, `gen_rm`, `off_policy_sampler` 等
- `idx`: engine 在该 actor 内的编号
- `rank`: TP rank

示例：`sglang_sampler_engine0_rank0.log`, `vllm_gen_rm_engine0_rank0.log`

## 5. 环境变量传播

主进程通过 `ray.init(runtime_env={"env_vars": ...})` 将已存在的 task logging 环境变量传播到所有 Ray actor。`orches.init()` 会在 `setup_gpatch_logging()` 后设置核心变量，再从 `PROPAGATE_ENV_KEYS` 中挑选当前已存在于 `os.environ` 的键传给 Ray：

| 环境变量 | 来源 | 用途 |
|----------|------|------|
| `GPATCH_TASK_LOG_DIR` | `derive_task_log_dir()` | 训练任务日志目录 |
| `GPATCH_LOG_LEVEL` | `report.log_level` | actor 里决定是否启用 debug 级别日志 |
| `GPATCH_DEBUG_LOG_TO_FILE` | `report.debug_log_to_file` | actor 里决定是否生成 DEBUG-only 分片日志 |
| `GPATCH_CAPTURE_INFER_ENGINE_LOG` | `report.capture_infer_engine_log` | actor 里决定是否重定向引擎 stdout/stderr |
| `GPATCH_LOG_TO_DRIVER` | `orches.init()` / `report.log_to_driver` | actor 里决定是否走 driver 聚合模式；标准训练入口会设为 `"1"` |
| `GPATCH_LOG_STDIO` | 可选外部环境变量 | 设为 `"0"` 可禁用 TeeStream stderr 拦截；未设置时 `redirect_stdio_to_level_logs()` 按默认 `"1"` 处理，不一定会传播给 actor |

推理引擎相关的 `GPATCH_ENGINE_LOG_FILE`、`GPATCH_ENGINE_LOG_DIR`、`GPATCH_ENGINE_ROLE` 是 per-engine 状态，通常由 actor 内的 `configure_third_party_logging()` 在 engine 初始化时设置。它们也在 `PROPAGATE_ENV_KEYS` 中，但只有在 Ray 初始化前已经存在时才会作为启动环境传播。

## 6. API 速查

```python
from gpatch_v4.utils.logging_utils import (
    log,                           # INFO 级别日志，等价于 logging_with_rank_and_datetime
    log_debug,                     # DEBUG 级别日志，受 log_level / debug_log_to_file 控制
    setup_gpatch_logging,          # 初始化进程级日志配置
    configure_third_party_logging, # 为推理引擎配置日志 (handler 重装 + 分片路径)
    derive_task_log_dir,           # 获取/创建 task log 目录
    get_infer_engine_log_path,     # 计算引擎分片日志路径
    redirect_stdio_fds_to_file,    # fd 级别 stdout/stderr 重定向 (context manager)
    write_infer_engine_log_marker, # 在引擎日志中写入 ppo_step 分隔标记
)
```
