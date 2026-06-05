# AI coding 建议

本文档是 AI coding 一些操作的经验积累，持续更新中。

AI 可以做的事情：
1. 写测试（强烈推荐）
2. 写框架和粗略的流程；
3. 重写流程，clean up 代码；
4. 在 gemini 跑测试，做简单的 fix（一定要 review，AI 会乱 fix）。

AI 不能做的事情：
1. 抽象问题思考
	1. 思考本质，例如突破 tokenizer 与 RM 的 prompt 的表象，考虑到 LCS 编辑问题的本质；
	2. 不能理解冷门抽象的算法，边界处理容易错误；
	3. 多线程多协程多进程 gpu 混合编程，容易写出奇怪的 lock。
2. 工程质量的长期维系

## 目录整理

首先，先创建本地代码目录 `~/work/wepsdl`（就从 `https://git.woa.com/groups/wepsdl/-/projects/list` 大目录 clone）：
```bash
~/work/wepsdl
	├── CLAUDE.md
	├── gcore-dev
	├── mbridge
	├── Megatron-Bridge
	├── Megatron-LM
	├── sglang
	└── vllm
```

CLAUDE.md 可参考 `gcore-dev/docs/source/CLAUDE-example.md`；建议你先读一下里面的内容。

然后 mount 你的 CephFS 远程开发目录，参考：
- https://iwiki.woa.com/p/1356046360
- https://iwiki.woa.com/p/4015254477
- https://iwiki.woa.com/p/4013358526

## 远程测试

先在 GEMINI 开资源，略过不解释。
拿到服务器 head 的跳转命令
```bash
./gemini-go <container> <token>
```

参考 https://git.woa.com/kaiyuanpeng/gemini-remote-mcp 安装 MCP：
```
claude-internal mcp add wx-gemini-remote -- gemini-mcp
MCP_TIMEOUT=86400000 GEMINI_PATH="$HOME/work/gemini-remote-mcp/bin/gemini-go" claude-internal
```

## 远程目录 mount

不要让 AI 做代码同步...，我实际测试发现 claude 会干奇怪的事情，不停 ask for permission。所以建议你用 lsyncd 自动做同步。

```bash
-- wepsdl.conf，注意替换 $YOUR_RTX 。
targets = {
  "/mnt/ceph-hz1-csp/mm-base-plt2/$YOUR_RTX/work/wepsdl",
}
for _, target in ipairs(targets)
do
  sync {
    default.rsync,
    source="/home/$YOUR_RTX/work/wepsdl",
    target=target,
    exclude={ "__pycache__", ".git", },
    delete=false,
    maxDelays=1,
    delay=1,
    rsync={
        binary="/usr/bin/rsync",
        archive=true,
        compress=false,
        verbose=true,
    }
  }
end
```

然后启动自动同步：
```bash
pkill -9 -f lsyncd
lsyncd wepsdl.conf
```

Claude 的模型对于 working directory 不太敏感，为了防止它乱跑，我在 CLAUDE.md 里明确写了工作目录。
由于可能会有多个 Agent 同时在做不同的事情，为了避免一直修改 CLAUDE.md，所以我个人的做法是在不同的 gemini cluster 里把
不同的工作目录 link 到同一个路径下。

在 gemini job 1 所有 pod 批量执行：
```bash
mkdir /work
ln -s /mnt/ceph-hz1-csp/mm-base-plt2/yourrtx/work/wepsdl-todo-1 /work/wepsdl
```

在 gemini job 2 所有 pod 批量执行：
```bash
mkdir /work
ln -s /mnt/ceph-hz1-csp/mm-base-plt2/yourrtx/work/wepsdl-todo-2 /work/wepsdl
```

## PYTHONPATH

如前所述，Claude 对于工作目录不太敏感，而且有时候我们会修改 Megatron / mbridge 等代码，所以最好是在 bash 里直接写死 PYTHONPATH，避免 AI 乱链接。

而且，明确告诉 AI 要 run 那个 bash，避免 AI 自作聪明胡乱忽略 py path 或者是 ray setup：
```
运行 tests/my_test.sh，测试 xxx，根据测试结果，修复代码。
```

## 清理

参考 `tests/test_gpatch_v4/mpirun-stop-ray.sh`，避免 pkill，否则 AI 会傻傻杀自己 MCP。

## 共享经验

如果大家有更好的用法经验，请 share 给我！我来改进文档。