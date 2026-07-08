import sys


class TraceContext:
    def __init__(self, max_depth=10, do_trace=True):
        self.original_trace = None
        self.depth = 0
        self.max_depth = max_depth  # 最大追踪深度，可配置
        self.do_trace = do_trace

    def __enter__(self):
        self.original_trace = sys.gettrace()
        self.depth = 0  # 调用深度，用来控制缩进
        sys.settrace(self._trace)
        if self.do_trace:
            print("\n=== 进入 Trace 上下文 ===")
        return self

    def __exit__(self, *args):
        sys.settrace(self.original_trace)
        if self.do_trace:
            print("=== 退出 Trace 上下文 ===\n")

    def _trace(self, frame, event, arg):
        if not self.do_trace:
            return self._trace

        func_name = frame.f_code.co_name

        # 过滤系统函数
        if func_name.startswith("<"):
            return self._trace

        # 函数调用时：深度+1，打印缩进
        if event == "call":
            if self.depth <= self.max_depth:
                filename = frame.f_code.co_filename  # 文件
                lineno = frame.f_lineno  # 行号
                # 🔥 最终打印：缩进 + 函数 + 文件 + 行
                indent = "  " * self.depth
                print(f"{indent}📌 {func_name}()  |  file: {filename}  |  line: {lineno}")

            self.depth += 1

        # 函数返回时：深度-1
        elif event == "return":
            self.depth = max(self.depth - 1, 0)

        return self._trace
