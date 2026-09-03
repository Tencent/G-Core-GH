import inspect
import linecache
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

import torch


class PythonCallTracer:
    """
    Trace Python function calls / returns using sys.setprofile().

    特点：
    - 只输出 include_paths 中的函数。
    - FROM 不使用直接的 frame.f_back，而是向上查找最近一个
      同样位于 include_paths 中的调用者。
    - 自动跳过 torch.nn.Module.__call__ / _call_impl 等中间层。
    - 记录参数、返回值以及 Tensor metadata。
    - 不读取 Tensor 实际内容，不主动触发 GPU -> CPU copy。

    Example
    -------
    with PythonCallTracer(
        output_file="log/sglang_trace.log",
        include_paths=[
            "/root/sglang",
        ],
    ):
        # some python code
        ...
    """

    def __init__(
        self,
        output_file: str,
        include_paths: Optional[Sequence[str]] = None,
        exclude_paths: Optional[Sequence[str]] = None,
        max_depth: Optional[int] = None,
        max_container_items: int = 8,
        max_string_length: int = 200,
        trace_threads: bool = False,
        flush_each_write: bool = False,
        show_skipped_frames: bool = False,
    ):
        self.output_file = os.path.abspath(output_file)

        self.include_paths = [
            os.path.abspath(p)
            for p in (include_paths or [])
        ]

        self.exclude_paths = [
            os.path.abspath(p)
            for p in (exclude_paths or [])
        ]

        self.max_depth = max_depth
        self.max_container_items = max_container_items
        self.max_string_length = max_string_length

        self.trace_threads = trace_threads
        self.flush_each_write = flush_each_write

        # 是否在 FROM 后显示跳过了多少个中间 frame
        # 例如 torch.nn.Module._call_impl
        self.show_skipped_frames = show_skipped_frames

        self._file = None
        self._local = threading.local()
        self._write_lock = threading.Lock()

        self._old_sys_profile = None
        self._old_threading_profile = None

    # ==================================================================
    # Thread-local state
    # ==================================================================

    def _get_state(self):
        if not hasattr(self._local, "depth"):
            self._local.depth = 0

        return self._local

    # ==================================================================
    # Path filtering
    # ==================================================================

    @staticmethod
    def _normalize_filename(filename: str) -> Optional[str]:
        if not filename:
            return None

        # <frozen ...>, <string>, <stdin> 等
        if filename.startswith("<"):
            return None

        try:
            return os.path.abspath(filename)
        except Exception:
            return None

    @staticmethod
    def _path_belongs_to(filename: str, root: str) -> bool:
        """
        判断 filename 是否等于 root，或者位于 root 目录之下。

        避免：
            /foo/bar2
        被：
            /foo/bar
        错误匹配。
        """
        return (
            filename == root
            or filename.startswith(root + os.sep)
        )

    def _is_included_filename(self, filename: str) -> bool:
        filename = self._normalize_filename(filename)

        if filename is None:
            return False

        # --------------------------------------------------------------
        # include_paths
        # --------------------------------------------------------------

        if self.include_paths:
            included = any(
                self._path_belongs_to(filename, path)
                for path in self.include_paths
            )

            if not included:
                return False

        # --------------------------------------------------------------
        # exclude_paths
        # --------------------------------------------------------------

        if self.exclude_paths:
            excluded = any(
                self._path_belongs_to(filename, path)
                for path in self.exclude_paths
            )

            if excluded:
                return False

        return True

    def _should_trace(self, frame) -> bool:
        return self._is_included_filename(
            frame.f_code.co_filename
        )

    # ==================================================================
    # Find logical caller
    # ==================================================================

    def _find_included_caller(self, frame):
        """
        从给定 frame 开始一路向 f_back 查找，返回最近一个
        位于 include_paths 中、且没有被 exclude_paths 排除的 frame。

        这会把类似：

            DeepseekV4Model.forward
                  ↓
            torch.nn.Module.__call__
                  ↓
            torch.nn.Module._call_impl
                  ↓
            DeepseekV4DecoderLayer.forward

        变成：

            DeepseekV4Model.forward
                  ↓
            DeepseekV4DecoderLayer.forward

        返回：
            (caller_frame, skipped_count)

        caller_frame:
            最近的 include_paths 中的父 frame。

        skipped_count:
            为找到它跳过了多少个不在 include_paths 中的 frame。
        """

        skipped = 0

        current = frame

        while current is not None:
            if self._should_trace(current):
                return current, skipped

            skipped += 1
            current = current.f_back

        return None, skipped

    # ==================================================================
    # Frame formatting
    # ==================================================================

    @staticmethod
    def _frame_name(frame) -> str:
        module_name = frame.f_globals.get(
            "__name__",
            "<unknown-module>",
        )

        # Python 3.11+ 有 co_qualname，
        # 能够得到 ClassName.method，
        # 比单独 co_name 更有用。
        qualname = getattr(
            frame.f_code,
            "co_qualname",
            frame.f_code.co_name,
        )

        return f"{module_name}.{qualname}"

    def _get_source_line(self, frame) -> str:
        try:
            source = linecache.getline(
                frame.f_code.co_filename,
                frame.f_lineno,
            )

            return source.strip()

        except Exception:
            return ""

    # ==================================================================
    # Object summary
    # ==================================================================

    def _summarize(
        self,
        obj,
        depth: int = 0,
    ) -> str:
        """
        对参数/返回值做摘要。

        特别注意：
        不调用 tensor.cpu() / item() / tolist()，
        因此不会为了记录日志主动读取 CUDA Tensor 数据。
        """

        if obj is None:
            return "None"

        # --------------------------------------------------------------
        # Tensor
        # --------------------------------------------------------------

        if isinstance(obj, torch.Tensor):
            try:
                return (
                    "Tensor("
                    f"shape={tuple(obj.shape)}, "
                    f"dtype={obj.dtype}, "
                    f"device={obj.device}, "
                    f"stride={obj.stride()}, "
                    f"requires_grad={obj.requires_grad}"
                    ")"
                )
            except Exception as e:
                return (
                    "<Tensor summary failed: "
                    f"{type(e).__name__}: {e}>"
                )

        # --------------------------------------------------------------
        # nn.Module
        # --------------------------------------------------------------

        if isinstance(obj, torch.nn.Module):
            cls = type(obj)

            return (
                f"<{cls.__module__}.{cls.__qualname__} "
                f"id=0x{id(obj):x}>"
            )

        if isinstance(obj, torch.dtype):
            return str(obj)

        if isinstance(obj, torch.device):
            return str(obj)

        # --------------------------------------------------------------
        # Simple types
        # --------------------------------------------------------------

        if isinstance(obj, (bool, int, float, complex)):
            return repr(obj)

        if isinstance(obj, str):
            if len(obj) > self.max_string_length:
                return repr(
                    obj[: self.max_string_length] + "..."
                )

            return repr(obj)

        if isinstance(obj, bytes):
            if len(obj) > self.max_string_length:
                return (
                    repr(
                        obj[: self.max_string_length]
                    )
                    + "..."
                )

            return repr(obj)

        # --------------------------------------------------------------
        # 防止 container 无限递归
        # --------------------------------------------------------------

        if depth >= 2:
            cls = type(obj)

            return (
                f"<{cls.__module__}.{cls.__qualname__}>"
            )

        # --------------------------------------------------------------
        # tuple
        # --------------------------------------------------------------

        if isinstance(obj, tuple):
            values = []

            for x in obj[: self.max_container_items]:
                values.append(
                    self._summarize(
                        x,
                        depth + 1,
                    )
                )

            if len(obj) > self.max_container_items:
                values.append("...")

            if len(obj) == 1:
                return "(" + values[0] + ",)"

            return "(" + ", ".join(values) + ")"

        # --------------------------------------------------------------
        # list
        # --------------------------------------------------------------

        if isinstance(obj, list):
            values = []

            for x in obj[: self.max_container_items]:
                values.append(
                    self._summarize(
                        x,
                        depth + 1,
                    )
                )

            if len(obj) > self.max_container_items:
                values.append("...")

            return "[" + ", ".join(values) + "]"

        # --------------------------------------------------------------
        # dict
        # --------------------------------------------------------------

        if isinstance(obj, dict):
            values = []

            for i, (key, value) in enumerate(obj.items()):
                if i >= self.max_container_items:
                    values.append("...")
                    break

                values.append(
                    f"{self._summarize(key, depth + 1)}: "
                    f"{self._summarize(value, depth + 1)}"
                )

            return "{" + ", ".join(values) + "}"

        # --------------------------------------------------------------
        # set
        # --------------------------------------------------------------

        if isinstance(obj, set):
            values = []

            for i, value in enumerate(obj):
                if i >= self.max_container_items:
                    values.append("...")
                    break

                values.append(
                    self._summarize(
                        value,
                        depth + 1,
                    )
                )

            return "{" + ", ".join(values) + "}"

        # --------------------------------------------------------------
        # Generic object
        # --------------------------------------------------------------

        # 不直接调用 repr(obj)：
        #
        # 1. 有些对象 repr 非常大
        # 2. repr 可能执行 Python 代码
        # 3. Tensor-like 对象可能读取真实数据
        # 4. 某些框架对象 repr 本身就非常昂贵

        cls = type(obj)

        return (
            f"<{cls.__module__}.{cls.__qualname__} "
            f"id=0x{id(obj):x}>"
        )

    # ==================================================================
    # Function arguments
    # ==================================================================

    def _get_arguments(self, frame) -> str:
        try:
            arg_info = inspect.getargvalues(frame)

            result = []

            # ----------------------------------------------------------
            # 普通参数
            # ----------------------------------------------------------

            for name in arg_info.args:
                if name not in frame.f_locals:
                    continue

                value = frame.f_locals[name]

                result.append(
                    f"{name}={self._summarize(value)}"
                )

            # ----------------------------------------------------------
            # *args
            # ----------------------------------------------------------

            if arg_info.varargs is not None:
                name = arg_info.varargs

                if name in frame.f_locals:
                    result.append(
                        f"*{name}="
                        f"{self._summarize(frame.f_locals[name])}"
                    )

            # ----------------------------------------------------------
            # **kwargs
            # ----------------------------------------------------------

            if arg_info.keywords is not None:
                name = arg_info.keywords

                if name in frame.f_locals:
                    result.append(
                        f"**{name}="
                        f"{self._summarize(frame.f_locals[name])}"
                    )

            return ", ".join(result)

        except Exception as e:
            return (
                "<failed to inspect arguments: "
                f"{type(e).__name__}: {e}>"
            )

    # ==================================================================
    # Output
    # ==================================================================

    def _write(self, text: str):
        if self._file is None:
            return

        with self._write_lock:
            self._file.write(text)
            self._file.write("\n")

            if self.flush_each_write:
                self._file.flush()

    # ==================================================================
    # Profile callback
    # ==================================================================

    def _profile(self, frame, event, arg):
        """
        sys.setprofile callback.

        我们只处理 Python 层面的：

            call
            return

        忽略：

            c_call
            c_return
            c_exception
        """

        if event not in ("call", "return"):
            return

        # 只输出 include_paths 中的函数
        if not self._should_trace(frame):
            return

        state = self._get_state()

        # ==============================================================
        # CALL
        # ==============================================================

        if event == "call":
            depth = state.depth

            # 无论是否因为 max_depth 而输出，
            # 都增加 logical traced depth。
            #
            # 这样 return 时才能正确恢复，
            # 避免原实现中 max_depth 造成 depth 不平衡。
            state.depth += 1

            if (
                self.max_depth is not None
                and depth >= self.max_depth
            ):
                return

            indent = "  " * depth

            # ----------------------------------------------------------
            # Current callee
            # ----------------------------------------------------------

            callee_name = self._frame_name(frame)

            callee_file = os.path.abspath(
                frame.f_code.co_filename
            )

            # 这里是函数定义的起始行
            callee_def_line = (
                frame.f_code.co_firstlineno
            )

            self._write(
                f"{indent}CALL {callee_name}"
            )

            self._write(
                f"{indent}  DEF: "
                f"{callee_file}:{callee_def_line}"
            )

            # ----------------------------------------------------------
            # Logical caller
            # ----------------------------------------------------------
            #
            # 注意：
            #
            # 不直接使用：
            #
            #     caller = frame.f_back
            #
            # 因为 nn.Module 调用经常是：
            #
            #     DeepseekV4Model.forward
            #         ↓
            #     torch.nn.Module.__call__
            #         ↓
            #     torch.nn.Module._call_impl
            #         ↓
            #     DeepseekV4DecoderLayer.forward
            #
            # 我们真正希望看到：
            #
            #     DeepseekV4Model.forward
            #         ↓
            #     DeepseekV4DecoderLayer.forward
            #
            # 所以从直接父 frame 开始一直向上找，
            # 直到找到 include_paths 中的 frame。
            # ----------------------------------------------------------

            caller, skipped = (
                self._find_included_caller(
                    frame.f_back
                )
            )

            if caller is not None:
                caller_name = self._frame_name(
                    caller
                )

                caller_file = os.path.abspath(
                    caller.f_code.co_filename
                )

                # 这是逻辑父 frame 当前执行到的位置，
                # 即调用当前函数附近的源码位置。
                caller_line = caller.f_lineno

                self._write(
                    f"{indent}  FROM: "
                    f"{caller_name} "
                    f"at {caller_file}:{caller_line}"
                )

                if (
                    self.show_skipped_frames
                    and skipped > 0
                ):
                    self._write(
                        f"{indent}        "
                        f"[skipped {skipped} "
                        f"non-included frame(s)]"
                    )

                # 调用位置对应源码
                source = self._get_source_line(
                    caller
                )

                if source:
                    self._write(
                        f"{indent}        "
                        f"{source}"
                    )

            else:
                self._write(
                    f"{indent}  FROM: "
                    f"<no caller in include_paths>"
                )

            # ----------------------------------------------------------
            # Arguments
            # ----------------------------------------------------------

            arguments = self._get_arguments(
                frame
            )

            if arguments:
                self._write(
                    f"{indent}  ARGS: "
                    f"{arguments}"
                )
            else:
                self._write(
                    f"{indent}  ARGS: <none>"
                )

        # ==============================================================
        # RETURN
        # ==============================================================

        elif event == "return":
            state.depth = max(
                state.depth - 1,
                0,
            )

            depth = state.depth

            if (
                self.max_depth is not None
                and depth >= self.max_depth
            ):
                return

            indent = "  " * depth

            func_name = self._frame_name(
                frame
            )

            self._write(
                f"{indent}RETURN {func_name}"
            )

            self._write(
                f"{indent}  RET: "
                f"{self._summarize(arg)}"
            )

    # ==================================================================
    # Context manager
    # ==================================================================

    def __enter__(self):
        output_path = Path(
            self.output_file
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._file = open(
            output_path,
            mode="w",
            encoding="utf-8",
            buffering=1024 * 1024,
        )

        state = self._get_state()
        state.depth = 0

        pid = os.getpid()
        tid = threading.get_ident()

        self._write("=" * 100)
        self._write(
            "PythonCallTracer started"
        )
        self._write(
            f"PID: {pid}"
        )
        self._write(
            f"TID: {tid}"
        )

        if self.include_paths:
            self._write(
                "Include paths:"
            )

            for path in self.include_paths:
                self._write(
                    f"  {path}"
                )

        if self.exclude_paths:
            self._write(
                "Exclude paths:"
            )

            for path in self.exclude_paths:
                self._write(
                    f"  {path}"
                )

        self._write("=" * 100)

        # --------------------------------------------------------------
        # 保存已有 profiler
        # --------------------------------------------------------------

        self._old_sys_profile = (
            sys.getprofile()
        )

        # --------------------------------------------------------------
        # 当前线程
        # --------------------------------------------------------------

        sys.setprofile(
            self._profile
        )

        # --------------------------------------------------------------
        # 后续通过 threading 创建的新线程
        # --------------------------------------------------------------

        if self.trace_threads:
            try:
                self._old_threading_profile = (
                    threading.getprofile()
                )
            except AttributeError:
                self._old_threading_profile = (
                    None
                )

            threading.setprofile(
                self._profile
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        # --------------------------------------------------------------
        # 必须首先关掉 profiler
        # --------------------------------------------------------------

        sys.setprofile(
            self._old_sys_profile
        )

        if self.trace_threads:
            threading.setprofile(
                self._old_threading_profile
            )

        # --------------------------------------------------------------
        # Finish log
        # --------------------------------------------------------------

        if self._file is not None:
            self._write("=" * 100)

            if exc_type is None:
                self._write(
                    "PythonCallTracer "
                    "finished normally"
                )
            else:
                self._write(
                    "PythonCallTracer "
                    "finished with exception:"
                )

                self._write(
                    f"{exc_type.__name__}: "
                    f"{exc_value}"
                )

            self._write("=" * 100)

            self._file.flush()
            self._file.close()
            self._file = None

        # 不吞掉异常
        return False