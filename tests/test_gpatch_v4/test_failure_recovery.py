"""Tests for failure recovery: crash detection, hang detection, and node replacement."""

import asyncio
import os
import re
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from gpatch_v4.orches.failure import FailureEvent, FailureType
from gpatch_v4.orches.node_replacer import MockNodeReplacer, NodeReplacer
from gpatch_v4.orches.resource_allocator import ResourceAllocation
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

try:
    import ray
    import ray.exceptions
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

LOG_RECORD_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL) - role="
)


def assert_log_records_not_interleaved(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return

    lines = path.read_text(errors="replace").splitlines()
    assert lines, f"{path} is empty"
    assert LOG_RECORD_RE.match(lines[0]), f"{path} starts with a non-record line: {lines[0]!r}"

    for line in lines[1:]:
        if LOG_RECORD_RE.match(line):
            continue
        assert "�" not in line, f"{path} contains replacement character in {line!r}"
        if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ", line):
            if re.match(r"^(Generating|Map|Filter|Saving|Flattening)\b", line):
                continue
            raise AssertionError(
                f"{path} has an embedded log record prefix in continuation line: {line!r}"
            )


class TestFailureEvent(unittest.TestCase):
    """Test FailureEvent creation and constraints."""
    def test_valid_hang_event(self):
        """验证可以正常构造 HANG 类型的 FailureEvent，且对象在布尔上下文中为 True
        （这样调用方可以用 `if event:` 判断"是否发生了故障"）。"""
        event = FailureEvent(
            failure_type=FailureType.HANG,
            failed_node_ips=["10.0.1.5"],
            failed_actor_names=["policy_0"],
            timestamp=time.time(),
        )
        assert event.failure_type == FailureType.HANG
        assert bool(event) is True

    def test_valid_crash_event(self):
        """验证可以构造 CRASH 类型、含 details 字段和多个节点/actor 的 FailureEvent，
        覆盖"actor 进程死亡 → RayActorError 上抛"场景下事件的常用字段。"""
        event = FailureEvent(
            failure_type=FailureType.CRASH,
            failed_node_ips=["10.0.1.5", "10.0.1.6"],
            failed_actor_names=["policy_0", "policy_1"],
            timestamp=time.time(),
            details="RayActorError: worker died",
        )
        assert event.failure_type == FailureType.CRASH
        assert bool(event) is True

    def test_falsy_none(self):
        """约定式测试：`check_liveness()` 返回 None 表示无故障，应在布尔上下文为 False。
        这里只是显式声明这一约定（与旧的 `bool` 返回值兼容），并未调用 FailureEvent 本身。"""
        result = None
        assert not result

    def test_repr(self):
        """验证 __repr__ 输出包含失败类型和节点 IP，便于日志 / 报错信息可读。"""
        event = FailureEvent(
            failure_type=FailureType.HANG,
            failed_node_ips=["10.0.1.5"],
            failed_actor_names=["policy_0"],
        )
        r = repr(event)
        assert "hang" in r
        assert "10.0.1.5" in r


class TestResourceAllocation(unittest.TestCase):
    """Test ResourceAllocation validation constraints."""
    def test_valid_allocation(self):
        """正常用例：每个角色的 GPU 数都能被 per-replica 最小 GPU 数整除，
        GBS 也能被 policy 的 DP-size 整除，validate() 应返回 True。"""
        alloc = ResourceAllocation(
            role_nnodes={
                "policy": 2,
                "sampler": 1,
                "gen_rm": 1
            },
            role_gpus_per_node={
                "policy": 8,
                "sampler": 8,
                "gen_rm": 4
            },
            role_min_gpus_per_replica={
                "policy": 4,
                "sampler": 2,
                "gen_rm": 2
            },
        )
        assert alloc.validate(train_gbs=32)

    def test_role_num_gpus(self):
        """role_num_gpus 返回 role 的 nnodes * gpus_per_node，缺失 key 时返回 0。"""
        alloc = ResourceAllocation(
            role_nnodes={
                "policy": 2,
                "sampler": 1
            },
            role_gpus_per_node={
                "policy": 8,
                "sampler": 4
            },
            role_min_gpus_per_replica={
                "policy": 4,
                "sampler": 2
            },
        )
        assert alloc.role_num_gpus("policy") == 16
        assert alloc.role_num_gpus("sampler") == 4
        assert alloc.role_num_gpus("missing") == 0

    def test_gpu_not_divisible_by_min(self):
        """约束 1：角色 GPU 数必须是 per-replica 最小 GPU 数的整数倍。
        policy 6 GPU、min=4 不能整除，应该抛 AssertionError。"""
        alloc = ResourceAllocation(
            role_nnodes={"policy": 1},
            role_gpus_per_node={"policy": 6},
            role_min_gpus_per_replica={"policy": 4},
        )
        with self.assertRaises(AssertionError):
            alloc.validate(train_gbs=32)

    def test_gbs_not_divisible_by_dp(self):
        """约束 2：训练 GBS 必须能被 policy 的 DP-size 整除。
        policy 8 GPU / min=4 → DP=2；GBS=5 不能被 2 整除，应抛 AssertionError。"""
        alloc = ResourceAllocation(
            role_nnodes={"policy": 1},
            role_gpus_per_node={"policy": 8},
            role_min_gpus_per_replica={"policy": 4},
        )
        with self.assertRaises(AssertionError):
            alloc.validate(train_gbs=5)

    def test_required_role_missing_min(self):
        """policy/sampler/gen_rm/bt_rm 必须显式声明 min_gpus_per_replica，
        缺失应抛 AssertionError。"""
        alloc = ResourceAllocation(
            role_nnodes={
                "policy": 1,
                "sampler": 1
            },
            role_gpus_per_node={
                "policy": 8,
                "sampler": 4
            },
            role_min_gpus_per_replica={"policy": 4},  # sampler 缺
        )
        with self.assertRaises(AssertionError):
            alloc.validate(train_gbs=8)

    def test_zero_role_skipped(self):
        """role_nnodes[role] == 0 时不校验该 role 的 min_gpus_per_replica，
        允许 allocation 记录空 role。"""
        alloc = ResourceAllocation(
            role_nnodes={
                "policy": 1,
                "sampler": 0
            },
            role_gpus_per_node={
                "policy": 8,
                "sampler": 0
            },
            role_min_gpus_per_replica={"policy": 4},  # sampler 虽然缺但 nnodes=0，跳过
        )
        assert alloc.validate(train_gbs=8)

    def test_kv_training_plt_default_min_one(self):
        """kv 和 training_plt 允许缺失 min_gpus_per_replica（默认 1），不报错。"""
        alloc = ResourceAllocation(
            role_nnodes={
                "policy": 1,
                "kv": 1,
                "training_plt": 1
            },
            role_gpus_per_node={
                "policy": 8,
                "kv": 1,
                "training_plt": 1
            },
            role_min_gpus_per_replica={"policy": 4},  # kv / training_plt 缺是合法的
        )
        assert alloc.validate(train_gbs=8)


class TestMockNodeReplacer(unittest.TestCase):
    """Test MockNodeReplacer basic behavior."""
    def test_evict_does_not_raise(self):
        """MockNodeReplacer.evict_nodes 只打日志、不真正驱逐节点，调用过程不应抛异常
        （契约测试，确保 mock 实现满足接口）。"""
        replacer = MockNodeReplacer()
        replacer.evict_nodes(["10.0.1.5"])

    def test_provision_returns_empty(self):
        """MockNodeReplacer.provision_nodes 不真正分配节点，约定返回空列表，
        告诉调用方"等待人工/外部补充节点"。"""
        replacer = MockNodeReplacer()
        result = replacer.provision_nodes(2)
        assert result == []


class TestLaunchWithRetryHang(unittest.IsolatedAsyncioTestCase):
    """Test launch_then_run_with_recovery handles hang (FailureEvent from check_liveness)."""
    async def test_hang_triggers_restart(self):
        """check_liveness 第一次返回 HANG 事件 → 触发一次重启；第二次返回 None → 训练正常结束。
        预期 launch 被调用 2 次（首次 + 1 次重试）。"""
        from gpatch_v4.configs.config import T2iRlConfig
        from gpatch_v4.trainer import T2iGrpoTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_t2i', T2iRlConfig)
        config.training.max_restart_attempts = 1

        trainer = T2iGrpoTrainer()

        hang_event = FailureEvent(
            failure_type=FailureType.HANG,
            failed_node_ips=["10.0.1.5"],
            failed_actor_names=["policy_0"],
            timestamp=time.time(),
        )

        with patch.object(trainer, 'launch') as mock_launch, \
            patch.object(trainer, 'train_group') as mock_train_group:

            mock_train_loop = AsyncMock()
            mock_train_group.train_loop = mock_train_loop

            # First call: hang detected; second call: training finishes normally
            mock_check_liveness = AsyncMock(side_effect=[hang_event, None])
            mock_train_group.check_liveness = mock_check_liveness

            await trainer.launch_then_run_with_recovery(config)

            assert mock_launch.call_count == 2

    async def test_no_failure_single_launch(self):
        """快乐路径：check_liveness 始终返回 None（无故障）→ launch 只被调用一次，
        且 launch_then_run_with_recovery 应原样返回 train_loop 的结果。"""
        from gpatch_v4.configs.config import T2iRlConfig
        from gpatch_v4.trainer import T2iGrpoTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_t2i', T2iRlConfig)
        config.training.max_restart_attempts = 3

        trainer = T2iGrpoTrainer()

        with patch.object(trainer, 'launch') as mock_launch, \
            patch.object(trainer, 'train_group') as mock_train_group:

            mock_train_loop = AsyncMock(return_value="done")
            mock_train_group.train_loop = mock_train_loop

            # check_liveness returns None = training finished normally
            mock_check_liveness = AsyncMock(return_value=None)
            mock_train_group.check_liveness = mock_check_liveness

            result = await trainer.launch_then_run_with_recovery(config)

            assert mock_launch.call_count == 1
            assert result == "done"

    async def test_max_restarts_exhausted(self):
        """check_liveness 永远返回 HANG → 重试次数耗尽（max_restart_attempts=1，即 1 次初始 + 1 次重试 = 2 次 launch）
        后应抛出 RuntimeError，避免无限重启。"""
        from gpatch_v4.configs.config import T2iRlConfig
        from gpatch_v4.trainer import T2iGrpoTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_t2i', T2iRlConfig)
        config.training.max_restart_attempts = 1

        trainer = T2iGrpoTrainer()

        hang_event = FailureEvent(
            failure_type=FailureType.HANG,
            failed_node_ips=["10.0.1.5"],
            failed_actor_names=["policy_0"],
            timestamp=time.time(),
        )

        with patch.object(trainer, 'launch') as mock_launch, \
            patch.object(trainer, 'train_group') as mock_train_group:

            mock_train_loop = AsyncMock()
            mock_train_group.train_loop = mock_train_loop

            # Always stuck -> will exhaust all retries
            mock_check_liveness = AsyncMock(return_value=hang_event)
            mock_train_group.check_liveness = mock_check_liveness

            with self.assertRaises(RuntimeError):
                await trainer.launch_then_run_with_recovery(config)

            assert mock_launch.call_count == 2  # initial + 1 retry


@unittest.skipUnless(_RAY_AVAILABLE, "ray not installed")
class TestLaunchWithRetryCrash(unittest.IsolatedAsyncioTestCase):
    """Test launch_then_run_with_recovery handles crash (RayActorError from train_loop)."""
    async def test_crash_triggers_restart(self):
        """另一种故障路径：train_loop 第一次抛出 RayActorError（actor 进程崩溃）→ 触发重启；
        第二次成功返回 "recovered"。验证 crash 与 hang 走同一套 recovery 逻辑、launch 调用 2 次。
        check_liveness 用一个永远 pending 的 future，模拟"是 train_loop 先完成（异常）"的竞态分支。"""
        from gpatch_v4.configs.config import T2iRlConfig
        from gpatch_v4.trainer import T2iGrpoTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_t2i', T2iRlConfig)
        config.training.max_restart_attempts = 1

        trainer = T2iGrpoTrainer()

        call_count = 0

        async def mock_train_loop_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ray.exceptions.RayActorError()
            return "recovered"

        async def mock_check_liveness_fn():
            # Never returns — waits forever until cancelled
            await asyncio.get_event_loop().create_future()

        with patch.object(trainer, 'launch') as mock_launch, \
            patch.object(trainer, 'train_group') as mock_train_group:

            mock_train_group.train_loop = mock_train_loop_fn
            mock_train_group.check_liveness = mock_check_liveness_fn

            result = await trainer.launch_then_run_with_recovery(config)

            assert mock_launch.call_count == 2
            assert result == "recovered"


class TestLaunchWithRetryNodeReplacement(unittest.IsolatedAsyncioTestCase):
    """Test that NodeReplacer is called during restart."""
    async def test_node_replacer_called_on_hang(self):
        """当 config 里配置了 node_replacer_cls 且检测到 hang（带有 failed_node_ips）时，
        recovery 流程必须依次调用 NodeReplacer 的 evict_nodes / provision_nodes / wait_cluster_ready，
        且参数与失败事件中的节点信息一致（驱逐 1 个节点 → 申请 1 个节点）。"""
        from gpatch_v4.configs.config import T2iRlConfig
        from gpatch_v4.trainer import T2iGrpoTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_t2i', T2iRlConfig)
        config.training.max_restart_attempts = 1
        config.training.node_replacer_cls = "gpatch_v4.orches.node_replacer.MockNodeReplacer"

        trainer = T2iGrpoTrainer()

        hang_event = FailureEvent(
            failure_type=FailureType.HANG,
            failed_node_ips=["10.0.1.5"],
            failed_actor_names=["policy_0"],
            timestamp=time.time(),
        )

        with patch.object(trainer, 'launch') as mock_launch, \
            patch.object(trainer, 'train_group') as mock_train_group, \
            patch('gpatch_v4.trainer.trainer_mixin._instantiate_node_replacer') as mock_inst:

            mock_replacer = MagicMock(spec=NodeReplacer)
            mock_replacer.wait_cluster_ready.return_value = True
            mock_inst.return_value = mock_replacer

            mock_train_loop = AsyncMock(return_value="done")
            mock_train_group.train_loop = mock_train_loop

            mock_check_liveness = AsyncMock(side_effect=[hang_event, None])
            mock_train_group.check_liveness = mock_check_liveness

            await trainer.launch_then_run_with_recovery(config)

            mock_replacer.evict_nodes.assert_called_once_with(["10.0.1.5"])
            mock_replacer.provision_nodes.assert_called_once_with(1)
            mock_replacer.wait_cluster_ready.assert_called_once()


class TestInstantiateNodeReplacer(unittest.TestCase):
    """Test _instantiate_node_replacer helper."""
    def test_none_returns_none(self):
        """配置为 None（即不启用节点替换）时，工厂返回 None，调用方据此跳过 replacer 流程。"""
        from gpatch_v4.trainer.trainer_mixin import _instantiate_node_replacer
        assert _instantiate_node_replacer(None) is None

    def test_valid_cls_path(self):
        """传入合法的全限定类路径，应成功 import 并实例化为对应的 NodeReplacer 子类。"""
        from gpatch_v4.trainer.trainer_mixin import _instantiate_node_replacer
        replacer = _instantiate_node_replacer("gpatch_v4.orches.node_replacer.MockNodeReplacer")
        assert isinstance(replacer, MockNodeReplacer)

    def test_invalid_cls_path_raises(self):
        """模块存在但类名不存在，应抛 ImportError 或 AttributeError，让用户尽早发现配置错误。"""
        from gpatch_v4.trainer.trainer_mixin import _instantiate_node_replacer
        with self.assertRaises((ImportError, AttributeError)):
            _instantiate_node_replacer("gpatch_v4.orches.node_replacer.NonExistentClass")

    def test_non_replacer_subclass_raises(self):
        """类型安全：传入的类必须是 NodeReplacer 的子类，否则抛 AssertionError。
        这里用 ResourceAllocation（非 NodeReplacer）来触发该检查。"""
        from gpatch_v4.trainer.trainer_mixin import _instantiate_node_replacer
        with self.assertRaises(AssertionError):
            _instantiate_node_replacer("gpatch_v4.orches.resource_allocator.ResourceAllocation")


# ------------------------------------------------------------------ #
#  Integration tests (require real Ray cluster + GPU)
# ------------------------------------------------------------------ #


def _count_calls(func):
    """Decorator that counts async function invocations."""
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return await func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


@unittest.skipUnless(_RAY_AVAILABLE, "ray not installed")
class TestLaunchRetryIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests that exercise the real launch + check_liveness path."""
    def setUp(self):
        import os
        import shutil
        if ray.is_initialized():
            kill_all_actors_and_shutdown_ray()
        if os.path.exists("test_launch_retry"):
            shutil.rmtree("test_launch_retry")

    def tearDown(self):
        import os
        import shutil
        kill_all_actors_and_shutdown_ray()
        if os.path.exists("test_launch_retry"):
            shutil.rmtree("test_launch_retry")

    async def test_check_no_hang(self):
        """端到端：用真实的 SFT trainer + Ray，给一个充裕的 step 超时（3600s）和 300s 的整体超时，
        预期训练正常完成、launch 只调用 1 次（没有触发任何 recovery）。
        这条用例验证"无故障路径"在真实集群上端到端可走通。"""
        from gpatch_v4.configs.config import FinetuneConfig
        from gpatch_v4.trainer import FinetuneTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_sft', FinetuneConfig)
        config.training.max_restart_attempts = 1
        config.training.max_train_step_waiting_time = 3600
        config.training.skip_train_step = False
        trainer = FinetuneTrainer()

        original_launch = trainer.launch
        decorated_launch = _count_calls(original_launch)
        trainer.launch = decorated_launch

        try:
            await asyncio.wait_for(trainer.launch_then_run_with_recovery(config), timeout=300)
            assert decorated_launch.call_count == 1, (
                f"Expected 1 call, but got {decorated_launch.call_count}"
            )
        except asyncio.TimeoutError:
            assert False, "train_loop exceeded 300 second timeout limit, maybe hanging"
        finally:
            trainer.launch = original_launch

    async def test_check_hang(self):
        """端到端：把 max_train_step_waiting_time 设为 0.001s，强制让 check_liveness 把任何真实
        step 都判为 hang，从而每次 launch 后都触发重试；max_restart_attempts=1 → 最终耗尽重试
        抛出 RuntimeError，且 launch 恰好被调用 2 次（首次 + 1 次重试）。
        这条用例端到端验证"hang 检测 + 重启 + 重试耗尽"链路。"""
        from gpatch_v4.configs.config import T2iRlConfig
        from gpatch_v4.trainer import T2iGrpoTrainer
        from gpatch_v4_test_helper import load_config

        config = load_config('test_launch_retry_t2i', T2iRlConfig)
        config.training.max_restart_attempts = 1
        config.training.max_train_step_waiting_time = 0.001
        config.training.skip_train_step = False
        trainer = T2iGrpoTrainer()

        original_launch = trainer.launch
        decorated_launch = _count_calls(original_launch)
        trainer.launch = decorated_launch

        try:
            with self.assertRaises(RuntimeError):
                await trainer.launch_then_run_with_recovery(config)
            assert decorated_launch.call_count == 2, (
                f"Expected 2 calls, but got {decorated_launch.call_count}"
            )
        finally:
            trainer.launch = original_launch

    async def test_actor_crash_stack_written_to_task_logs(self):
        """Crash one real training actor and verify task-wise logs capture the stack."""
        import shutil
        import tempfile

        from gpatch_v4.actor.finetune_actor import FinetuneActor
        from gpatch_v4.configs.config import FinetuneConfig
        from gpatch_v4.trainer import FinetuneTrainer
        from gpatch_v4_test_helper import load_config

        crash_token = "GPATCH_TEST_ACTOR_CRASH_TOKEN"
        debug_token = "GPATCH_TEST_DEBUG_BEFORE_CRASH_TOKEN"
        # Use cwd-based temp dir (on distributed filesystem) so that
        # Ray actors on any node can write to the same path.
        tmp_root = Path(tempfile.mkdtemp(
            prefix="gpatch_logging_crash_", dir=os.getcwd(),
        ))

        async def crash_train_loop(self):
            from gpatch_v4.utils import log_debug

            log_debug(f"{debug_token} rank={getattr(self, '_rank', 'unknown')}")
            raise RuntimeError(crash_token)

        try:
            config = load_config('test_launch_retry_sft', FinetuneConfig)
            config.report.log_dir = str(tmp_root / "logs")
            config.report.log_level = "debug"
            config.report.debug_log_to_file = True
            config.checkpoint.save_ckpt_path = str(tmp_root / "ckpt")
            config.training.max_restart_attempts = 0
            config.training.max_train_step_waiting_time = 3600
            config.training.skip_train_step = True

            trainer = FinetuneTrainer()
            with patch.object(FinetuneActor, "_train_loop", crash_train_loop):
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(
                        trainer.launch_then_run_with_recovery(config),
                        timeout=300,
                    )

            # Allow distributed filesystem to flush actor logs.
            await asyncio.sleep(2)

            log_dirs = sorted((tmp_root / "logs").glob("*"))
            assert len(log_dirs) == 1, f"expected exactly one task log dir, got {log_dirs}"
            log_dir = log_dirs[0]

            training_log = (log_dir / "training.log").read_text(errors="replace")

            debug_log_dir = log_dir / "debug_log"
            assert debug_log_dir.is_dir(), (
                f"expected debug_log/ subdir, got {sorted(log_dir.iterdir())}"
            )
            debug_shards = sorted(debug_log_dir.glob("debug_*.log"))
            assert len(debug_shards) >= 1, (
                f"expected >=1 debug shard, got {sorted(debug_log_dir.iterdir())}"
            )
            debug_log = "\n".join(
                f.read_text(errors="replace") for f in debug_shards
            )

            assert debug_token in training_log
            assert debug_token in debug_log
            assert crash_token in training_log
            assert "Traceback" in training_log

            assert_log_records_not_interleaved(log_dir / "training.log")
            for shard in debug_shards:
                assert_log_records_not_interleaved(shard)
        finally:
            kill_all_actors_and_shutdown_ray()
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
