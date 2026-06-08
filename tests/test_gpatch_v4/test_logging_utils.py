import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import gpatch_v4.utils.logging_utils as logging_utils
from gpatch_v4.orches.train_actor import get_actor_log_role
from gpatch_v4.utils.logging_utils import (
    AtomicRecordHandler,
    CAPTURE_INFER_ENGINE_LOG_ENV,
    DEBUG_LOG_TO_FILE_ENV,
    LOG_LEVEL_ENV,
    LOG_TO_DRIVER_ENV,
    TASK_LOG_DIR_ENV,
    TeeStream,
    configure_third_party_logging,
    derive_task_log_dir,
    get_debug_log_path,
    get_console_stream,
    get_infer_engine_log_path,
    install_engine_file_handler,
    log,
    log_debug,
    redirect_stdio_fds_to_file,
    redirect_stdio_to_level_logs,
    setup_gpatch_logging,
)

LOG_RECORD_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL) - role="
)


def test_actor_log_role_uses_normalized_class_name():
    assert get_actor_log_role("DistillStudentActor") == "distill_student_actor"
    assert get_actor_log_role("DistillTeacherActor") == "distill_teacher_actor"
    assert get_actor_log_role("OffPolicyDistillStudentActor") == (
        "off_policy_distill_student_actor"
    )
    assert get_actor_log_role("GrpoTrainActor") == "grpo_train_actor"


def reset_gpatch_logging_env():
    for key in (
        TASK_LOG_DIR_ENV,
        LOG_LEVEL_ENV,
        DEBUG_LOG_TO_FILE_ENV,
        CAPTURE_INFER_ENGINE_LOG_ENV,
        LOG_TO_DRIVER_ENV,
    ):
        os.environ.pop(key, None)
    logging_utils.stdio_redirected = False
    for name, logger_ref in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger_ref, logging.Logger):
            continue
        if not name.startswith("gpatch_v4"):
            continue
        close_handlers(logger_ref)
    close_handlers(logging.getLogger())


def close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if (
            getattr(handler, "gpatch_level_handler", None) is not None
            or getattr(handler, "gpatch_console", False)
        ):
            logger.removeHandler(handler)
            handler.close()


def make_config(
    tmp_path: Path,
    *,
    report_log_dir=None,
    log_level="info",
    debug_log_to_file=False,
    capture_infer_engine_log=True,
):
    return SimpleNamespace(
        report=SimpleNamespace(
            wandb_exp_name="logging-test",
            log_dir=None if report_log_dir is None else str(report_log_dir),
            log_level=log_level,
            debug_log_to_file=debug_log_to_file,
            capture_infer_engine_log=capture_infer_engine_log,
        ),
        checkpoint=SimpleNamespace(save_ckpt_path=str(tmp_path / "ckpt")),
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
        assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ", line), (
            f"{path} has an embedded log record prefix in continuation line: {line!r}"
        )


def read_file(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def read_debug_log(log_dir: Path, role: str, rank: int | None) -> str:
    return read_file(Path(get_debug_log_path(str(log_dir), role, rank)))


def assert_no_level_file_handlers(logger: logging.Logger) -> None:
    assert not any(isinstance(h, AtomicRecordHandler) for h in logger.handlers)


def assert_no_shared_training_or_debug_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        path = getattr(handler, "gpatch_file_path", "")
        assert not path.endswith("training.log")
        assert not path.endswith("debug.log")


def make_tee_stream(tmp_path: Path, stream_name: str = "stdout"):
    log_path = tmp_path / f"{stream_name}.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    stream = TeeStream(None, fd, stream_name, "train_main", 0)
    return stream, log_path, fd


def test_report_log_dir_overrides_checkpoint_log_dir(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, report_log_dir=tmp_path / "custom_logs")

    log_dir = Path(derive_task_log_dir(config, force_new=True))
    assert log_dir.parent == tmp_path / "custom_logs"
    assert log_dir.name.startswith("logging-test_")

    logger = setup_gpatch_logging(
        config, log_dir=str(log_dir), role="train-main", rank=0, console_rank=None
    )
    logger.info("INFO_CUSTOM_LOG_DIR_TOKEN")
    logger.warning("WARN_CUSTOM_LOG_DIR_TOKEN")
    logger.error("ERR_CUSTOM_LOG_DIR_TOKEN")

    assert "INFO_CUSTOM_LOG_DIR_TOKEN" in read_file(log_dir / "training.log")
    assert "WARN_CUSTOM_LOG_DIR_TOKEN" in read_file(log_dir / "training.log")
    assert "ERR_CUSTOM_LOG_DIR_TOKEN" in read_file(log_dir / "training.log")
    assert not str(log_dir).startswith(str(tmp_path / "ckpt" / "logs"))

    assert_log_records_not_interleaved(log_dir / "training.log")


def test_log_dir_falls_back_to_checkpoint_when_report_log_dir_is_missing(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, report_log_dir=None)

    log_dir = Path(derive_task_log_dir(config, force_new=True))
    assert log_dir.parent == tmp_path / "ckpt" / "logs"
    assert log_dir.name.startswith("logging-test_")

    logger = setup_gpatch_logging(
        config, log_dir=str(log_dir), role="train-main", rank=0, console_rank=None
    )
    logger.info("INFO_FALLBACK_LOG_DIR_TOKEN")

    assert "INFO_FALLBACK_LOG_DIR_TOKEN" in read_file(log_dir / "training.log")
    assert_log_records_not_interleaved(log_dir / "training.log")


def test_log_debug_is_suppressed_when_log_level_is_info(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(config, log_dir=str(log_dir), role="policy", rank=0, console_rank=None)

    log("INFO_ONLY_TOKEN")
    log_debug("DEBUG_SUPPRESSED_TOKEN")

    training_log = read_file(log_dir / "training.log")
    assert "INFO_ONLY_TOKEN" in training_log
    assert "DEBUG_SUPPRESSED_TOKEN" not in training_log
    assert not (log_dir / "debug.log").exists()
    assert_log_records_not_interleaved(log_dir / "training.log")


def test_log_debug_goes_to_training_log_when_log_level_is_debug(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="debug", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(config, log_dir=str(log_dir), role="policy", rank=0, console_rank=None)

    log("INFO_DEBUG_MODE_TOKEN")
    log_debug("DEBUG_TRAINING_TOKEN")

    training_log = read_file(log_dir / "training.log")
    assert "INFO_DEBUG_MODE_TOKEN" in training_log
    assert "DEBUG_TRAINING_TOKEN" in training_log
    assert " - DEBUG - " in training_log
    assert not (log_dir / "debug.log").exists()
    assert_log_records_not_interleaved(log_dir / "training.log")


def test_log_debug_goes_only_to_debug_file_when_enabled_with_info_level(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=True)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(config, log_dir=str(log_dir), role="policy", rank=0, console_rank=None)

    log("INFO_DEBUG_FILE_TOKEN")
    log_debug("DEBUG_FILE_ONLY_TOKEN")

    training_log = read_file(log_dir / "training.log")
    debug_log = read_debug_log(log_dir, "policy", 0)
    assert "INFO_DEBUG_FILE_TOKEN" in training_log
    assert "DEBUG_FILE_ONLY_TOKEN" not in training_log
    assert "DEBUG_FILE_ONLY_TOKEN" in debug_log
    assert "INFO_DEBUG_FILE_TOKEN" not in debug_log
    assert " - DEBUG - " in debug_log
    assert not (log_dir / "debug.log").exists()
    assert_log_records_not_interleaved(log_dir / "training.log")
    assert_log_records_not_interleaved(Path(get_debug_log_path(str(log_dir), "policy", 0)))


def test_log_debug_goes_to_training_and_debug_file_when_both_switches_are_enabled(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="debug", debug_log_to_file=True)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(config, log_dir=str(log_dir), role="policy", rank=0, console_rank=None)

    log_debug("DEBUG_BOTH_TARGETS_TOKEN")

    training_log = read_file(log_dir / "training.log")
    debug_log = read_debug_log(log_dir, "policy", 0)
    assert "DEBUG_BOTH_TARGETS_TOKEN" in training_log
    assert "DEBUG_BOTH_TARGETS_TOKEN" in debug_log
    assert not (log_dir / "debug.log").exists()
    assert_log_records_not_interleaved(log_dir / "training.log")
    assert_log_records_not_interleaved(Path(get_debug_log_path(str(log_dir), "policy", 0)))


def test_console_rank_accepts_multiple_ranks(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))

    first_logger = setup_gpatch_logging(
        config,
        log_dir=str(log_dir),
        role="console-first",
        rank=0,
        console_rank=(0, 3),
    )
    middle_logger = setup_gpatch_logging(
        config,
        log_dir=str(log_dir),
        role="console-middle",
        rank=1,
        console_rank=(0, 3),
    )
    last_logger = setup_gpatch_logging(
        config,
        log_dir=str(log_dir),
        role="console-last",
        rank=3,
        console_rank=(0, 3),
    )

    assert any(getattr(handler, "gpatch_console", False) for handler in first_logger.handlers)
    assert not any(getattr(handler, "gpatch_console", False) for handler in middle_logger.handlers)
    assert any(getattr(handler, "gpatch_console", False) for handler in last_logger.handlers)


def test_actor_log_to_driver_installs_console_without_file_handlers(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    logger = setup_gpatch_logging(
        config,
        log_dir=str(log_dir),
        role="policy",
        rank=2,
        console_rank=None,
        capture_stdio=True,
        install_root=True,
        log_to_driver=True,
    )
    root_logger = logging.getLogger()

    assert any(getattr(handler, "gpatch_console", False) for handler in logger.handlers)
    assert any(getattr(handler, "gpatch_console", False) for handler in root_logger.handlers)
    assert_no_level_file_handlers(logger)
    assert_no_level_file_handlers(root_logger)
    assert not (log_dir / "training.log").exists()
    assert not logging_utils.stdio_redirected


def test_actor_debug_log_to_file_writes_debug_shard_only(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=True)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(
        config,
        log_dir=str(log_dir),
        role="policy",
        rank=3,
        console_rank=None,
        log_to_driver=True,
    )

    log("ACTOR_INFO_TOKEN")
    log_debug("ACTOR_DEBUG_SHARD_TOKEN")

    debug_log = read_debug_log(log_dir, "policy", 3)
    assert "ACTOR_DEBUG_SHARD_TOKEN" in debug_log
    assert "ACTOR_INFO_TOKEN" not in debug_log
    assert not (log_dir / "training.log").exists()
    assert not (log_dir / "debug.log").exists()


def test_actor_debug_log_to_driver_console_respects_log_level(tmp_path, monkeypatch):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="debug", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    console_path = tmp_path / "console.log"
    with console_path.open("w", encoding="utf-8") as console:
        monkeypatch.setattr(sys, "stderr", console)
        setup_gpatch_logging(
            config,
            log_dir=str(log_dir),
            role="policy",
            rank=4,
            console_rank=None,
            log_to_driver=True,
        )
        log_debug("ACTOR_DEBUG_CONSOLE_TOKEN")

    console_text = read_file(console_path)
    assert "ACTOR_DEBUG_CONSOLE_TOKEN" in console_text
    assert " - DEBUG - " in console_text


def test_log_to_driver_console_uses_current_stderr_for_wandb_capture(tmp_path, monkeypatch):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    console_path = tmp_path / "wandb-captured-console.log"

    with console_path.open("w", encoding="utf-8") as console:
        monkeypatch.setattr(sys, "stderr", console)
        logger = setup_gpatch_logging(
            config,
            log_dir=str(log_dir),
            role="policy",
            rank=0,
            console_rank=None,
            log_to_driver=True,
        )
        logger.info("WANDB_CAPTURED_CONSOLE_TOKEN")

    console_text = read_file(console_path)
    assert "WANDB_CAPTURED_CONSOLE_TOKEN" in console_text


def test_log_to_driver_console_avoids_driver_tee_stream(tmp_path, monkeypatch):
    reset_gpatch_logging_env()
    tee_stream, _, fd = make_tee_stream(tmp_path, "stderr")
    console_path = tmp_path / "safe-console.log"

    try:
        with console_path.open("w", encoding="utf-8") as console:
            monkeypatch.setattr(sys, "stderr", tee_stream)
            monkeypatch.setattr(sys, "__stderr__", console)

            assert get_console_stream(use_current_stderr=True) is console
    finally:
        os.close(fd)


def test_install_root_captures_third_party_logger_without_duplicate_gpatch_logs(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    third_party_logger = logging.getLogger("mbridge.test_logging_utils")
    original_level = third_party_logger.level
    original_propagate = third_party_logger.propagate
    third_party_logger.setLevel(logging.INFO)
    third_party_logger.propagate = True
    try:
        logger = setup_gpatch_logging(
            config,
            log_dir=str(log_dir),
            role="policy",
            rank=0,
            console_rank=None,
            install_root=True,
        )
        third_party_logger.info("MBRIDGE_FIRST_TOKEN")
        logger.info("GPATCH_FIRST_TOKEN")

        logger = setup_gpatch_logging(
            config,
            log_dir=str(log_dir),
            role="policy",
            rank=0,
            console_rank=None,
            install_root=True,
        )
        third_party_logger.info("MBRIDGE_SECOND_TOKEN")
        logger.info("GPATCH_SECOND_TOKEN")
    finally:
        third_party_logger.setLevel(original_level)
        third_party_logger.propagate = original_propagate

    training_log = read_file(log_dir / "training.log")
    assert training_log.count("MBRIDGE_FIRST_TOKEN") == 1
    assert training_log.count("GPATCH_FIRST_TOKEN") == 1
    assert training_log.count("MBRIDGE_SECOND_TOKEN") == 1
    assert training_log.count("GPATCH_SECOND_TOKEN") == 1
    assert_log_records_not_interleaved(log_dir / "training.log")


def test_infer_engine_log_path_is_sharded_by_backend_role_engine_and_rank(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path)
    log_dir = Path(derive_task_log_dir(config, force_new=True))

    path = Path(
        get_infer_engine_log_path(
            str(log_dir), backend="sglang", role="gen-rm", engine_idx=3, rank=7
        )
    )

    assert path.parent == log_dir / "infer_engine_log"
    assert path.name == "sglang_gen_rm_engine3_rank7.log"


def test_infer_engine_log_path_includes_gen_rm_index_when_present(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path)
    log_dir = Path(derive_task_log_dir(config, force_new=True))

    path = Path(
        get_infer_engine_log_path(
            str(log_dir),
            backend="sglang",
            role="gen-rm",
            rm_idx=3,
            engine_idx=0,
            rank=0,
        )
    )

    assert path.parent == log_dir / "infer_engine_log"
    assert path.name == "sglang_gen_rm3_engine0_rank0.log"


def test_configure_third_party_logging_respects_capture_infer_engine_log_switch(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, capture_infer_engine_log=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(config, log_dir=str(log_dir), role="policy", rank=0, console_rank=None)

    infer_engine_log = configure_third_party_logging(
        role="sampler", backend="vllm", rank=2, engine_idx=1, log_dir=str(log_dir)
    )

    assert infer_engine_log is None
    assert os.environ["GPATCH_ENGINE_LOG_FILE"] == ""


def test_configure_third_party_logging_uses_gen_rm_index_for_engine_log(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    setup_gpatch_logging(config, log_dir=str(log_dir), role="policy", rank=0, console_rank=None)

    infer_engine_log = configure_third_party_logging(
        role="gen-rm",
        backend="sglang",
        rank=0,
        engine_idx=0,
        rm_idx=2,
        log_dir=str(log_dir),
    )

    expected = str(log_dir / "infer_engine_log" / "sglang_gen_rm2_engine0_rank0.log")
    assert infer_engine_log == expected
    assert os.environ["GPATCH_ENGINE_LOG_FILE"] == expected


def test_configure_third_party_logging_actor_mode_keeps_console_and_debug_shard(
    tmp_path,
    monkeypatch,
):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=True)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    console_path = tmp_path / "third_party_console.log"
    with console_path.open("w", encoding="utf-8") as console:
        monkeypatch.setattr(sys, "stderr", console)
        logger = setup_gpatch_logging(
            config,
            log_dir=str(log_dir),
            role="sampler",
            rank=5,
            console_rank=None,
            install_root=True,
            log_to_driver=True,
        )
        close_handlers(logger)
        infer_engine_log = configure_third_party_logging(
            role="sampler",
            backend="vllm",
            rank=5,
            engine_idx=2,
            log_dir=str(log_dir),
        )
        logger.info("RESTORED_CONSOLE_TOKEN")
        log_debug("RESTORED_DEBUG_SHARD_TOKEN")

    root_logger = logging.getLogger()
    assert infer_engine_log == str(log_dir / "infer_engine_log" / "vllm_sampler_engine2_rank5.log")
    assert any(getattr(handler, "gpatch_console", False) for handler in logger.handlers)
    assert_no_shared_training_or_debug_handlers(root_logger)
    assert_no_shared_training_or_debug_handlers(logger)
    assert "RESTORED_CONSOLE_TOKEN" in read_file(console_path)
    debug_log = read_debug_log(log_dir, "vllm_sampler", 5)
    assert "RESTORED_DEBUG_SHARD_TOKEN" in debug_log
    assert not (log_dir / "training.log").exists()
    assert not (log_dir / "debug.log").exists()


def test_install_engine_file_handler_replaces_previous_engine_shard(tmp_path):
    reset_gpatch_logging_env()
    logger = logging.getLogger("vllm")
    original_level = logger.level
    original_propagate = logger.propagate
    first_path = tmp_path / "infer_engine_log" / "vllm_sampler_engine0_rank0.log"
    second_path = tmp_path / "infer_engine_log" / "vllm_sampler_engine1_rank0.log"
    try:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        install_engine_file_handler("vllm", str(first_path))
        install_engine_file_handler("vllm", str(second_path))
        logger.info("VLLM_SECOND_ENGINE_TOKEN")
        for handler in logger.handlers:
            handler.flush()
    finally:
        for handler in list(logger.handlers):
            if getattr(handler, "_gpatch_engine_log", None) is not None:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    assert "VLLM_SECOND_ENGINE_TOKEN" not in read_file(first_path)
    assert "VLLM_SECOND_ENGINE_TOKEN" in read_file(second_path)


def test_driver_tee_stream_captures_ray_prefixed_gpatch_record(tmp_path):
    reset_gpatch_logging_env()
    stream, log_path, fd = make_tee_stream(tmp_path, "stdout")
    try:
        stream.write(
            "(FinetuneActor pid=123, ip=1.2.3.4) "
            "2026-05-14 19:00:00,123 - INFO - role=finetune - "
            "rank=7 - pid=999 - node=1.2.3.4 - GPATCH_RAY_TOKEN\n"
        )
        stream.flush()
    finally:
        os.close(fd)

    content = read_file(log_path)
    assert "GPATCH_RAY_TOKEN" in content
    assert "(FinetuneActor pid=123" not in content
    assert content.startswith("2026-05-14 19:00:00,123 - INFO - role=finetune")


def test_driver_tee_stream_wraps_ray_prefixed_plain_output(tmp_path):
    reset_gpatch_logging_env()
    stream, log_path, fd = make_tee_stream(tmp_path, "stdout")
    try:
        stream.write("(FinetuneActor pid=123, ip=1.2.3.4) MBRIDGE_PRINT_TOKEN\n")
        stream.flush()
    finally:
        os.close(fd)

    content = read_file(log_path)
    assert " - INFO - role=ray_actor - rank=none - " in content
    assert "stream=stdout" in content
    assert "(FinetuneActor pid=123, ip=1.2.3.4) MBRIDGE_PRINT_TOKEN" in content
    assert_log_records_not_interleaved(log_path)


def test_driver_tee_stream_appends_after_existing_training_log(tmp_path):
    reset_gpatch_logging_env()
    config = make_config(tmp_path, log_level="info", debug_log_to_file=False)
    log_dir = Path(derive_task_log_dir(config, force_new=True))
    logger = setup_gpatch_logging(
        config,
        log_dir=str(log_dir),
        role="train_main",
        rank=0,
        console_rank=None,
        capture_stdio=False,
    )
    logger.info("EXISTING_TRAINING_LOG_TOKEN")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        redirect_stdio_to_level_logs(
            str(log_dir),
            role="train_main",
            rank=0,
            tee_to_console=False,
        )
        sys.stdout.write("(Actor pid=1, ip=1.1.1.1) RAY_AFTER_EXISTING_TOKEN\n")
        sys.stdout.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logging_utils.stdio_redirected = False

    content = read_file(log_dir / "training.log")
    assert "EXISTING_TRAINING_LOG_TOKEN" in content
    assert "RAY_AFTER_EXISTING_TOKEN" in content
    assert content.index("EXISTING_TRAINING_LOG_TOKEN") < content.index(
        "RAY_AFTER_EXISTING_TOKEN"
    )


def test_driver_tee_stream_ignores_bare_structured_stdout(tmp_path):
    reset_gpatch_logging_env()
    stream, log_path, fd = make_tee_stream(tmp_path, "stdout")
    try:
        stream.write(
            "2026-05-14 19:00:00,123 - INFO - role=train_main - "
            "rank=0 - pid=1 - node=driver - DRIVER_DUP_TOKEN\n"
        )
        stream.flush()
    finally:
        os.close(fd)

    assert read_file(log_path) == ""


def test_driver_tee_stream_keeps_stderr_error_capture(tmp_path):
    reset_gpatch_logging_env()
    stream, log_path, fd = make_tee_stream(tmp_path, "stderr")
    try:
        stream.write("RuntimeError: DRIVER_STDERR_TOKEN\n")
        stream.flush()
    finally:
        os.close(fd)

    content = read_file(log_path)
    assert "DRIVER_STDERR_TOKEN" in content
    assert " - ERROR - role=train_main - " in content


def test_redirect_stdio_fds_to_file_writes_to_target_file(tmp_path):
    reset_gpatch_logging_env()
    log_path = tmp_path / "infer_engine_log" / "sglang_sampler_engine0_rank0.log"

    with redirect_stdio_fds_to_file(str(log_path)):
        os.write(1, b"STDOUT_ENGINE_TOKEN\n")
        os.write(2, b"STDERR_ENGINE_TOKEN\n")

    content = read_file(log_path)
    assert "STDOUT_ENGINE_TOKEN" in content
    assert "STDERR_ENGINE_TOKEN" in content
