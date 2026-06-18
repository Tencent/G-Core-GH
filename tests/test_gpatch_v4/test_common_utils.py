from gpatch_v4.utils import common_utils


def test_safe_import_class_logs_original_traceback_on_import_failure(tmp_path, monkeypatch):
    package_dir = tmp_path / "bad_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "bad_module.py").write_text(
        "raise ImportError('original import boom')\n",
        encoding="utf-8",
    )

    messages = []
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        common_utils,
        "logging_with_rank_and_datetime",
        lambda message, *args, **kwargs: messages.append(message),
    )

    cls = common_utils.safe_import_class("bad_pkg.bad_module.BadClass")

    assert cls is None
    assert len(messages) == 1
    assert "Failed to import class 'bad_pkg.bad_module.BadClass'" in messages[0]
    assert "Traceback (most recent call last):" in messages[0]
    assert "ImportError: original import boom" in messages[0]
