"""VLA 镜像环境冒烟：torch / MuJoCo EGL / Isaac Sim / Isaac Lab。"""

from __future__ import annotations

import importlib.metadata as md
import os
import subprocess
import sysconfig
import time
from pathlib import Path

import pytest

_ISAACLAB_ROOT = Path("/opt/IsaacLab")
_SPHERE_XML = """
<mujoco>
  <worldbody>
    <geom type="sphere" size="0.1"/>
  </worldbody>
</mujoco>
"""


def _dist_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _ensure_sim_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")


def test_torch_cu128_pin():
    torch = pytest.importorskip("torch")
    assert torch.__version__.startswith("2.11.0"), torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert "+cu128" in torch.__version__, torch.__version__


def test_mujoco_egl_render():
    mujoco = pytest.importorskip("mujoco")
    _ensure_sim_env()
    from mujoco import Renderer

    model = mujoco.MjModel.from_xml_string(_SPHERE_XML)
    data = mujoco.MjData(model)
    renderer = Renderer(model, 64, 64)
    try:
        renderer.update_scene(data)
        frame = renderer.render()
    finally:
        renderer.close()
    assert frame.shape == (64, 64, 3), frame.shape


def test_isaacsim_kit_layout():
    ver = _dist_version("isaacsim")
    if ver is None:
        pytest.skip("isaacsim not installed")
    assert ver.startswith("6.0.1"), ver
    kit = Path(sysconfig.get_paths()["purelib"]) / "isaacsim" / "kit"
    assert kit.is_dir(), kit


def test_isaacsim_import():
    if _dist_version("isaacsim") is None:
        pytest.skip("isaacsim not installed")
    _ensure_sim_env()
    import isaacsim  # noqa: F401

    assert isaacsim.SimulationApp is not None


def test_isaaclab_import_and_path():
    if _dist_version("isaaclab") is None:
        pytest.skip("isaaclab not installed")
    assert (_ISAACLAB_ROOT / "isaaclab.sh").is_file(), _ISAACLAB_ROOT
    import isaaclab

    assert isaaclab.__version__, isaaclab.__version__


@pytest.mark.skipif(
    os.environ.get("VLA_HEADLESS_SMOKE") != "1",
    reason="set VLA_HEADLESS_SMOKE=1 to run Kit headless smoke",
)
def test_isaaclab_create_empty_headless():
    # 看到 Setup complete 即通过；脚本本身会一直 step，需主动 kill。
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("create_empty headless needs GPU")
    if not (_ISAACLAB_ROOT / "isaaclab.sh").is_file():
        pytest.skip("IsaacLab not at /opt/IsaacLab")
    if _dist_version("isaacsim") is None:
        pytest.skip("isaacsim not installed")

    _ensure_sim_env()
    script = _ISAACLAB_ROOT / "scripts" / "tutorials" / "00_sim" / "create_empty.py"
    assert script.is_file(), script

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [str(_ISAACLAB_ROOT / "isaaclab.sh"), "-p", str(script)],
        cwd=str(_ISAACLAB_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    found = False
    lines: list[str] = []
    deadline = time.time() + 600
    try:
        assert proc.stdout is not None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            lines.append(line)
            if "[INFO]: Setup complete" in line:
                found = True
                break
    finally:
        proc.kill()
        proc.wait(timeout=60)
    tail = "".join(lines[-80:])
    assert found, f"create_empty headless missing Setup complete; exit={proc.returncode}\n{tail}"
