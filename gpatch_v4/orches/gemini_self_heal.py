"""Gemini GPU fault self-healing via :class:`GeminiNodeReplacer`.

Usage::

    training:
      enable_self_heal: true
      # optional, all fields have defaults:
      self_heal_config:
        recovery_max_wait: 1800        # max seconds to wait for pod replacement
        recovery_poll_interval: 15     # seconds between recovery-ready polls
        tag_recovery_timeout: 300      # timeout for tag_the_pods_to_recovery API
        ray_port: 6379                 # Ray head port for cluster rebuild

Gemini API references:
  - https://mmdcbff.woa.com/#/apidocsnew?app=gemini
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray
import requests
from typing_extensions import override

from gpatch_v4.orches.node_replacer import NodeReplacer
from gpatch_v4.utils import log

HOSTFILE = "/root/hostfile"


@dataclass
class PodInfo:
    pod_name: str
    pod_ip: str
    container: str
    is_launcher: bool


class GeminiApiError(Exception):
    def __init__(self, errcode: int, errmsg: str, endpoint: str = ""):
        self.errcode = errcode
        self.errmsg = errmsg
        self.endpoint = endpoint
        super().__init__(f"Gemini API {endpoint} errcode={errcode}: {errmsg}")


class SelfHealExitError(RuntimeError):
    """Launcher has a bad GPU and must exit for platform replacement."""


class GeminiClient:
    """Wrapper around Gemini platform APIs."""

    BASE = "http://weflowapi.woa.com"
    _RETRY_ATTEMPTS = 3
    _RETRY_BACKOFF = 2.0

    def __init__(self, task_instance_id: int, signature: str, app_name: str):
        self.task_instance_id = task_instance_id
        self.signature = signature
        self.app_name = app_name
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @classmethod
    def from_env(cls) -> GeminiClient:
        task_instance_id = int(os.environ["__SYS_TASK_RUNTIME_ID__"])
        signature = os.environ["__SYS_JOB_INSTANCE_SIGNATURE__"]
        app_name = os.environ["APP_NAME"]
        return cls(task_instance_id, signature, app_name)

    def get_abnormal_nodes(self) -> Dict[str, str]:
        """Returns ``{ip: reason}`` for faulty GPU nodes."""
        data = self._post(
            "/weflow/operation/gpu_fault_recovery/nodecheck/"
            "get_task_abnormal_nodes/",
            {"task_instance_id": self.task_instance_id},
        )
        return data if isinstance(data, dict) else {}

    def get_ips_pods(self, ips: List[str]) -> List[PodInfo]:
        """Map IPs to PodInfo."""
        if not ips:
            return []
        data = self._post(
            "/weflow/operation/suanli/get_ips_pods/",
            {"ips": ips},
        )
        result: List[PodInfo] = []
        for p in (data or {}).get("pods", []):
            pname = p["pod_name"]
            is_launcher = "launcher" in pname
            result.append(
                PodInfo(
                    pod_name=pname,
                    pod_ip=p.get("pod_ip") or p.get("node_ip", ""),
                    container="mpi-launcher" if is_launcher else "mpi-worker",
                    is_launcher=is_launcher,
                )
            )
        return result

    def exec_cmd(self, targets: List[PodInfo], cmd: str) -> Dict[str, Any]:
        if not targets:
            return {}
        data = self._post(
            "/weflow/job/runtime/k8s_app_pod_exec_cmd/",
            {
                "app_name": self.app_name,
                "cmd": cmd,
                "pod_containers":
                    [{
                        "pod_name": t.pod_name,
                        "container": t.container
                    } for t in targets],
                "signature": self.signature,
            },
        )
        return data or {}

    def tag_recovery(
        self,
        pod_names: List[str],
        timeout: int = 300,
        ip_list: Optional[List[str]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "task_instance_id": self.task_instance_id,
            "pod_names": pod_names,
            "force_replace": True,
            "timeout": timeout,
            "signature": self.signature,
        }
        if ip_list is not None:
            payload["ip_list"] = ip_list
        self._post(
            "/weflow/operation/runtime/tag_the_pods_to_recovery/",
            payload,
        )

    def check_recovery_ready(self) -> Dict[str, Any]:
        data = self._post(
            "/weflow/operation/runtime/check_task_pods_recovery_ready/",
            {
                "task_instance_id": self.task_instance_id,
                "signature": self.signature,
            },
        )
        return data if isinstance(data, dict) else {}

    def _post(self, path: str, payload: dict) -> Any:
        url = f"{self.BASE}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(1, self._RETRY_ATTEMPTS + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                last_err = exc
                if attempt < self._RETRY_ATTEMPTS:
                    time.sleep(self._RETRY_BACKOFF * attempt)
                    continue
                raise GeminiApiError(-1, str(exc), path) from exc

            errcode = body.get("errcode", -1)
            if errcode == 0:
                return body.get("data")
            # errcode=9999: check_recovery_ready 返回此码表示当前无活跃的换机流程
            if errcode == 9999 and "check_task_pods_recovery_ready" in path:
                raise GeminiApiError(9999, body.get("errmsg", ""), path)

            last_err = GeminiApiError(errcode, body.get("errmsg", ""), path)
            # errcode=3101: 换机频率限制（平台要求间隔 120s），等待后重试
            if errcode == 3101:
                log(f"[GeminiClient] Rate-limited (3101), waiting 130s ...")
                time.sleep(130)
                continue
            # errcode=1003: 缺少 signature，权限错误（不可重试）
            # errcode=9004: AC 鉴权失败，通常是误带 Wx-Ac-Ticket（不可重试）
            if attempt < self._RETRY_ATTEMPTS and errcode not in (1003, 9004):
                time.sleep(self._RETRY_BACKOFF * attempt)
                continue
            raise last_err

        raise last_err  # type: ignore[misc]


class GeminiNodeReplacer(NodeReplacer):
    """Detects faulty GPUs, replaces bad pods, reconnects Ray workers
    to the existing head.  Launcher's Ray daemon is never stopped.
    """
    def __init__(self, config=None):
        self.cfg = config.training.self_heal_config
        self.client = GeminiClient.from_env()
        self._my_ip = os.environ.get("__HOST_IP__", "")
        self._my_pod = os.environ.get("POD_NAME", "")
        self._all_pods: List[PodInfo] = []
        self._bad_pods: List[PodInfo] = []

    @override
    def evict_nodes(self, node_ips: List[str]) -> None:
        abnormal = self.client.get_abnormal_nodes()
        if not abnormal:
            log("[GeminiNodeReplacer] No abnormal GPU nodes detected")
            return

        log(f"[GeminiNodeReplacer] Abnormal nodes: {abnormal}")

        bad_ips = list(abnormal.keys())
        self._bad_pods = self.client.get_ips_pods(bad_ips)
        self._all_pods = _read_hostfile_pods()
        bad_pod_names = [p.pod_name for p in self._bad_pods]

        log(f"[GeminiNodeReplacer] Bad pods: {bad_pod_names}")

        if self._my_ip in abnormal:
            raise SelfHealExitError(
                f"Launcher {self._my_pod} has bad GPU "
                f"({abnormal[self._my_ip]}), exiting"
            )

        worker_pods = [p for p in self._all_pods if not p.is_launcher]
        if worker_pods:
            self.client.exec_cmd(worker_pods, "ray stop --force")

        self.client.tag_recovery(
            bad_pod_names,
            timeout=self.cfg.tag_recovery_timeout,
            ip_list=bad_ips,
        )

    @override
    def provision_nodes(self, count: int) -> List[str]:
        if not self._bad_pods:
            return []
        log("[GeminiNodeReplacer] Waiting for pod recovery ...")
        self._wait_recovery()
        return []

    @override
    def wait_cluster_ready(self, expected_nnodes: int, timeout: float = 300) -> bool:
        if not self._bad_pods:
            return True

        worker_pods = [p for p in _read_hostfile_pods() if not p.is_launcher]
        port = self.cfg.ray_port
        start_cmd = (
            f"ray stop --force 2>/dev/null; "
            f"ray start --address='{self._my_ip}:{port}'"
        )
        if worker_pods:
            self.client.exec_cmd(worker_pods, start_cmd)

        log(f"[GeminiNodeReplacer] Waiting for {expected_nnodes} Ray nodes ...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not ray.is_initialized():
                ray.init(address="auto", namespace="train", log_to_driver=False)
            alive = len([n for n in ray.nodes() if n.get("Alive")])
            if alive == expected_nnodes:
                log(f"[GeminiNodeReplacer] Cluster ready: {alive} nodes")
                return True
            time.sleep(5)
            log(
                f"[GeminiNodeReplacer] Current alive={alive}, expected={expected_nnodes}, waiting..."
            )

        ray.shutdown()
        return False

    def _wait_recovery(self) -> None:
        max_wait = self.cfg.recovery_max_wait
        interval = self.cfg.recovery_poll_interval
        elapsed = 0
        result: Dict[str, Any] = {}

        time.sleep(interval)
        elapsed += interval

        while elapsed < max_wait:
            try:
                result = self.client.check_recovery_ready()
            except GeminiApiError as e:
                if e.errcode == 9999:
                    time.sleep(interval)
                    elapsed += interval
                    continue
                raise

            if result.get("finished", False):
                log("[GeminiNodeReplacer] Pod recovery finished.")
                return

            not_ready = result.get("not_ready_pods", [])
            not_exist = result.get("not_exist_pods", [])
            if not_ready or not_exist:
                log(
                    f"[GeminiNodeReplacer] Recovery in progress — "
                    f"not_ready={len(not_ready)}, not_exist={len(not_exist)}, "
                    f"elapsed={elapsed}s/{max_wait}s"
                )

            time.sleep(interval)
            elapsed += interval

        raise RuntimeError(f"Pod recovery timed out after {max_wait}s. Last status: {result}")


def _read_hostfile_pods() -> List[PodInfo]:
    """Parse ``/root/hostfile`` into PodInfo list (IPs left empty)."""
    pods: List[PodInfo] = []
    if os.path.isfile(HOSTFILE):
        with open(HOSTFILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pod_name = line.split()[0]
                is_launcher = "launcher" in pod_name
                pods.append(
                    PodInfo(
                        pod_name=pod_name,
                        pod_ip="",
                        container="mpi-launcher" if is_launcher else "mpi-worker",
                        is_launcher=is_launcher,
                    )
                )
    if not pods:
        raise ValueError(f"Hostfile {HOSTFILE} not found or empty")
    return pods
