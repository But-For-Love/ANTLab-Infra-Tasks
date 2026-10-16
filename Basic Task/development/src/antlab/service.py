"""llama-server 进程管理与 GPU 监控。

功能
----
- 按服务配置启动 llama-server（GPU 推理、关闭主机 RAM prompt cache）
- 健康检查、就绪确认、状态查询
- 异常检测、进程重启、恢复就绪确认
- 运行期间 GPU 采样（显存、利用率、温度、功耗），作为 GPU 驻留证据

设计说明
--------
llama-server 监听 127.0.0.1:<port>，仅本机访问。
`--cache-ram 0` 显式关闭主机内存 prompt cache，使基础阶段的缓存工作集驻留显存
（对应任务介绍第二节要求）。KV 相关参数由服务配置的 extra_args 给出。
"""
from __future__ import annotations

import csv
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import RESULTS_DIR, SERVICE_LOG_DIR, ServiceConfig
from .llm_client import LlamaClient

GPU_DIR = RESULTS_DIR / "gpu"
GPU_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BIN = os.environ.get("LLAMA_BIN_DIR", "/root/llama.cpp/build/bin")


def build_args(cfg: ServiceConfig, bin_dir: str = DEFAULT_BIN) -> list[str]:
    """把服务配置翻译成 llama-server 命令行。"""
    args = [
        os.path.join(bin_dir, "llama-server"),
        "-m", cfg.model_path,
        "--alias", "antlab-local",
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--jinja",
        "--metrics",
        "--slots",
        # 关闭主机 RAM prompt cache：基础阶段缓存工作集驻留显存
        "--cache-ram", "0",
    ]
    args.extend(cfg.extra_args)
    return args


@dataclass
class ServiceState:
    running: bool
    ready: bool
    pid: int | None
    health: dict | None


class ServiceManager:
    def __init__(self, cfg: ServiceConfig, bin_dir: str = DEFAULT_BIN):
        self.cfg = cfg
        self.bin_dir = bin_dir
        self.proc: subprocess.Popen | None = None
        self.log_path: Path = SERVICE_LOG_DIR / f"{cfg.config_id}.log"
        self._log_fh = None
        self.client = LlamaClient(cfg.base_url)

    # ---------- 生命周期 ----------
    def start(self, wait_ready_s: int = 180) -> bool:
        if self.is_alive():
            return self.is_ready()
        args = build_args(self.cfg, self.bin_dir)
        self._log_fh = self.log_path.open("a", encoding="utf-8")
        self._log_fh.write(
            f"\n\n########## 启动 {datetime.now().isoformat(timespec='seconds')} ##########\n"
            f"# 命令: {' '.join(args)}\n"
        )
        self._log_fh.flush()
        self.proc = subprocess.Popen(
            args, stdout=self._log_fh, stderr=subprocess.STDOUT,
            start_new_session=True, cwd="/root",
        )
        return self.wait_ready(wait_ready_s)

    def wait_ready(self, timeout_s: int = 180) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False  # 进程已退出
            if self.client.is_ready():
                return True
            time.sleep(1.0)
        return False

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def is_ready(self) -> bool:
        return self.client.is_ready()

    def state(self) -> ServiceState:
        return ServiceState(self.is_alive(), self.is_ready(),
                            self.proc.pid if self.proc else None, self.client.health())

    def stop(self, grace_s: float = 5.0) -> None:
        """正常停止（用于配置切换）。"""
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                self.proc.kill()
            self.proc.wait(timeout=grace_s)
        self.proc = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def kill(self) -> None:
        """强制杀死推理进程，用于模拟服务异常退出。"""
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:
            self.proc.kill()
        self.proc.wait(timeout=10)
        self.proc = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def restart(self, wait_ready_s: int = 180) -> bool:
        """重启并完成就绪确认。"""
        self.stop(grace_s=3.0)
        time.sleep(1.0)
        return self.start(wait_ready_s)


class GpuMonitor:
    """运行期间采样 GPU 指标，写入 results/gpu/<name>.csv。"""

    def __init__(self, name: str, interval_s: float = 1.0):
        self.path = GPU_DIR / f"{name}.csv"
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[list[str]] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,"
                     "temperature.gpu,power.draw,clocks.sm",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=False,
                ).stdout.strip()
                if out:
                    row = [c.strip() for c in out.splitlines()[0].split(",")]
                    self.samples.append(row)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        header = ["timestamp", "util_pct", "mem_used_mib", "mem_total_mib",
                  "temp_c", "power_w", "sm_clock_mhz"]
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(self.samples)

        def col(i: int) -> list[float]:
            out = []
            for r in self.samples:
                try:
                    out.append(float(r[i]))
                except (ValueError, IndexError):
                    pass
            return out

        mem, util, temp, power = col(2), col(1), col(4), col(5)
        return {
            "samples": len(self.samples),
            "csv_path": str(self.path),
            "mem_used_mib_max": max(mem) if mem else None,
            "mem_used_mib_min": min(mem) if mem else None,
            "util_pct_mean": (sum(util) / len(util)) if util else None,
            "temp_c_max": max(temp) if temp else None,
            "power_w_max": max(power) if power else None,
        }
