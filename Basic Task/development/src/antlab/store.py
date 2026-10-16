"""结果存储：运行编号、逐请求原始记录、汇总与输出文件。

目录结构
--------
results/runs/<run_id>/
    meta.json        环境、模型、服务配置、负载定义（可复现入口）
    requests.jsonl   逐请求原始记录（保留重试与错误明细）
    summary.json     汇总统计
    outputs/         每个规定任务的模型输出原文
results/service_logs/ llama-server 启动与运行日志（GPU 驻留证据）
results/gpu/         GPU 监控采样
results/index.json   所有运行索引
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from .config import RESULTS_DIR, RUNS_DIR, OUTPUTS_DIR, SERVICE_LOG_DIR, load_json

GPU_DIR = RESULTS_DIR / "gpu"
GPU_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = RESULTS_DIR / "index.json"


def _now() -> datetime:
    return datetime.now()


def new_run_id(tag: str) -> str:
    """运行编号：R<日期>-<序号>-<标签>，保证可排序且唯一。"""
    stamp = _now().strftime("%Y%m%d-%H%M%S")
    return f"R{stamp}-{tag}"


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False).stdout.strip()
    except Exception:
        return ""


def gpu_info() -> dict:
    """采集硬件信息（Windows 或 WSL 内均可）。"""
    info = {"hostname": socket.gethostname(), "platform": platform.platform()}
    smi = _sh(["nvidia-smi",
               "--query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu",
               "--format=csv,noheader"])
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        if len(parts) >= 5:
            info.update({
                "gpu_name": parts[0], "gpu_memory_total": parts[1],
                "gpu_memory_used": parts[2], "driver_version": parts[3],
                "gpu_temperature_c": parts[4],
            })
    uname = _sh(["uname", "-a"])
    if uname:
        info["uname"] = uname
    return info


def llama_version(bin_dir: str = "/root/llama.cpp/build/bin") -> dict:
    """llama.cpp 版本信息（来自 llama-server --version）。"""
    out = _sh([os.path.join(bin_dir, "llama-server"), "--version"], timeout=30)
    return {"raw": out}


def git_commit(path: str = "/root/llama.cpp") -> str:
    return _sh(["git", "-C", path, "rev-parse", "HEAD"])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_output(run_id: str, prompt_id: str, content: str, meta: dict | None = None) -> Path:
    """保存模型输出原文，便于输出效果检查。"""
    d = OUTPUTS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prompt_id}.md"
    header = ""
    if meta:
        header = (
            f"<!-- run_id={run_id} prompt_id={prompt_id} "
            f"prompt_tokens={meta.get('prompt_tokens')} "
            f"output_tokens={meta.get('output_tokens')} "
            f"ttft_ms={meta.get('ttft_ms')} -->\n\n"
        )
    path.write_text(header + content, encoding="utf-8")
    return path


def update_index(run_id: str, payload: dict) -> None:
    """维护 results/index.json，便于前端与报告统一索引。"""
    idx = load_json(INDEX_PATH) if INDEX_PATH.is_file() else {"runs": []}
    runs = [r for r in idx.get("runs", []) if r.get("run_id") != run_id]
    runs.append({"run_id": run_id, **payload})
    runs.sort(key=lambda r: r.get("run_id", ""))
    write_json(INDEX_PATH, {"updated_at": _now().isoformat(timespec="seconds"), "runs": runs})
