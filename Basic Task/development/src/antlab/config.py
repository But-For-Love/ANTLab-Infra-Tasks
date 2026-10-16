"""路径与配置加载。

所有路径相对 `Basic Task/development/` 解析，便于在不同机器上复现。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _find_development_root() -> Path:
    """从当前文件向上定位 development 目录。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "development" and (parent / "configs").is_dir():
            return parent
    # 回退：src/antlab/config.py -> development
    return here.parents[2]


DEV_ROOT = _find_development_root()
BASIC_ROOT = DEV_ROOT.parent

SRC_DIR = DEV_ROOT / "src"
CONFIGS_DIR = DEV_ROOT / "configs"
RESULTS_DIR = DEV_ROOT / "results"
REPORTS_DIR = DEV_ROOT / "reports"
TESTS_DIR = DEV_ROOT / "tests"

# 规定输入
PROMPTS_MD = BASIC_ROOT / "提示词与输入文档" / "测试提示词.md"
INPUT_DOCS_DIR = BASIC_ROOT / "提示词与输入文档" / "输入文档"

RUNS_DIR = RESULTS_DIR / "runs"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
SERVICE_LOG_DIR = RESULTS_DIR / "service_logs"

for _d in (RUNS_DIR, OUTPUTS_DIR, SERVICE_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class LoadConfig:
    """一次测试运行的配置。"""

    load_id: str
    description: str
    prompt_ids: list[str]
    concurrency: int
    repeats: int
    output_limit: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    arrival_mode: str  # simultaneous | fixed_interval
    arrival_interval_ms: int
    request_timeout_s: int
    max_retries: int
    enable_thinking: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "LoadConfig":
        return cls(
            load_id=d["load_id"],
            description=d.get("description", ""),
            prompt_ids=list(d["prompt_ids"]),
            concurrency=int(d.get("concurrency", 1)),
            repeats=int(d.get("repeats", 1)),
            output_limit=int(d.get("output_limit", 512)),
            temperature=float(d.get("temperature", 0.7)),
            top_p=float(d.get("top_p", 0.8)),
            top_k=int(d.get("top_k", 20)),
            seed=int(d.get("seed", 42)),
            arrival_mode=d.get("arrival_mode", "simultaneous"),
            arrival_interval_ms=int(d.get("arrival_interval_ms", 0)),
            request_timeout_s=int(d.get("request_timeout_s", 600)),
            max_retries=int(d.get("max_retries", 1)),
            enable_thinking=bool(d.get("enable_thinking", False)),
        )


@dataclass
class ServiceConfig:
    """llama-server 启动配置。"""

    config_id: str
    description: str
    host: str
    port: int
    model_path: str
    extra_args: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "ServiceConfig":
        return cls(
            config_id=d["config_id"],
            description=d.get("description", ""),
            host=d.get("host", "127.0.0.1"),
            port=int(d.get("port", 8080)),
            model_path=os.path.expanduser(d["model_path"]),
            extra_args=list(d.get("extra_args", [])),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_service_config(config_id: str) -> ServiceConfig:
    return ServiceConfig.from_dict(load_json(CONFIGS_DIR / "service" / f"{config_id}.json"))


def load_service_configs() -> list[ServiceConfig]:
    d = CONFIGS_DIR / "service"
    return [ServiceConfig.from_dict(load_json(p)) for p in sorted(d.glob("*.json"))]


def load_load_config(load_id: str) -> LoadConfig:
    return LoadConfig.from_dict(load_json(CONFIGS_DIR / "loads" / f"{load_id}.json"))


def load_load_configs() -> list[LoadConfig]:
    d = CONFIGS_DIR / "loads"
    return [LoadConfig.from_dict(load_json(p)) for p in sorted(d.glob("*.json"))]
