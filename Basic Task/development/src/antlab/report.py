"""从原始运行记录生成报告数据表格（Markdown）。

用途
----
报告中的每张表都由本模块从 `results/runs/*/summary.json` 与
`results/runs/*/requests.jsonl` 重新计算得到，保证报告数字与原始记录一致、
可复核。生成结果写入 `results/report_tables.md`。

运行： python3 run.py tables
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import RESULTS_DIR, RUNS_DIR, load_service_configs
from .store import read_jsonl


def _f(v, d: int = 1, dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return dash


def load_runs() -> list[dict]:
    """读取所有运行的汇总与逐请求记录。"""
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        sp = d / "summary.json"
        if not sp.is_file():
            continue
        try:
            s = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        s["_dir"] = str(d.relative_to(RESULTS_DIR.parent))
        s["_requests"] = read_jsonl(d / "requests.jsonl")
        runs.append(s)
    return runs


# ---------------------------------------------------------------- 表格
def table_kv_capacity() -> str:
    """KV 容量口径表：配置总容量 / 每 slot 可用 / KV 精度 / 显存。

    数据来自各服务配置以 -lv 5 启动后的日志（results/kv_capacity.json），
    并给出按 GGUF 架构参数独立推算的 KV 大小用于交叉验证。"""
    p = RESULTS_DIR / "kv_capacity.json"
    if not p.is_file():
        return "_（缺少 results/kv_capacity.json，请先运行 `python3 run.py kvlog`）_\n"
    rows = json.loads(p.read_text(encoding="utf-8"))
    out = [
        "| 配置 | 服务参数 | KV 类型 K/V | slot 数 | KV 分配 | 每 slot 可用容量（token） | "
        "配置总容量（token） | KV 日志实测（MiB） | 按架构推算（MiB） | 模型权重（MiB） | "
        "计算缓冲（MiB） | GPU 显存占用（MiB） |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        kvt = f"{r.get('kv_type_k', '—')} / {r.get('kv_type_v', '—')}"
        calc = (r.get("kv_mib_computed_f16") if r.get("kv_type_k") == "f16"
                else r.get("kv_mib_computed_q8_0"))
        out.append(
            f"| {r.get('config_id')} | `{r.get('params', '')}` | {kvt} | "
            f"{r.get('n_slots', '—')} | {r.get('kv_seqs', '—')} | "
            f"{r.get('n_ctx_slot', '—')} | {r.get('total_tokens', '—')} | "
            f"{_f(r.get('kv_total_mib'), 1)} | {_f(calc, 1)} | "
            f"{_f(r.get('cuda_model_mib'), 1)} | {_f(r.get('cuda_compute_mib'), 1)} | "
            f"{_f(r.get('gpu_used_mib'), 0)} |"
        )
    return "\n".join(out) + "\n"


def table_capacity() -> str:
    """容量边界验证表：逐步加长输入，记录实测可用容量与失败位置。"""
    files = sorted(RESULTS_DIR.glob("capacity_*.json"))
    if not files:
        return "_（缺少 results/capacity_*.json，请先运行 `python3 run.py capacity <配置>`）_\n"
    out = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append(f"**{d['config_id']}**：`{d['params']}` → "
                   f"每 slot 配置 {d['n_ctx_slot']} token，"
                   f"**实测最大可用输入 {d['max_ok_input_tokens']} token**")
        out.append("")
        out.append("| 目标输入（token） | 实测输入（token） | 结果 | 错误 |")
        out.append("| ---: | ---: | --- | --- |")
        for r in d["probes"]:
            err = (r.get("error_message") or "").replace("|", "/")
            err = err[:120] if err else ""
            out.append(f"| {r['target_tokens']} | {r.get('measured_input_tokens') or '—'} | "
                       f"{'成功' if r['ok'] else '**失败**'} | {err} |")
        out.append("")
    return "\n".join(out)


def table_env() -> str:
    """环境与模型信息表。"""
    p = RESULTS_DIR / "env.json"
    if not p.is_file():
        return "_（缺少 results/env.json，请先运行 `python3 run.py env`）_\n"
    e = json.loads(p.read_text(encoding="utf-8"))
    g = e.get("gpu") or {}
    rows = [
        ("采集时间", e.get("collected_at")),
        ("Windows/WSL", e.get("uname")),
        ("Python", e.get("python")),
        ("GPU", g.get("gpu_name")),
        ("GPU 显存总量", g.get("gpu_memory_total")),
        ("驱动版本", g.get("driver_version")),
        ("CUDA (nvcc)", " / ".join((e.get("nvcc") or "").splitlines())[:80]),
        ("CMake", e.get("cmake")),
        ("GCC", e.get("gcc")),
        ("llama.cpp commit", e.get("llama_commit")),
        ("模型文件", e.get("model_path")),
        ("模型大小（字节）", e.get("model_size_bytes")),
        ("模型 SHA256", f"`{e.get('model_sha256')}`"),
    ]
    out = ["| 项目 | 值 |", "| --- | --- |"]
    for k, v in rows:
        out.append(f"| {k} | {v if v not in (None, '') else '—'} |")
    return "\n".join(out) + "\n"


def table_runs(by_config: bool = True) -> str:
    """运行结果表，按（负载，服务配置，并发）分组聚合多次运行。

    分组必须包含 load_id：性能负载（bench_*）与输出质量负载（quality）
    的输出上限不同，混在一起会得到无意义的平均值。
    """
    runs = [r for r in load_runs() if r.get("overall")]
    if not runs:
        return "_（暂无运行数据）_\n"

    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(
            (r.get("load_id"), r.get("service_config"), r.get("concurrency")), []
        ).append(r)

    out = [
        "| 负载 | 服务配置 | 并发 | 运行次数 | 成功/总请求 | TTFT 中位 (ms) | TTFT P95 (ms) | "
        "TPOT 服务端 (ms) | 完成耗时中位 (ms) | 完成耗时 P95 (ms) | 解码吞吐 (tok/s) | "
        "请求吞吐 (req/s) | 显存峰值 (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (load, cfg, conc) in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), k[2] or 0)):
        rs = groups[(load, cfg, conc)]

        def avg(fn):
            vals = [fn(r) for r in rs]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        ok = sum(r["overall"]["success"] for r in rs)
        tot = sum(r["overall"]["total_requests"] for r in rs)
        mem = avg(lambda r: (r.get("gpu") or {}).get("mem_used_mib_max"))
        out.append(
            f"| {load} | {cfg} | {conc} | {len(rs)} | {ok}/{tot} | "
            f"{_f(avg(lambda r: r['aggregate']['ttft_p50_ms']['mean']), 0)} | "
            f"{_f(avg(lambda r: r['aggregate']['ttft_p95_ms']['mean']), 0)} | "
            f"{_f(avg(lambda r: r['aggregate']['tpot_server_ms']['mean']), 1)} | "
            f"{_f(avg(lambda r: r['aggregate']['total_p50_ms']['mean']), 0)} | "
            f"{_f(avg(lambda r: r['aggregate']['total_p95_ms']['mean']), 0)} | "
            f"{_f(avg(lambda r: r['aggregate']['decode_tokens_per_s']['mean']), 1)} | "
            f"{_f(avg(lambda r: r['aggregate']['requests_per_s']['mean']), 3)} | "
            f"{_f(mem, 0)} |"
        )
    return "\n".join(out) + "\n"


def table_rounds() -> str:
    """每轮明细表，用于说明重复测试的波动。"""
    runs = [r for r in load_runs() if r.get("rounds")]
    if not runs:
        return "_（暂无多轮数据）_\n"
    out = [
        "| 运行编号 | 负载 | 服务配置 | 并发 | 输出上限 | 轮次 | 成功/总 | "
        "TTFT 中位 (ms) | 解码吞吐 (tok/s) | 墙钟 (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(runs, key=lambda x: x.get("run_id", "")):
        for rs in r["rounds"]:
            out.append(
                f"| {r['run_id']} | {r.get('load_id')} | {r.get('service_config')} | "
                f"{r.get('concurrency')} | {r.get('output_limit')} | {rs.get('round')} | "
                f"{rs['success']}/{rs['total_requests']} | "
                f"{_f(rs['ttft_ms']['median'], 0)} | "
                f"{_f(rs['throughput']['decode_tokens_per_s'], 1)} | "
                f"{_f(rs['wall_clock_s'], 1)} |"
            )
    return "\n".join(out) + "\n"


def table_status() -> str:
    """请求状态与重试统计表，说明计数口径。"""
    runs = [r for r in load_runs() if r.get("_requests")]
    if not runs:
        return "_（暂无逐请求数据）_\n"
    agg: dict[str, int] = {}
    retried = 0
    total = 0
    for r in runs:
        for rec in r["_requests"]:
            st = rec.get("status", "unknown")
            agg[st] = agg.get(st, 0) + 1
            total += 1
            if (rec.get("retry_count") or 0) > 0:
                retried += 1
    out = [
        "| 状态 | 请求数 | 说明 |",
        "| --- | ---: | --- |",
    ]
    desc = {
        "success": "正常完成并取得生成内容",
        "failed": "收到错误响应或连接失败",
        "timeout": "超过请求超时上限",
        "cancelled": "提交前检测到服务中断，未发送",
    }
    for st, n in sorted(agg.items()):
        out.append(f"| {st} | {n} | {desc.get(st, '')} |")
    out.append(f"| **合计（逻辑请求）** | {total} | 重试请求 {retried} 个，按逻辑请求计数一次 |")
    return "\n".join(out) + "\n"


def table_errors() -> str:
    """错误明细表。"""
    rows = []
    for r in load_runs():
        for rec in r.get("_requests", []):
            if rec.get("status") != "success":
                rows.append((r.get("run_id"), rec))
    if not rows:
        return "_（所有请求均成功）_\n"
    out = [
        "| 运行编号 | 任务 | 状态 | 错误类型 | 错误信息（截断） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run_id, rec in rows[:40]:
        msg = (rec.get("error_message") or "").replace("|", "/")[:160]
        out.append(f"| {run_id} | {rec.get('prompt_id')} | {rec.get('status')} | "
                   f"{rec.get('error_type') or '—'} | {msg} |")
    if len(rows) > 40:
        out.append(f"| … | 其余 {len(rows) - 40} 条 | | | |")
    return "\n".join(out) + "\n"


def table_recovery() -> str:
    """异常恢复时间线表。"""
    p = RESULTS_DIR / "runs"
    files = sorted(p.glob("*/recovery.json"))
    if not files:
        return "_（暂无恢复测试记录）_\n"
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    out = [
        f"运行编号：`{d['run_id']}`　恢复耗时：**{d['recovery_s']} s**　"
        f"（恢复前请求={'成功' if d['before_ok'] else '失败'}，"
        f"中断期间={'成功' if d['during_ok'] else '失败'}，"
        f"恢复后={'成功' if d['after_ok'] else '失败'}）",
        "",
        "| 时刻 | 事件 | 说明 |",
        "| --- | --- | --- |",
    ]
    for ev in d["timeline"]:
        detail = {k: v for k, v in ev.items() if k not in ("event", "at")}
        out.append(f"| {ev['at']} | {ev['event']} | "
                   f"`{json.dumps(detail, ensure_ascii=False)}` |")
    return "\n".join(out) + "\n"


def table_checks() -> str:
    """输出效果检查结果表（办公事实/条款引用、代码执行与边界）。"""
    files = sorted((RESULTS_DIR / "checks").glob("*.json"))
    if not files:
        return "_（缺少 results/checks/*.json，请先运行 `python3 run.py check <run_id>`）_\n"
    out = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        s = d.get("summary", {})
        out.append(f"运行编号：`{d['run_id']}`　"
                   f"**{s.get('passed', 0)}/{s.get('total_checks', 0)} 项通过**")
        out.append("")
        out.append("| 任务 | 通过/检查项 |")
        out.append("| --- | ---: |")
        for pid, b in sorted((s.get("by_prompt") or {}).items()):
            out.append(f"| {pid} | {b['passed']}/{b['total']} |")
        out.append("")
        fails = [r for r in d.get("checks", []) if not r["passed"]]
        if fails:
            out.append("未通过项：")
            out.append("")
            out.append("| 任务 | 检查项 | 说明 |")
            out.append("| --- | --- | --- |")
            for r in fails:
                out.append(f"| {r['prompt_id']} | {r['check']} | "
                           f"{(r.get('detail') or '').replace('|', '/')} |")
            out.append("")
        det = (d.get("c02_detail") or {}).get("cases")
        if det:
            ok_n = sum(1 for v in det.values() if v == "ok")
            out.append(f"C02 生成代码的边界用例执行结果：{ok_n}/{len(det)} 项无异常。")
            out.append("")
    return "\n".join(out)


def generate() -> Path:
    path = RESULTS_DIR / "report_tables.md"
    parts = [
        "# 报告数据表（由原始记录自动生成）",
        "",
        "生成来源：`results/runs/*/summary.json`、`results/runs/*/requests.jsonl`、"
        "`results/kv_capacity.json`、`results/capacity_*.json`、`results/env.json`。",
        "重新生成：`python3 run.py tables`",
        "",
        "## 表 0　环境与模型信息",
        "",
        table_env(),
        "## 表 1　KV 容量口径",
        "",
        table_kv_capacity(),
        "## 表 2　容量边界验证（实测可用容量）",
        "",
        table_capacity(),
        "## 表 3　基线与优化结果（多轮聚合）",
        "",
        table_runs(),
        "## 表 4　逐轮明细与波动",
        "",
        table_rounds(),
        "## 表 5　请求状态与计数口径",
        "",
        table_status(),
        "## 表 6　错误明细",
        "",
        table_errors(),
        "## 表 7　服务异常与恢复时间线",
        "",
        table_recovery(),
        "## 表 8　模型输出效果检查",
        "",
        table_checks(),
    ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
