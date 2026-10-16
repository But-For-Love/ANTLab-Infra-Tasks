"""指标计算：TTFT / TPOT / 吞吐 / 完成耗时 / P95。

统计口径（与报告第 5 节一致）
---------------------------
成功请求   : status == "success" 且 output_tokens 非空
TTFT       : 客户端提交 → 首个非空生成内容，含排队与传输（客户端可观察）
TPOT(服务端): predicted_ms / predicted_n，仅生成阶段，排除首 token 与排队
TPOT(客户端): (total_ms - ttft_ms) / (output_tokens - 1)，排除首 token
吞吐       : 生成 token 总数 / 服务端生成耗时之和；另给出成功请求数/墙钟时间
完成耗时   : 客户端提交 → 流结束，含排队、提示词处理、生成与传输
P95        : 最近秩法（nearest-rank），对小样本比插值法更稳健
"""
from __future__ import annotations

from math import ceil


def percentile(values: list[float], p: float) -> float | None:
    """最近秩法百分位数。p 取 0–100。"""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    rank = ceil(p / 100.0 * len(xs))
    idx = min(max(rank - 1, 0), len(xs) - 1)
    return xs[idx]


def _vals(records: list[dict], key: str) -> list[float]:
    return [r[key] for r in records if r.get(key) is not None]


def summarize(records: list[dict], wall_clock_s: float | None = None) -> dict:
    """按统一口径汇总一组逐请求记录。"""
    total = len(records)
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1

    ok = [r for r in records if r.get("status") == "success"]

    ttfts = _vals(ok, "ttft_ms")
    totals = _vals(ok, "total_ms")
    tpot_srv = _vals(ok, "tpot_server_ms")
    tpot_cli = _vals(ok, "tpot_client_ms")

    out_tokens = sum(int(r.get("output_tokens") or 0) for r in ok)
    gen_ms = sum(float(r.get("server_gen_ms") or 0.0) for r in ok)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in ok)
    cached_tokens = sum(int(r.get("cached_tokens") or 0) for r in ok)

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    summary = {
        "total_requests": total,
        "by_status": by_status,
        "success": len(ok),
        "success_rate": (len(ok) / total) if total else None,
        "retried_requests": sum(1 for r in records if (r.get("retry_count") or 0) > 0),
        "ttft_ms": {
            "mean": mean(ttfts),
            "median": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "min": min(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        "total_ms": {
            "mean": mean(totals),
            "median": percentile(totals, 50),
            "p95": percentile(totals, 95),
        },
        "tpot_server_ms": {"mean": mean(tpot_srv), "median": percentile(tpot_srv, 50)},
        "tpot_client_ms": {"mean": mean(tpot_cli), "median": percentile(tpot_cli, 50)},
        "tokens": {
            "prompt_total": prompt_tokens,
            "cached_total": cached_tokens,
            "output_total": out_tokens,
            "output_mean": (out_tokens / len(ok)) if ok else None,
        },
        "throughput": {
            # 生成 token 总数 / 服务端生成耗时之和（纯解码速率）
            "decode_tokens_per_s": (out_tokens / (gen_ms / 1000.0)) if gen_ms > 0 else None,
            # 成功请求数 / 墙钟时间（端到端，含排队与提示词处理）
            "requests_per_s": (len(ok) / wall_clock_s) if wall_clock_s else None,
            # 生成 token 总数 / 墙钟时间
            "wall_tokens_per_s": (out_tokens / wall_clock_s) if wall_clock_s else None,
        },
        "wall_clock_s": wall_clock_s,
    }
    return summary


def aggregate_rounds(round_summaries: list[dict]) -> dict:
    """把多轮同配置结果汇总，给出均值与波动（用于 3 轮重复测试）。"""
    def collect(path: list[str]) -> list[float]:
        out = []
        for s in round_summaries:
            cur = s
            for k in path:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    def stat(path: list[str]) -> dict:
        xs = collect(path)
        if not xs:
            return {"mean": None, "min": None, "max": None, "spread_pct": None}
        m = sum(xs) / len(xs)
        spread = ((max(xs) - min(xs)) / m * 100.0) if m else None
        return {"mean": m, "min": min(xs), "max": max(xs), "spread_pct": spread, "n_rounds": len(xs)}

    return {
        "rounds": len(round_summaries),
        "success_rate": stat(["success_rate"]),
        "ttft_p50_ms": stat(["ttft_ms", "median"]),
        "ttft_p95_ms": stat(["ttft_ms", "p95"]),
        "total_p50_ms": stat(["total_ms", "median"]),
        "total_p95_ms": stat(["total_ms", "p95"]),
        "tpot_server_ms": stat(["tpot_server_ms", "mean"]),
        "decode_tokens_per_s": stat(["throughput", "decode_tokens_per_s"]),
        "requests_per_s": stat(["throughput", "requests_per_s"]),
        "wall_clock_s": stat(["wall_clock_s"]),
    }
