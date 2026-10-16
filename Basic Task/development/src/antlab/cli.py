"""命令行入口（在 WSL 内运行）。

用法
----
python3 run.py tasks                     列出解析到的规定任务
python3 run.py serve   <service_cfg>     启动 llama-server 并等待就绪
python3 run.py stop    <service_cfg>     停止服务
python3 run.py status  <service_cfg>     查询服务状态与 KV/显存信息
python3 run.py bench   <service_cfg> <load_cfg>   执行固定负载测试
python3 run.py quality <service_cfg>     跑 10 个规定任务并保存输出原文
python3 run.py recover <service_cfg>     异常恢复测试（强杀→检测→重启→复跑）
python3 run.py matrix                    跑基线与优化的完整对比矩阵
"""
from __future__ import annotations

import json
import sys
import time

from .config import (DEV_ROOT, RESULTS_DIR, load_load_config, load_service_config,
                     load_service_configs, load_load_configs)
from .llm_client import GenParams, LlamaClient
from .prompts import SYSTEM_MESSAGE, load_tasks, task_manifest, sha256_text
from .runner import LoadRunner
from .service import ServiceManager, build_args, DEFAULT_BIN
from .store import (gpu_info, git_commit, llama_version, new_run_id, run_dir,
                    update_index, write_json)
from .metrics import summarize


def _p(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------- tasks
def cmd_tasks(_args: list[str]) -> int:
    tasks = load_tasks()
    _p(f"解析到 {len(tasks)} 个规定任务（来源：测试提示词.md）")
    _p(f"系统消息 sha256: {sha256_text(SYSTEM_MESSAGE)[:16]}…")
    for m in task_manifest(tasks):
        _p(f"  {m['prompt_id']}  {m['kind']:6s}  chars={m['input_chars']:6d}  "
           f"file={m['file_name'] or '-'}")
    return 0


# ---------------------------------------------------------------- serve
def cmd_serve(args: list[str]) -> int:
    cfg = load_service_config(args[0])
    mgr = ServiceManager(cfg)
    _p("启动命令:", " ".join(build_args(cfg)))
    ok = mgr.start(wait_ready_s=240)
    st = mgr.state()
    _p(f"运行={st.running} 就绪={st.ready} pid={st.pid}")
    if not ok:
        _p("启动失败，日志尾部：")
        _p(mgr.log_path.read_text(encoding="utf-8")[-2000:])
        return 1
    props = mgr.client.props() or {}
    _p("服务端自报：")
    for k in ("n_ctx", "total_slots", "model_path", "chat_template"):
        if k in props:
            _p(f"  {k} = {'(已加载)' if k == 'chat_template' else props[k]}")
    return 0


def cmd_stop(args: list[str]) -> int:
    cfg = load_service_config(args[0])
    run_dir_name = f"{cfg.config_id}"
    # 通过 pid 文件停止（由 serve 启动的进程组）
    import os
    import signal
    from .config import RESULTS_DIR as _R
    pidf = RESULTS_DIR / "service_logs" / f"{cfg.config_id}.pid"
    if pidf.is_file():
        pid = int(pidf.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            _p(f"已停止 pid={pid}")
        except Exception as exc:
            _p(f"停止失败: {exc}")
        pidf.unlink(missing_ok=True)
    else:
        _p("未找到 pid 文件；尝试 pkill")
        os.system("pkill -f llama-server || true")
    return 0


def cmd_status(args: list[str]) -> int:
    cfg = load_service_config(args[0])
    client = LlamaClient(cfg.base_url)
    health = client.health()
    _p("health:", json.dumps(health, ensure_ascii=False) if health else "无响应")
    props = client.props() or {}
    for k in ("n_ctx", "total_slots", "model_path"):
        if k in props:
            _p(f"{k} = {props[k]}")
    slots = client.slots()
    if slots:
        _p(f"slots: {len(slots)} 个，n_ctx={slots[0].get('n_ctx')}")
    return 0


# ---------------------------------------------------------------- bench
def cmd_bench(args: list[str]) -> int:
    svc = load_service_config(args[0])
    load = load_load_config(args[1])
    tasks = load_tasks()
    runner = LoadRunner(svc, load, tasks)
    _p(f"运行编号 {runner.run_id}  并发={load.concurrency}  轮次={load.repeats}  "
       f"输出上限={load.output_limit}")
    result = runner.execute()
    ov = result["overall"]
    _p(f"成功 {ov['success']}/{ov['total_requests']}  "
       f"TTFT中位 {ov['ttft_ms']['median']:.0f}ms  P95 {ov['ttft_ms']['p95']:.0f}ms")
    _p(f"TPOT(服务端) {ov['tpot_server_ms']['mean']:.1f}ms  "
       f"解码吞吐 {ov['throughput']['decode_tokens_per_s']:.1f} tok/s")
    _p("汇总:", str((runner.dir / 'summary.json').relative_to(DEV_ROOT)))
    return 0


# ---------------------------------------------------------------- quality
def cmd_quality(args: list[str]) -> int:
    """跑 10 个规定任务，保存完整输出原文（用于输出效果检查）。"""
    svc = load_service_config(args[0])
    load = load_load_config("quality")
    tasks = load_tasks()
    mgr = ServiceManager(svc)
    if not mgr.start(wait_ready_s=300):
        _p("服务启动失败")
        return 1
    try:
        runner = LoadRunner(svc, load, tasks)
        _p(f"运行编号 {runner.run_id}（输出质量采集）")
        result = runner.execute(monitor_gpu=True)
        ok = result["overall"]["success"]
        _p(f"完成 {ok}/{result['overall']['total_requests']}，输出见 "
           f"results/outputs/{runner.run_id}/")
    finally:
        mgr.stop()
    return 0


# ---------------------------------------------------------------- recover
def cmd_recover(args: list[str]) -> int:
    """异常恢复测试：运行中强杀服务，验证检测、重启、就绪确认与恢复后请求。"""
    cfg = load_service_config(args[0])
    tasks = load_tasks()
    client = LlamaClient(cfg.base_url, timeout_s=120)
    mgr = ServiceManager(cfg)
    run_id = new_run_id(f"{cfg.config_id}-recovery")
    d = run_dir(run_id)
    timeline: list[dict] = []

    def ev(kind: str, **kw) -> None:
        row = {"event": kind, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw}
        timeline.append(row)
        _p(f"  [{row['at']}] {kind} {kw if kw else ''}")

    _p(f"恢复测试运行编号 {run_id}")
    if not mgr.start(wait_ready_s=240):
        _p("服务未能启动")
        return 1
    ev("service_ready", pid=mgr.proc.pid)

    # 1) 恢复前基线请求
    msgs = tasks["P01"].build_messages()
    r0 = client.chat_stream(msgs, GenParams(max_tokens=64))
    ev("request_before", ok=r0.ok, ttft_ms=r0.ttft_ms, error=r0.error_message)

    # 2) 强制杀死服务，模拟异常退出
    ev("kill_service")
    mgr.kill()

    # 3) 检测：健康检查应失败
    time.sleep(1.5)
    healthy = client.is_ready()
    ev("detect_after_kill", service_ready=healthy)

    # 4) 中断期间的请求（已发送请求 → 应分类为失败，不重复计数）
    r1 = client.chat_stream(msgs, GenParams(max_tokens=64))
    ev("request_during_outage", ok=r1.ok, error_type=r1.error_type,
       status="failed" if not r1.ok else "success")

    # 5) 重启并等待就绪
    t_restart = time.time()
    ok = mgr.start(wait_ready_s=240)
    recovery_s = time.time() - t_restart
    ev("service_restarted", ready=ok, recovery_s=round(recovery_s, 2), pid=mgr.proc.pid)

    # 6) 恢复后请求，验证推理功能已恢复
    r2 = client.chat_stream(msgs, GenParams(max_tokens=64))
    ev("request_after_recovery", ok=r2.ok, ttft_ms=r2.ttft_ms)

    payload = {
        "run_id": run_id,
        "service_config": cfg.config_id,
        "timeline": timeline,
        "recovery_s": round(recovery_s, 2),
        "before_ok": r0.ok,
        "during_ok": r1.ok,
        "after_ok": r2.ok,
        "after_content_head": (r2.content or "")[:200],
        "handling": {
            "waiting_requests": "提交前检测到服务中断即标记 cancelled，不发送",
            "sent_requests": "已发送请求在中断时归为 failed/service_unavailable",
            "failed_requests": "按 max_retries 有限次重试，按逻辑请求去重计数",
        },
    }
    write_json(d / "recovery.json", payload)
    update_index(run_id, {
        "service_config": cfg.config_id, "kind": "recovery",
        "recovery_s": payload["recovery_s"],
        "summary_path": str((d / "recovery.json").relative_to(DEV_ROOT / "results")),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _p(f"恢复耗时 {payload['recovery_s']}s；前={r0.ok} 中断中={r1.ok} 恢复后={r2.ok}")
    return 0


# ---------------------------------------------------------------- matrix
def cmd_matrix(args: list[str]) -> int:
    """跑对比矩阵。参数形式 <service_cfg>:<load_cfg>，可给多个。

    默认跑基线与最终优化配置的单并发/多并发对比。
    """
    from collections import OrderedDict

    specs = args or [
        "baseline:bench_c1", "baseline:bench_c4",
        "opt_final:bench_c1", "opt_final:bench_c4",
    ]
    by_cfg: "OrderedDict[str, list[str]]" = OrderedDict()
    for spec in specs:
        cfg_id, load_id = spec.split(":")
        by_cfg.setdefault(cfg_id, []).append(load_id)

    tasks = load_tasks()
    meta = {
        "hardware": gpu_info(),
        "llama_version": llama_version(),
        "llama_commit": git_commit(),
        "system_message_sha256": sha256_text(SYSTEM_MESSAGE),
        "task_manifest": task_manifest(tasks),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": [{"service_config": s, "loads": l} for s, l in by_cfg.items()],
    }
    write_json(RESULTS_DIR / "matrix_meta.json", meta)

    for svc_id, load_ids in by_cfg.items():
        svc = load_service_config(svc_id)
        mgr = ServiceManager(svc)
        _p(f"\n===== 服务配置 {svc_id} =====")
        if not mgr.start(wait_ready_s=300):
            _p("启动失败，跳过")
            continue
        try:
            for load_id in load_ids:
                load = load_load_config(load_id)
                runner = LoadRunner(svc, load, tasks)
                res = runner.execute()
                ov = res["overall"]
                _p(f"  {load_id}: 成功 {ov['success']}/{ov['total_requests']} "
                   f"TTFT中位 {ov['ttft_ms']['median']:.0f}ms "
                   f"解码 {ov['throughput']['decode_tokens_per_s'] or 0:.1f} tok/s")
        finally:
            mgr.stop()
    _p("\n矩阵完成")
    return 0


def cmd_web(args: list[str]) -> int:
    """启动前端展示。用法：web [service_config_id] [port]"""
    from .web import serve
    cfg_id = args[0] if args else "opt_final"
    port = int(args[1]) if len(args) > 1 else 8000
    serve(cfg_id, port=port)
    return 0


def _ctx_from_args(extra_args: list[str]) -> tuple[int, int]:
    """从服务配置的 extra_args 中解析 -c 与 -np。"""
    ctx, np_ = 0, 1
    for i, a in enumerate(extra_args):
        if a in ("-c", "--ctx-size") and i + 1 < len(extra_args):
            ctx = int(extra_args[i + 1])
        if a in ("-np", "--parallel") and i + 1 < len(extra_args):
            np_ = int(extra_args[i + 1])
    return ctx, np_


def cmd_capacity(args: list[str]) -> int:
    """KV 容量验证：逐步加长输入，记录实测可用容量与失败位置。

    用法： capacity <service_cfg> [最大目标 token]
    """
    import json as _json
    from .store import write_json

    svc = load_service_config(args[0])
    targets = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384]
    if len(args) > 1:
        targets = [t for t in targets if t <= int(args[1])]

    ctx, np_ = _ctx_from_args(svc.extra_args)
    n_ctx_slot = ctx // np_ if np_ else ctx

    mgr = ServiceManager(svc)
    if not mgr.start(wait_ready_s=300):
        _p("服务启动失败")
        return 1
    try:
        client = LlamaClient(svc.base_url, timeout_s=300)
        filler = "这是一段用于测试上下文容量上限的填充文本，内容本身没有实际含义。"
        # 先测一段填充文本的 token 数，用于按目标长度缩放
        probe_msgs = [{"role": "user", "content": filler * 10}]
        per10 = client.count_input_tokens(probe_msgs)
        if not per10:
            _p("无法测定填充文本的 token 数，终止")
            return 1
        per_rep = per10 / 10.0

        rows = []
        _p(f"配置 {svc.config_id}: -c {ctx}, -np {np_} → 每 slot 上下文 {n_ctx_slot} token")
        for target in targets:
            reps = max(1, int(round(target / per_rep)))
            content = (filler * reps) + "\n\n请用一句话确认你收到了这段资料。"
            msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": content}]
            actual = client.count_input_tokens(msgs)
            r = client.chat_stream(msgs, GenParams(max_tokens=32, enable_thinking=False))
            rows.append({
                "target_tokens": target,
                "measured_input_tokens": actual,
                "ok": r.ok,
                "ttft_ms": r.ttft_ms,
                "total_ms": r.total_ms,
                "prompt_tokens": r.prompt_tokens,
                "error_type": r.error_type,
                "error_message": (r.error_message or "")[:200],
            })
            _p(f"  目标 {target:6d} → 实测输入 {actual if actual else '—':>6} "
               f"{'成功' if r.ok else '失败'} {r.error_type or ''} "
               f"{(r.error_message or '')[:90]}")
            if not r.ok and "exceed_context_size_error" in (r.error_message or ""):
                _p("  已达到上下文上限，停止加长")
                break

    finally:
        mgr.stop()

    payload = {
        "config_id": svc.config_id,
        "params": " ".join(svc.extra_args),
        "ctx_total": ctx,
        "n_slots": np_,
        "n_ctx_slot": n_ctx_slot,
        "probes": rows,
        "max_ok_input_tokens": max([r["measured_input_tokens"] or 0 for r in rows if r["ok"]],
                                   default=0),
    }
    p = RESULTS_DIR / f"capacity_{svc.config_id}.json"
    write_json(p, payload)
    _p(f"实测最大可用输入 token = {payload['max_ok_input_tokens']}；结果写入 {p.name}")
    return 0


def cmd_kvlog(args: list[str]) -> int:
    """以高日志级别短暂启动各服务配置，抓取 KV 缓冲区的日志证据。

    抓取内容：KV 单元格数（token 容量）、层数、seq 分配、K/V 精度与显存、
    模型权重与计算缓冲显存，以及由 GGUF 元数据推算的每 token KV 字节数。

    用法： kvlog [config_id ...]   默认全部配置
    """
    import re
    from .store import write_json

    RE_KV_DETAIL = re.compile(
        r"size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells,\s*(\d+)\s*layers,\s*"
        r"(\d+)/(\d+)\s*seqs\),\s*K\s*\(([^)]+)\):\s*([\d.]+)\s*MiB,\s*"
        r"V\s*\(([^)]+)\):\s*([\d.]+)\s*MiB")
    RE_KV_BUF = re.compile(r"CUDA0 KV buffer size\s*=\s*([\d.]+)\s*MiB")
    RE_CMP_BUF = re.compile(r"CUDA0 compute buffer size\s*=\s*([\d.]+)\s*MiB")
    RE_MODEL_BUF = re.compile(r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB")
    RE_META = re.compile(r"-\s+kv\s+\d+:\s+(\S+)\s+(?:u32|i32|f32|str)\s+=\s+([^\s]+)")

    ids = args or [c.config_id for c in load_service_configs()]
    out = []
    for cid in ids:
        svc = load_service_config(cid)
        log = RESULTS_DIR / "service_logs" / f"{cid}_kvlog.log"
        log.unlink(missing_ok=True)
        mgr = ServiceManager(svc)
        mgr.log_path = log
        mgr.cfg = svc
        mgr.cfg.extra_args = svc.extra_args + ["-lv", "5"]
        if not mgr.start(wait_ready_s=300):
            _p(f"{cid}: 启动失败")
            continue
        # 服务就绪后、停止前采集显存占用
        try:
            import subprocess as _sp
            smi = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=10, check=False).stdout.strip()
            parts = [x.strip() for x in smi.splitlines()[0].split(",")]
            rec_gpu = {"gpu_used_mib": int(parts[0]), "gpu_total_mib": int(parts[1])}
        except Exception:
            rec_gpu = {}
        mgr.stop()
        text = log.read_text(encoding="utf-8", errors="replace")

        rec = {"config_id": cid, "params": " ".join(svc.extra_args), **rec_gpu}
        m = re.search(r"n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'", text)
        if m:
            rec["n_slots"] = int(m.group(1))
            rec["n_ctx_slot"] = int(m.group(2))
            rec["kv_unified"] = m.group(3)
            rec["total_tokens"] = rec["n_slots"] * rec["n_ctx_slot"]

        m = RE_KV_DETAIL.search(text)
        if m:
            rec.update({
                "kv_total_mib": float(m.group(1)),
                "kv_cells_per_seq": int(m.group(2)),
                "n_layers": int(m.group(3)),
                "kv_seqs": f"{m.group(4)}/{m.group(5)}",
                "kv_type_k": m.group(6), "kv_k_mib": float(m.group(7)),
                "kv_type_v": m.group(8), "kv_v_mib": float(m.group(9)),
            })

        def mx(rx):
            vals = [float(v) for v in rx.findall(text)]
            return max(vals) if vals else None

        rec["cuda_kv_mib"] = mx(RE_KV_BUF)
        rec["cuda_compute_mib"] = mx(RE_CMP_BUF)
        rec["cuda_model_mib"] = mx(RE_MODEL_BUF)

        # GGUF 元数据 -> 每 token KV 字节数（独立于日志的交叉验证）
        meta = dict(RE_META.findall(text))
        rec["gguf"] = {k: meta.get(k) for k in (
            "qwen3.block_count", "qwen3.attention.head_count",
            "qwen3.attention.head_count_kv", "qwen3.attention.key_length",
            "qwen3.attention.value_length", "qwen3.context_length") if k in meta}
        try:
            nl = int(meta["qwen3.block_count"])
            nkv = int(meta["qwen3.attention.head_count_kv"])
            kl = int(meta["qwen3.attention.key_length"])
            vl = int(meta["qwen3.attention.value_length"])
            f16 = 2
            q8 = 34 / 32.0
            rec["bytes_per_token_f16"] = nl * nkv * (kl + vl) * f16
            rec["bytes_per_token_q8_0"] = round(nl * nkv * (kl + vl) * q8)
            if rec.get("total_tokens"):
                rec["kv_mib_computed_f16"] = round(
                    rec["bytes_per_token_f16"] * rec["total_tokens"] / 1024 / 1024, 2)
                rec["kv_mib_computed_q8_0"] = round(
                    rec["bytes_per_token_q8_0"] * rec["total_tokens"] / 1024 / 1024, 2)
        except (KeyError, ValueError):
            pass

        out.append(rec)
        _p(f"{cid}: {json.dumps(rec, ensure_ascii=False)}")
    write_json(RESULTS_DIR / "kv_capacity.json", out)
    _p(f"已写入 results/kv_capacity.json（{len(out)} 个配置）")
    return 0


def cmd_env(_args: list[str]) -> int:
    """采集环境与模型信息（含模型文件 SHA256），写入 results/env.json。"""
    import hashlib
    import os as _os
    import platform as _pf
    import subprocess as _sp
    from .store import write_json

    def sh(cmd: str, timeout: int = 60) -> str:
        try:
            return _sp.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, check=False).stdout.strip()
        except Exception:
            return ""

    svc = load_service_config("opt_final")
    model = _os.path.expanduser(svc.model_path)
    info = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": _pf.platform(),
        "python": sys.version.split()[0],
        "uname": sh("uname -a"),
        "gpu": gpu_info(),
        "nvidia_smi_full": sh("nvidia-smi"),
        "nvcc": sh("nvcc --version | tail -2"),
        "cmake": sh("cmake --version | head -1"),
        "gcc": sh("gcc --version | head -1"),
        "llama_commit": git_commit(),
        "llama_version": llama_version(),
        "model_path": model,
        "model_size_bytes": None,
        "model_sha256": None,
        "model_gguf_meta": sh(
            f"python3 /root/tools/gguf_meta.py {model} 2>/dev/null | head -40", timeout=120),
    }
    if _os.path.isfile(model):
        info["model_size_bytes"] = _os.path.getsize(model)
        h = hashlib.sha256()
        with open(model, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        info["model_sha256"] = h.hexdigest()
    write_json(RESULTS_DIR / "env.json", info)
    _p(f"模型 {model}")
    _p(f"SHA256 = {info['model_sha256']}")
    _p(f"大小   = {info['model_size_bytes']}")
    _p(f"GPU    = {info['gpu'].get('gpu_name')}")
    _p("已写入 results/env.json")
    return 0


def cmd_check(args: list[str]) -> int:
    """对某次运行的模型输出做效果检查。用法： check <run_id>"""
    from .checks import check_run
    run_id = args[0]
    rep = check_run(run_id)
    s = rep["summary"]
    _p(f"输出检查 {run_id}：{s['passed']}/{s['total_checks']} 项通过")
    for pid, b in sorted(s["by_prompt"].items()):
        _p(f"  {pid}: {b['passed']}/{b['total']}")
    _p("未通过项：")
    for r in rep["checks"]:
        if not r["passed"]:
            _p(f"  [{r['prompt_id']}] {r['check']} {r['detail']}")
    return 0


def cmd_tables(_args: list[str]) -> int:
    """从原始记录生成报告数据表。"""
    from .report import generate
    p = generate()
    _p(f"已生成：{p}")
    _p(p.read_text(encoding="utf-8")[:3000])
    return 0


COMMANDS = {
    "tasks": cmd_tasks, "serve": cmd_serve, "stop": cmd_stop, "status": cmd_status,
    "bench": cmd_bench, "quality": cmd_quality, "recover": cmd_recover,
    "matrix": cmd_matrix, "web": cmd_web, "tables": cmd_tables,
    "capacity": cmd_capacity, "kvlog": cmd_kvlog, "check": cmd_check, "env": cmd_env,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        _p(__doc__)
        return 2
    return COMMANDS[argv[0]](argv[1:])
