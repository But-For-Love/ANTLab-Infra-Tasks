"""固定负载执行器。

职责
----
- 按固定负载定义组织请求（提示词、并发数、重复轮次、到达方式）
- 记录请求顺序与实际到达时间、重试关联与状态分类
- 逐请求写 JSONL 原始记录，按轮次与整体写汇总
- 运行期间采样 GPU

请求状态
--------
queued    已入队、尚未发送（服务不可用时置为 cancelled，不重复计数）
sent      已发送、等待响应
success   正常完成且拿到生成内容
failed    收到错误响应或连接失败
timeout   超过请求超时
cancelled 提交前因服务中断被取消

重试与重复计数
--------------
每个逻辑请求有唯一 request_id。重试记录在同一 request_id 的 attempts 中，
汇总时按逻辑请求计数一次，retry_count 单独统计，避免重复请求被重复计数。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .config import LoadConfig, ServiceConfig
from .llm_client import GenParams, LlamaClient
from .metrics import aggregate_rounds, summarize
from .service import GpuMonitor
from .store import append_jsonl, new_run_id, run_dir, save_output, update_index, write_json


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class LoadRunner:
    def __init__(self, service_cfg: ServiceConfig, load_cfg: LoadConfig,
                 tasks: dict, run_id: str | None = None):
        self.service_cfg = service_cfg
        self.load_cfg = load_cfg
        self.tasks = tasks
        self.run_id = run_id or new_run_id(f"{service_cfg.config_id}-c{load_cfg.concurrency}")
        self.dir = run_dir(self.run_id)
        self.jsonl = self.dir / "requests.jsonl"
        self.client = LlamaClient(service_cfg.base_url,
                                  timeout_s=load_cfg.request_timeout_s)
        self.gen = GenParams(
            max_tokens=load_cfg.output_limit,
            temperature=load_cfg.temperature,
            top_p=load_cfg.top_p,
            top_k=load_cfg.top_k,
            seed=load_cfg.seed,
            enable_thinking=load_cfg.enable_thinking,
        )
        self._lock = threading.Lock()
        self._submit_seq = 0

    # ---------- 单请求 ----------
    def _attempt(self, prompt_id: str, attempt: int) -> dict:
        task = self.tasks[prompt_id]
        messages = task.build_messages()
        with self._lock:
            self._submit_seq += 1
            seq = self._submit_seq
        planned = _iso()
        res = self.client.chat_stream(messages, self.gen)
        record = res.to_dict()
        record.update({
            "run_id": self.run_id,
            "attempt": attempt,
            "submit_seq": seq,
            "planned_submit_at": planned,
            "prompt_id": prompt_id,
            "prompt_title": task.title,
            "task_kind": task.kind,
            "file_name": task.file_name,
            "file_sha256": task.file_sha256,
            "concurrency": self.load_cfg.concurrency,
            "output_limit": self.load_cfg.output_limit,
            "service_config": self.service_cfg.config_id,
        })
        # 考题要求逐请求记录包含「输出内容」
        record["content"] = res.content
        record["reasoning"] = res.reasoning
        if res.ok and res.content.strip():
            record["status"] = "success"
        elif res.error_type == "timeout":
            record["status"] = "timeout"
        elif res.error_type == "connection_error":
            record["status"] = "failed"
            record["error_type"] = "service_unavailable"
        else:
            record["status"] = "failed"
        return record

    def _run_logical(self, prompt_id: str, index: int, stop_evt: threading.Event) -> dict:
        """执行一个逻辑请求，含有限次重试与去重计数。"""
        request_id = f"{self.run_id}-{prompt_id}-{index:03d}"
        if stop_evt.is_set():
            return {
                "request_id": request_id, "run_id": self.run_id, "prompt_id": prompt_id,
                "status": "cancelled", "retry_count": 0,
                "error_type": "service_interrupted",
                "error_message": "提交前检测到服务中断，请求未发送",
                "attempts": [],
            }

        attempts = []
        last = None
        for attempt in range(1, self.load_cfg.max_retries + 2):
            last = self._attempt(prompt_id, attempt)
            last["request_id"] = request_id
            attempts.append(last)
            if last["status"] == "success":
                break
            if stop_evt.is_set():
                break
            time.sleep(1.0)  # 重试前退避

        final = dict(last)
        final["request_id"] = request_id
        final["retry_count"] = len(attempts) - 1
        final["attempt_count"] = len(attempts)
        # 保留每次尝试的摘要，避免原始记录丢失重试过程
        final["attempts"] = [
            {
                "attempt": a["attempt"],
                "status": a["status"],
                "error_type": a.get("error_type"),
                "error_message": a.get("error_message"),
                "ttft_ms": a.get("ttft_ms"),
                "total_ms": a.get("total_ms"),
            }
            for a in attempts
        ]
        return final

    # ---------- 单轮 ----------
    def _one_round(self, round_idx: int, stop_evt: threading.Event) -> tuple[list[dict], float]:
        jobs = list(enumerate(self.load_cfg.prompt_ids))
        records: list[dict] = []
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.load_cfg.concurrency) as pool:
            futures = []
            for i, pid in jobs:
                if self.load_cfg.arrival_mode == "fixed_interval":
                    time.sleep(self.load_cfg.arrival_interval_ms / 1000.0)
                futures.append(pool.submit(self._run_logical, pid, i, stop_evt))
            for fut in as_completed(futures):
                rec = fut.result()
                rec["round"] = round_idx
                records.append(rec)

        wall = time.perf_counter() - t0
        records.sort(key=lambda r: r.get("submit_seq") or 0)
        return records, wall

    # ---------- 整体 ----------
    def execute(self, monitor_gpu: bool = True) -> dict:
        """执行完整负载（repeats 轮），返回汇总结果。"""
        monitor = GpuMonitor(self.run_id) if monitor_gpu else None
        if monitor:
            monitor.start()

        stop_evt = threading.Event()
        all_records: list[dict] = []
        round_summaries: list[dict] = []
        round_walls: list[float] = []

        try:
            for rnd in range(1, self.load_cfg.repeats + 1):
                # 服务不可用时置位，使未发送请求被标记而不是无限等待
                if not self.client.is_ready():
                    stop_evt.set()
                recs, wall = self._one_round(rnd, stop_evt)
                for rec in recs:
                    append_jsonl(self.jsonl, rec)
                all_records.extend(recs)
                s = summarize(recs, wall)
                s["round"] = rnd
                round_summaries.append(s)
                round_walls.append(wall)
                # 每轮之间留间隔，减少上一轮残留负载对下一轮的影响
                if rnd < self.load_cfg.repeats:
                    time.sleep(2.0)
        finally:
            gpu = monitor.stop() if monitor else {}

        overall = summarize(all_records, sum(round_walls))
        result = {
            "run_id": self.run_id,
            "service_config": self.service_cfg.config_id,
            "load_id": self.load_cfg.load_id,
            "concurrency": self.load_cfg.concurrency,
            "repeats": self.load_cfg.repeats,
            "output_limit": self.load_cfg.output_limit,
            "overall": overall,
            "rounds": round_summaries,
            "aggregate": aggregate_rounds(round_summaries),
            "gpu": gpu,
            "finished_at": _iso(),
        }
        write_json(self.dir / "summary.json", result)

        # 保存每个规定任务的输出原文（取每个 prompt_id 的首次成功结果）
        for rec in all_records:
            if rec.get("status") == "success" and rec.get("content"):
                p = save_output(self.run_id, rec["prompt_id"], rec["content"], rec)
                rec["output_path"] = str(p.relative_to(self.dir.parent.parent))
        write_json(self.dir / "summary.json", result)

        update_index(self.run_id, {
            "service_config": self.service_cfg.config_id,
            "load_id": self.load_cfg.load_id,
            "concurrency": self.load_cfg.concurrency,
            "repeats": self.load_cfg.repeats,
            "success_rate": overall.get("success_rate"),
            "ttft_p50_ms": overall.get("ttft_ms", {}).get("median"),
            "ttft_p95_ms": overall.get("ttft_ms", {}).get("p95"),
            "output_tokens": overall.get("tokens", {}).get("output_total"),
            "decode_tokens_per_s": overall.get("throughput", {}).get("decode_tokens_per_s"),
            "summary_path": str((self.dir / "summary.json").relative_to(self.dir.parent.parent)),
            "finished_at": result["finished_at"],
        })
        return result
