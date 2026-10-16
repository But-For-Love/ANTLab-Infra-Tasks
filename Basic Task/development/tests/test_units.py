"""单元测试：提示词解析、指标计算、输入组织。

运行： python3 -m pytest tests/ -q   或   python3 tests/test_units.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from antlab.metrics import aggregate_rounds, percentile, summarize  # noqa: E402
from antlab.prompts import (SYSTEM_MESSAGE, load_tasks, sha256_file,  # noqa: E402
                            sha256_text, task_manifest)


class TestPromptParsing(unittest.TestCase):
    """规定任务的解析必须与考题文件一致。"""

    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks()

    def test_ten_tasks(self):
        self.assertEqual(len(self.tasks), 10)

    def test_ids(self):
        self.assertEqual(
            sorted(self.tasks), ["C01", "C02", "D01", "D02", "D03", "D04",
                                 "P01", "P02", "P03", "P04"])

    def test_kinds(self):
        self.assertEqual(self.tasks["P01"].kind, "纯提示词")
        self.assertEqual(self.tasks["D01"].kind, "文档")
        self.assertEqual(self.tasks["C01"].kind, "源文件")

    def test_pure_prompts_have_no_file(self):
        for pid in ("P01", "P02", "P03", "P04"):
            self.assertIsNone(self.tasks[pid].file_path, pid)
            self.assertIsNone(self.tasks[pid].file_text, pid)

    def test_file_tasks_have_content_and_hash(self):
        for pid in ("D01", "D02", "D03", "D04", "C01", "C02"):
            t = self.tasks[pid]
            self.assertIsNotNone(t.file_path, pid)
            self.assertTrue(t.file_text and len(t.file_text) > 500, pid)
            self.assertEqual(len(t.file_sha256), 64, pid)

    def test_file_sha256_matches_disk(self):
        for pid, t in self.tasks.items():
            if t.file_path:
                self.assertEqual(t.file_sha256, sha256_file(t.file_path), pid)

    def test_metadata_line_stripped_from_task_text(self):
        # 「关联文件：...」属于元数据，不应出现在任务正文里
        for t in self.tasks.values():
            self.assertNotIn("关联文件：", t.task_text, t.prompt_id)

    def test_input_order_system_doc_task(self):
        msgs = self.tasks["D01"].build_messages()
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(msgs[0]["content"], SYSTEM_MESSAGE)
        body = msgs[1]["content"]
        self.assertIn("【资料区】", body)
        self.assertIn("【任务区】", body)
        # 资料区必须在任务区之前
        self.assertLess(body.index("【资料区】"), body.index("【任务区】"))

    def test_pure_prompt_has_no_doc_section(self):
        body = self.tasks["P01"].build_messages()[1]["content"]
        self.assertNotIn("【资料区】", body)

    def test_inputs_within_50000_token_budget(self):
        # 按字符数粗核：考题要求单条输入（含系统消息与模板）不超过 50,000 tokens。
        # 中文约 1 token/字，字符数远低于上限即可确认不超。
        for m in task_manifest(self.tasks):
            self.assertLess(m["input_chars"], 20000, m["prompt_id"])

    def test_system_message_hash_stable(self):
        self.assertEqual(len(sha256_text(SYSTEM_MESSAGE)), 64)


class TestMetrics(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(percentile(xs, 50), 5)
        self.assertEqual(percentile(xs, 95), 10)
        self.assertEqual(percentile(xs, 100), 10)
        self.assertEqual(percentile([], 50), None)
        self.assertEqual(percentile([7], 95), 7)

    def test_summarize_counts_only_logical_requests(self):
        recs = [
            {"status": "success", "ttft_ms": 100, "total_ms": 1000,
             "tpot_server_ms": 20, "output_tokens": 50, "server_gen_ms": 1000,
             "prompt_tokens": 10, "retry_count": 0},
            {"status": "failed", "retry_count": 1, "output_tokens": None},
            {"status": "success", "ttft_ms": 300, "total_ms": 2000,
             "tpot_server_ms": 25, "output_tokens": 40, "server_gen_ms": 1000,
             "prompt_tokens": 10, "retry_count": 0},
        ]
        s = summarize(recs, wall_clock_s=10.0)
        self.assertEqual(s["total_requests"], 3)
        self.assertEqual(s["success"], 2)
        self.assertEqual(s["by_status"], {"success": 2, "failed": 1})
        self.assertEqual(s["retried_requests"], 1)
        # TTFT 只统计成功请求
        self.assertEqual(s["ttft_ms"]["median"], 100)   # 最近秩法: n=2, rank=1 -> 100
        self.assertEqual(s["ttft_ms"]["p95"], 300)
        # 吞吐分母为服务端生成耗时之和（2s），不是墙钟
        self.assertAlmostEqual(s["throughput"]["decode_tokens_per_s"], 45.0, places=3)
        self.assertAlmostEqual(s["throughput"]["requests_per_s"], 0.2, places=3)

    def test_success_rate_zero_division_safe(self):
        s = summarize([], wall_clock_s=1.0)
        self.assertEqual(s["total_requests"], 0)
        self.assertIsNone(s["success_rate"])
        self.assertIsNone(s["ttft_ms"]["median"])

    def test_aggregate_rounds_spread(self):
        rounds = [
            {"ttft_ms": {"median": 100.0}, "success_rate": 1.0,
             "throughput": {"decode_tokens_per_s": 50.0}},
            {"ttft_ms": {"median": 120.0}, "success_rate": 1.0,
             "throughput": {"decode_tokens_per_s": 60.0}},
        ]
        a = aggregate_rounds(rounds)
        self.assertEqual(a["rounds"], 2)
        self.assertAlmostEqual(a["ttft_p50_ms"]["mean"], 110.0, places=3)
        self.assertAlmostEqual(a["ttft_p50_ms"]["min"], 100.0, places=3)
        self.assertAlmostEqual(a["ttft_p50_ms"]["max"], 120.0, places=3)
        # 波动率 = (120-100)/110
        self.assertAlmostEqual(a["ttft_p50_ms"]["spread_pct"], 2000 / 110, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
