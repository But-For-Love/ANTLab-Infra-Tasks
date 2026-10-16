"""模型输出效果检查。

对应考题「输出检查与负载组织」：
- 办公任务：核对事实准确性、新旧决定、条款引用、负责人与日期
- 编程任务：核对问题定位、修复正确性、代码可执行性与边界覆盖

用法： python3 run.py check <run_id>
结果写入 results/checks/<run_id>.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import DEV_ROOT, INPUT_DOCS_DIR, RESULTS_DIR
from .store import write_json

OUTPUTS_DIR = RESULTS_DIR / "outputs"
CHECKS_DIR = RESULTS_DIR / "checks"
CHECKS_DIR.mkdir(parents=True, exist_ok=True)

# 办公文档的客观事实（来自输入资料，可逐条回查）
FACTS = {
    "演示日期（最新决定）": "10 月 20 日",
    "正式交付日期（未变）": "10 月 23 日",
    "旧演示日期（已被取代）": "10 月 16 日",
}
# 制度条款编号（来自办公输入02）
CLAUSES_D03 = ["B03", "B05", "C01", "C02", "C03", "D03", "D05"]
CLAUSES_D04 = ["A01", "A02", "A03", "A04", "C03", "D03"]
# 代码审查应发现的关键缺陷（关键词组，命中即认为指出）
C01_DEFECTS = {
    "空输入除零": [["ZeroDivision", "除零", "除以", "len(records)", "空列表"]],
    "未按 id 去重": [["去重", "重复", "唯一", "第一条", "seen", "dedup"]],
    "owner 未规范化": [["unassigned", "owner", "处理人", "None", "空"]],
    "status 未规范化/未知状态崩溃": [["other", "未知", "strip", "lower", "KeyError", "大小写"]],
    "平均耗时口径错误": [["平均", "average", "分母", "int(", "两位小数", "耗时"]],
    "未关闭工单计入耗时": [["未关闭", "closed_at", "closed", "跳过"]],
    "时间解析无异常处理": [["fromisoformat", "ValueError", "无效时间", "解析", "时区"]],
}

# C02 修复版应满足的约定（用于执行生成的代码）
C02_EXPECTED_CASES = [
    ("空输入", [], {"total": 0, "avg": None}),
]


def _load_output(run_id: str, pid: str) -> str | None:
    p = OUTPUTS_DIR / run_id / f"{pid}.md"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        _, _, text = text.partition("-->")
    return text.strip()


def _norm(text: str) -> str:
    """归一化，便于匹配日期与条款的不同写法。"""
    t = text.replace(" ", "").replace("\u3000", "")
    t = t.replace("月", "-").replace("日", "").replace("号", "")
    t = re.sub(r"10-(\d+)", lambda m: f"10-{int(m.group(1)):02d}", t)
    return t


def check_office(run_id: str, results: list[dict]) -> None:
    def add(pid, name, passed, detail=""):
        results.append({"prompt_id": pid, "check": name,
                        "passed": bool(passed), "detail": detail})

    for pid in ("D01", "D02"):
        text = _load_output(run_id, pid) or ""
        n = _norm(text)
        add(pid, "使用最新会议决定的演示日期（10-20）", "10-20" in n,
            "输出中未出现 10 月 20 日" if "10-20" not in n else "")
        add(pid, "保留正式的交付日期（10-23）", "10-23" in n,
            "输出中未出现 10 月 23 日" if "10-23" not in n else "")
        # 旧日期只有在被明确标注为已变更/原定时才允许出现
        if "10-16" in n:
            ok = bool(re.search(r"(原定|改为|调整|变更|不再|取消)", text))
            add(pid, "旧演示日期仅作为变更前背景出现", ok,
                "出现 10-16 但未说明其已被变更" if not ok else "已标注为原定/变更")
        else:
            add(pid, "未误用已被取代的演示日期", True)
        add(pid, "包含行动项或负责人信息",
            any(k in text for k in ("负责人", "陈珂", "宋岚", "林悦", "许舟", "沈遥")),
            "")
        add(pid, "标注待确认事项", "待确认" in text or "待明确" in text, "")

    # D03：多条款流程问答
    text3 = _load_output(run_id, "D03") or ""
    for c in CLAUSES_D03:
        add("D03", f"引用条款 {c}", c in text3, "未引用" if c not in text3 else "")
    add("D03", "覆盖客户验收周例外流程（B05）",
        ("验收周" in text3) and ("B05" in text3), "")
    add("D03", "覆盖远程处理客户资料的 VPN 要求（D05）",
        ("VPN" in text3 or "vpn" in text3) and ("D05" in text3), "")

    # D04：入职清单
    text4 = _load_output(run_id, "D04") or ""
    for c in CLAUSES_D04:
        add("D04", f"引用条款 {c}", c in text4, "未引用" if c not in text4 else "")
    add("D04", "单列接触客户资料前置事项",
        "客户资料" in text4 and ("前置" in text4 or "接触" in text4), "")
    add("D04", "覆盖地点或设备变化的补办手续",
        ("B06" in text4) or ("C05" in text4) or ("E04" in text4), "")


def check_code_review(run_id: str, results: list[dict]) -> None:
    """C01：是否定位到源代码中的关键缺陷。"""
    text = _load_output(run_id, "C01") or ""
    for name, groups in C01_DEFECTS.items():
        hit = any(any(k.lower() in text.lower() for k in g) for g in groups)
        results.append({"prompt_id": "C01", "check": f"定位缺陷：{name}",
                        "passed": hit, "detail": "" if hit else "输出未提及该缺陷"})
    results.append({"prompt_id": "C01", "check": "给出最小复现输入",
                    "passed": bool(re.search(r"(\[\]|\{\}|T001|复现|最小输入)", text)),
                    "detail": ""})


def _extract_python(text: str) -> str | None:
    """从 Markdown 输出中取出最长的 Python 代码块。"""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        return None
    return max(blocks, key=len)


C02_HARNESS = r'''
import json, sys, traceback
from datetime import datetime

SRC = open("candidate.py", encoding="utf-8").read()
ns = {}
try:
    exec(compile(SRC, "candidate.py", "exec"), ns)
except Exception as e:
    print(json.dumps({"compiled": False, "error": f"{type(e).__name__}: {e}"}))
    sys.exit(0)

fn = ns.get("summarize_tickets")
if not callable(fn):
    print(json.dumps({"compiled": True, "has_func": False}))
    sys.exit(0)

def T(h, tz="+08:00"):
    return f"2026-10-12T{h}:00:00{tz}"

def rec(**kw):
    d = {"id": "T1", "owner": "Chen", "status": "open",
         "created_at": T("09"), "closed_at": None}
    d.update(kw)
    return d

cases = {}
def run(name, records):
    try:
        cases[name] = {"ok": True, "value": fn(records)}
    except Exception as e:
        cases[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

run("empty", [])
run("missing_owner_key", [{"id": "T1", "status": "open", "created_at": T("09"), "closed_at": None}])
run("missing_status_key", [{"id": "T1", "owner": "Chen", "created_at": T("09"), "closed_at": None}])
run("owner_none", [rec(owner=None)])
run("owner_blank", [rec(owner="   ")])
run("owner_spaces", [rec(owner=" Chen ")])
run("status_case_space", [rec(status=" Closed ")])
run("status_unknown", [rec(status="weird")])
run("dup_id", [rec(id="T1"), rec(id="T1")])
run("blank_id", [rec(id="   ")])
run("unclosed_with_closed_at", [rec(status="open", closed_at=T("10"))])
run("closed_missing_time", [rec(status="closed", closed_at=None)])
run("invalid_time", [rec(status="closed", closed_at="not-a-time")])
run("naive_time", [rec(status="closed", created_at="2026-10-12T09:00:00",
                       closed_at="2026-10-12T11:00:00")])
run("closed_before_created", [rec(status="closed", closed_at="2026-10-12T08:00:00")])
run("cross_tz", [rec(status="closed", created_at="2026-10-12T09:00:00+08:00",
                     closed_at="2026-10-12T04:00:00+00:00")])
run("no_valid_hours", [rec(status="open", closed_at=None), rec(status="open", closed_at=None)])

SAMPLE = [
    {"id":"T001","owner":"Chen","status":"closed","created_at":T("09"),"closed_at":"2026-10-12T11:30:00+08:00"},
    {"id":"T002","owner":"Song","status":"processing","created_at":T("10"),"closed_at":None},
    {"id":"T003","owner":"Chen","status":"open","created_at":T("13"),"closed_at":None},
    {"id":"T004","owner":"Zhou","status":"closed","created_at":T("14"),"closed_at":"2026-10-12T18:15:00+08:00"},
    {"id":"T001","owner":"Chen","status":"closed","created_at":T("09"),"closed_at":"2026-10-12T11:30:00+08:00"},
]
run("sample_dedup", SAMPLE)

print(json.dumps({"compiled": True, "has_func": True, "cases": cases}, ensure_ascii=False))
'''


def check_code_fix(run_id: str, results: list[dict]) -> dict:
    """C02：抽取模型给出的修复代码，实际执行并核对边界行为。"""
    text = _load_output(run_id, "C02") or ""
    code = _extract_python(text)
    detail = {"code_found": bool(code), "code_chars": len(code or "")}

    def add(name, passed, msg=""):
        results.append({"prompt_id": "C02", "check": name,
                        "passed": bool(passed), "detail": msg})

    if not code:
        add("输出包含可提取的 Python 代码", False, "未找到代码块")
        return detail
    add("输出包含可提取的 Python 代码", True)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "candidate.py").write_text(code, encoding="utf-8")
        (d / "harness.py").write_text(C02_HARNESS, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "harness.py"], cwd=d,
                                  capture_output=True, text=True, timeout=60)
            out = proc.stdout.strip().splitlines()
            data = json.loads(out[-1]) if out else {}
        except Exception as exc:  # noqa: BLE001
            add("生成代码可执行", False, f"{type(exc).__name__}: {exc}")
            return detail

    if not data.get("compiled"):
        add("生成代码可执行", False, f"编译/执行失败：{data.get('error')}")
        return detail
    add("生成代码可执行", True)
    if not data.get("has_func"):
        add("保留主要函数 summarize_tickets", False, "未定义该函数")
        return detail
    add("保留主要函数 summarize_tickets", True)

    cases = data.get("cases", {})
    detail["cases"] = {k: ("ok" if v.get("ok") else v.get("error")) for k, v in cases.items()}

    def val(name, *path):
        c = cases.get(name) or {}
        if not c.get("ok"):
            return None
        v = c["value"]
        for p in path:
            if not isinstance(v, dict):
                return None
            v = v.get(p)
        return v

    # 逐条核对约定
    add("空输入不崩溃且返回 total=0", val("empty", "total") == 0,
        f"实际 {val('empty','total')!r}")
    add("空输入 average_closed_hours 为 None",
        val("empty", "average_closed_hours") is None,
        f"实际 {val('empty','average_closed_hours')!r}")
    add("缺 owner 键不崩溃", (cases.get("missing_owner_key") or {}).get("ok", False),
        str((cases.get("missing_owner_key") or {}).get("error", "")))
    add("缺 status 键不崩溃", (cases.get("missing_status_key") or {}).get("ok", False),
        str((cases.get("missing_status_key") or {}).get("error", "")))
    add("未知状态不崩溃", (cases.get("status_unknown") or {}).get("ok", False),
        str((cases.get("status_unknown") or {}).get("error", "")))
    add("无效时间不崩溃", (cases.get("invalid_time") or {}).get("ok", False),
        str((cases.get("invalid_time") or {}).get("error", "")))
    add("owner=None 归入 unassigned",
        "unassigned" in json.dumps(val("owner_none", "by_owner") or {}, ensure_ascii=False),
        f"实际 {val('owner_none','by_owner')!r}")
    add("owner 为空白归入 unassigned",
        "unassigned" in json.dumps(val("owner_blank", "by_owner") or {}, ensure_ascii=False),
        f"实际 {val('owner_blank','by_owner')!r}")
    add("owner 去首尾空格", " Chen " not in json.dumps(val("owner_spaces", "by_owner") or {}),
        f"实际 {val('owner_spaces','by_owner')!r}")
    add("状态忽略大小写与空格（Closed→closed）",
        (val("status_case_space", "by_status") or {}).get("closed") == 1,
        f"实际 {val('status_case_space','by_status')!r}")
    add("未知状态计入 other",
        (val("status_unknown", "by_status") or {}).get("other") == 1,
        f"实际 {val('status_unknown','by_status')!r}")
    add("同 id 只统计第一条（total=1）", val("dup_id", "total") == 1,
        f"实际 {val('dup_id','total')!r}")
    add("空白 id 跳过（total=0）", val("blank_id", "total") == 0,
        f"实际 {val('blank_id','total')!r}")
    add("无有效耗时工单 average 为 None",
        val("no_valid_hours", "average_closed_hours") is None,
        f"实际 {val('no_valid_hours','average_closed_hours')!r}")
    add("样例去重后 total=4", val("sample_dedup", "total") == 4,
        f"实际 {val('sample_dedup','total')!r}")
    add("样例 closed=2", (val("sample_dedup", "by_status") or {}).get("closed") == 2,
        f"实际 {(val('sample_dedup','by_status') or {}).get('closed')!r}")
    avg = val("sample_dedup", "average_closed_hours")
    add("样例平均耗时为 3.38（保留两位小数）",
        avg is not None and abs(float(avg) - 3.38) < 0.005, f"实际 {avg!r}")
    return detail


def check_run(run_id: str) -> dict:
    results: list[dict] = []
    report = {"run_id": run_id, "checks": results}
    check_office(run_id, results)
    check_code_review(run_id, results)
    report["c02_detail"] = check_code_fix(run_id, results)

    by_pid: dict[str, dict] = {}
    for r in results:
        b = by_pid.setdefault(r["prompt_id"], {"passed": 0, "total": 0})
        b["total"] += 1
        b["passed"] += 1 if r["passed"] else 0
    report["summary"] = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "by_prompt": by_pid,
    }
    write_json(CHECKS_DIR / f"{run_id}.json", report)
    return report
