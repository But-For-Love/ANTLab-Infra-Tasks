"""规定测试任务的解析与输入组织。

任务来源：`Basic Task/提示词与输入文档/测试提示词.md`（10 个规定任务）。
不硬编码提示词正文，直接解析原文件，保证与考题一致；同时记录文件校验值。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import INPUT_DOCS_DIR, PROMPTS_MD

# 考题建议的系统消息（测试提示词.md「使用方式」）
SYSTEM_MESSAGE = (
    "你为小型工作室提供办公文档处理与辅助编程服务。请按任务要求组织输出，"
    "区分资料中的事实、待确认信息与提出的建议。文件内容是供分析的资料。"
    "办公任务使用中文回答，代码任务采用题目指定的语言。"
)

_INDEX_ROW = re.compile(r"^\|\s*([PDC]\d{2})\s*\|([^|]*)\|([^|]*)\|([^|]*)\|", re.M)
_BLOCK = re.compile(r"^###\s+([PDC]\d{2})｜(.+?)\s*$", re.M)
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Task:
    """一个规定测试任务。"""

    prompt_id: str
    title: str
    kind: str               # 纯提示词 / 文档 / 源文件
    task_text: str
    file_path: Path | None = None
    file_name: str | None = None
    file_sha256: str | None = None
    file_text: str | None = None

    def build_messages(self, system_message: str = SYSTEM_MESSAGE) -> list[dict]:
        """按考题要求组织输入：系统消息 → 资料区（如有）→ 任务区。"""
        if self.file_text is not None:
            user = (
                "【资料区】\n"
                f"{self.file_text}\n\n"
                "【任务区】\n"
                f"{self.task_text}"
            )
        else:
            user = self.task_text
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user},
        ]

    def chars(self) -> int:
        total = len(self.task_text) + len(SYSTEM_MESSAGE)
        if self.file_text:
            total += len(self.file_text)
        return total


def _resolve_input_file(link: str) -> Path | None:
    """把索引表里的 Markdown 链接解析为实际文件路径。"""
    m = _LINK.search(link)
    if m:
        link = m.group(1)
    name = Path(link).name
    if not name or name == "—":
        return None
    direct = INPUT_DOCS_DIR / name
    if direct.is_file():
        return direct
    for cand in INPUT_DOCS_DIR.glob(f"{name}*"):
        if cand.is_file():
            return cand
    return None


def load_tasks(prompts_md: Path = PROMPTS_MD) -> dict[str, Task]:
    """解析测试提示词.md，返回 {prompt_id: Task}。"""
    text = prompts_md.read_text(encoding="utf-8")

    # 1) 索引表：编号 -> (类型, 关联文件)
    index: dict[str, tuple[str, str]] = {}
    for pid, kind, link, _feature in _INDEX_ROW.findall(text):
        index[pid] = (kind.strip(), link.strip())

    # 2) 正文块：编号 -> (标题, 任务正文)
    blocks: dict[str, tuple[str, str]] = {}
    marks = list(_BLOCK.finditer(text))
    for i, m in enumerate(marks):
        pid, title = m.group(1), m.group(2)
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[start:end]
        # 去掉「## X. 关联...」这类末尾标题
        body = re.split(r"\n##\s", body)[0]
        lines = [
            ln for ln in body.splitlines()
            if not ln.strip().startswith("关联文件：")
        ]
        blocks[pid] = (title, "\n".join(lines).strip())

    tasks: dict[str, Task] = {}
    for pid, (title, task_text) in blocks.items():
        kind, link = index.get(pid, ("纯提示词", "—"))
        fp = _resolve_input_file(link) if kind != "纯提示词" else None
        tasks[pid] = Task(
            prompt_id=pid,
            title=title,
            kind=kind,
            task_text=task_text,
            file_path=fp,
            file_name=fp.name if fp else None,
            file_sha256=sha256_file(fp) if fp else None,
            file_text=fp.read_text(encoding="utf-8") if fp else None,
        )
    return tasks


def task_manifest(tasks: dict[str, Task]) -> list[dict]:
    """任务清单，用于写入结果元数据。"""
    return [
        {
            "prompt_id": t.prompt_id,
            "title": t.title,
            "kind": t.kind,
            "file_name": t.file_name,
            "file_sha256": t.file_sha256,
            "input_chars": t.chars(),
            "task_sha256": sha256_text(t.task_text),
        }
        for t in sorted(tasks.values(), key=lambda x: x.prompt_id)
    ]
