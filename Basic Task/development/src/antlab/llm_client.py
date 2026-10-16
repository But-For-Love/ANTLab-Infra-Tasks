"""llama-server 推理客户端。

职责
----
1. 流式调用 `/v1/chat/completions`，采集客户端可观察的时间点：
   提交 → 首个有效生成内容 → 完成。
2. 采集服务端权威 token 计数与耗时（`timings` / `usage`）。
3. 通过 `/apply-template` + `/tokenize` 取得聊天模板处理后的真实输入 token 数。
4. 健康检查与就绪判定。

时间口径
--------
- ttft_ms：客户端提交至首个**非空**生成内容，包含排队与传输（对应考题的 TTFT 定义）。
- total_ms：客户端提交至流结束。
- 服务端时间单独取自 `timings`，用于区分客户端测量与服务端生成耗时。
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone


class ServerError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


@dataclass
class GenParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    seed: int = 42
    stream: bool = True
    # Qwen3 为思考模型：False 时关闭思考，正文直接进入 content；
    # None 表示不传该参数（使用服务端默认，即开启思考）。
    enable_thinking: bool | None = False

    def template_kwargs(self) -> dict:
        return {} if self.enable_thinking is None else {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }


@dataclass
class StreamResult:
    """一次请求的完整观测结果。"""

    ok: bool = False
    error_type: str | None = None
    error_message: str | None = None

    t_submit: str = ""
    t_first_content: str | None = None
    t_done: str | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None

    content: str = ""
    reasoning: str = ""           # 思考内容（enable_thinking=True 时非空）
    chunk_count: int = 0          # 流式数据块数（不等于 token 数）
    finish_reason: str | None = None

    # 服务端权威数据
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int | None:
        """聊天模板处理后的输入 token 数（含被缓存复用的部分）。"""
        if "prompt_n" in self.timings:
            return int(self.timings.get("prompt_n", 0)) + int(self.timings.get("cache_n", 0))
        if self.usage:
            return int(self.usage.get("prompt_tokens", 0))
        return None

    @property
    def cached_tokens(self) -> int | None:
        if "cache_n" in self.timings:
            return int(self.timings["cache_n"])
        if self.usage:
            det = self.usage.get("prompt_tokens_details") or {}
            if "cached_tokens" in det:
                return int(det["cached_tokens"])
        return None

    @property
    def output_tokens(self) -> int | None:
        if "predicted_n" in self.timings:
            return int(self.timings["predicted_n"])
        if self.usage:
            return int(self.usage.get("completion_tokens", 0))
        return None

    @property
    def server_gen_ms(self) -> float | None:
        if "predicted_ms" in self.timings:
            return float(self.timings["predicted_ms"])
        return None

    @property
    def server_prompt_ms(self) -> float | None:
        if "prompt_ms" in self.timings:
            return float(self.timings["prompt_ms"])
        return None

    @property
    def tpot_server_ms(self) -> float | None:
        """服务端 TPOT：生成耗时 / 生成 token 数（排除首 token 与排队）。"""
        n, ms = self.output_tokens, self.server_gen_ms
        if n and ms is not None:
            return ms / n
        return None

    @property
    def tpot_client_ms(self) -> float | None:
        """客户端 TPOT：首 token 之后到结束的均摊，排除首 token。"""
        n = self.output_tokens
        if not n or n <= 1 or self.ttft_ms is None or self.total_ms is None:
            return None
        return (self.total_ms - self.ttft_ms) / (n - 1)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "t_submit": self.t_submit,
            "t_first_content": self.t_first_content,
            "t_done": self.t_done,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "chunk_count": self.chunk_count,
            "finish_reason": self.finish_reason,
            "reasoning_chars": len(self.reasoning),
            "content_chars": len(self.content),
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "server_prompt_ms": self.server_prompt_ms,
            "server_gen_ms": self.server_gen_ms,
            "tpot_server_ms": self.tpot_server_ms,
            "tpot_client_ms": self.tpot_client_ms,
            "timings": self.timings,
            "usage": self.usage,
        }


class LlamaClient:
    def __init__(self, base_url: str, timeout_s: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # ---------- 基础请求 ----------
    def _post(self, path: str, payload: dict, timeout_s: int | None = None):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s)

    def _get(self, path: str, timeout_s: int = 10):
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        return urllib.request.urlopen(req, timeout=timeout_s)

    # ---------- 健康与就绪 ----------
    def health(self, timeout_s: float = 5.0) -> dict | None:
        try:
            with self._get("/health", timeout_s=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def is_ready(self) -> bool:
        """就绪判定：/health 返回 ok，且模型已加载。"""
        h = self.health()
        return bool(h) and h.get("status") in ("ok", "no slot available")

    def slots(self) -> list | None:
        try:
            with self._get("/slots", timeout_s=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def props(self) -> dict | None:
        try:
            with self._get("/props", timeout_s=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def metrics(self) -> str | None:
        try:
            with self._get("/metrics", timeout_s=10) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None

    # ---------- token 计数 ----------
    def apply_template(self, messages: list[dict]) -> str | None:
        """用服务端聊天模板渲染消息，得到实际送入模型的提示词。"""
        try:
            with self._post("/apply-template", {"messages": messages}, timeout_s=30) as resp:
                return json.loads(resp.read().decode("utf-8")).get("prompt")
        except Exception:
            return None

    def tokenize(self, content: str) -> list[int] | None:
        try:
            with self._post("/tokenize", {"content": content}, timeout_s=60) as resp:
                return json.loads(resp.read().decode("utf-8")).get("tokens")
        except Exception:
            return None

    def count_input_tokens(self, messages: list[dict]) -> int | None:
        """聊天模板处理后的实际输入 token 数（用于报告与容量核对）。"""
        prompt = self.apply_template(messages)
        if prompt is None:
            return None
        toks = self.tokenize(prompt)
        return len(toks) if toks is not None else None

    # ---------- 流式生成 ----------
    def chat_stream(self, messages: list[dict], params: GenParams) -> StreamResult:
        """流式对话补全，采集时间点与 token 数。"""
        res = StreamResult()
        payload = {
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "seed": params.seed,
            "stream": True,
            **params.template_kwargs(),
        }
        res.t_submit = _now_iso()
        t0 = time.perf_counter()
        parts: list[str] = []
        reason_parts: list[str] = []

        try:
            resp = self._post("/v1/chat/completions", payload)
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # 服务端权威数据可能出现在任意（通常是最后）数据块
                    if chunk.get("timings"):
                        res.timings = chunk["timings"]
                    if chunk.get("usage"):
                        res.usage = chunk["usage"]

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        thinking = delta.get("reasoning_content")
                        if piece or thinking:
                            if res.ttft_ms is None:
                                # 首个有效生成内容：正文或思考片段都算生成开始
                                res.ttft_ms = (time.perf_counter() - t0) * 1000.0
                                res.t_first_content = _now_iso()
                            if piece:
                                parts.append(piece)
                            if thinking:
                                reason_parts.append(thinking)
                        if choice.get("finish_reason"):
                            res.finish_reason = choice["finish_reason"]

            res.content = "".join(parts)
            res.reasoning = "".join(reason_parts)
            res.chunk_count = len(parts)
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.t_done = _now_iso()
            res.ok = True

        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            res.error_type = "http_error"
            res.error_message = f"HTTP {exc.code}: {body or exc.reason}"
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.t_done = _now_iso()
        except (socket.timeout, TimeoutError):
            res.error_type = "timeout"
            res.error_message = f"请求超过 {self.timeout_s}s 未完成"
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.t_done = _now_iso()
        except urllib.error.URLError as exc:
            res.error_type = "connection_error"
            res.error_message = str(exc.reason)
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.t_done = _now_iso()
        except Exception as exc:  # noqa: BLE001 - 记录任何异常，保证不中断整轮测试
            res.error_type = "unexpected"
            res.error_message = f"{type(exc).__name__}: {exc}"
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.t_done = _now_iso()

        return res

    def chat_once(self, messages: list[dict], params: GenParams) -> StreamResult:
        """非流式调用，用于对照或短请求。"""
        res = StreamResult()
        payload = {
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "seed": params.seed,
            "stream": False,
            **params.template_kwargs(),
        }
        res.t_submit = _now_iso()
        t0 = time.perf_counter()
        try:
            with self._post("/v1/chat/completions", payload) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            res.timings = body.get("timings") or {}
            res.usage = body.get("usage") or {}
            choices = body.get("choices") or [{}]
            res.content = (choices[0].get("message") or {}).get("content", "")
            res.finish_reason = choices[0].get("finish_reason")
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.ttft_ms = res.total_ms
            res.t_first_content = res.t_done = _now_iso()
            res.ok = True
        except Exception as exc:  # noqa: BLE001
            res.error_type = type(exc).__name__
            res.error_message = str(exc)
            res.total_ms = (time.perf_counter() - t0) * 1000.0
            res.t_done = _now_iso()
        return res
