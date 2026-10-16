"""前端展示（零依赖，标准库 http.server）。

功能
----
- 服务状态：健康检查、slot 数、每 slot 上下文、KV 配置与显存
- 测试配置：服务配置与固定负载定义
- 请求执行状态：逐请求明细（提交/首 token/完成、状态、错误）
- 模型输出：按提示词编号查看输出原文
- 性能统计：TTFT / TPOT / 吞吐 / P50 / P95 / 成功率
- 错误记录：失败、超时、取消与重试关联
- 基线与优化对比：按服务配置聚合的对比表

启动：python3 run.py web [config_id] [port]
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import CONFIGS_DIR, DEV_ROOT, RESULTS_DIR, load_service_config
from .llm_client import LlamaClient
from .store import read_jsonl

INDEX_PATH = RESULTS_DIR / "index.json"
RUNS_DIR = RESULTS_DIR / "runs"
OUTPUTS_DIR = RESULTS_DIR / "outputs"


def _json_file(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ANTLab Infra · 本地推理服务监控</title>
<style>
 :root{--bg:#0f1115;--card:#181b22;--line:#262b36;--fg:#e6e9ef;--dim:#98a2b3;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
 header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;
        align-items:baseline;gap:14px;flex-wrap:wrap}
 header h1{font-size:17px;margin:0;font-weight:600}
 header .sub{color:var(--dim);font-size:12px}
 main{padding:20px 26px;max-width:1400px;margin:0 auto}
 .grid{display:grid;gap:16px}
 .g4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
 .g2{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
 .card h2{font-size:13px;margin:0 0 12px;color:var(--dim);font-weight:600;
          text-transform:uppercase;letter-spacing:.4px}
 .kv{display:flex;justify-content:space-between;padding:4px 0;font-size:13px}
 .kv span:last-child{color:var(--acc);font-variant-numeric:tabular-nums}
 .big{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
 .dim{color:var(--dim)} .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
 table{width:100%;border-collapse:collapse;font-size:12.5px}
 th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);
       white-space:nowrap}
 th{color:var(--dim);font-weight:600;font-size:11.5px;text-transform:uppercase}
 tbody tr:hover{background:#1f232c;cursor:pointer}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 pre{background:#0b0d11;border:1px solid var(--line);border-radius:8px;padding:14px;
     overflow:auto;max-height:460px;white-space:pre-wrap;word-break:break-word;
     font-size:12.5px;line-height:1.55}
 .pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;
       border:1px solid var(--line)}
 .pill.success{color:var(--ok);border-color:#1f6f34}
 .pill.failed,.pill.timeout{color:var(--bad);border-color:#7d2b26}
 .pill.cancelled{color:var(--warn);border-color:#6b5314}
 .muted{color:var(--dim);font-size:12px}
 select,button{background:#11141a;color:var(--fg);border:1px solid var(--line);
   border-radius:7px;padding:6px 10px;font-size:12.5px;font-family:inherit}
 button{cursor:pointer} button:hover{border-color:var(--acc)}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
 .note{font-size:12px;color:var(--dim);margin-top:8px;line-height:1.7}
</style>
</head>
<body>
<header>
  <h1>ANTLab Infra · 本地推理服务与性能测试</h1>
  <span class="sub" id="svcinfo">加载中…</span>
  <span style="flex:1"></span>
  <span class="sub" id="clock"></span>
</header>
<main>
  <div class="grid g4" id="statusCards"></div>

  <div class="card" style="margin-top:16px">
    <h2>基线与优化对比（按服务配置聚合多轮结果）</h2>
    <div id="compare"></div>
    <div class="note" id="comparenote"></div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>运行记录</h2>
    <div class="row">
      <label class="muted">运行编号</label>
      <select id="runSel"></select>
      <button onclick="loadRun()">加载</button>
      <button onclick="location.reload()">刷新索引</button>
      <span class="muted" id="runmeta"></span>
    </div>
    <div id="runSummary" class="muted">选择一次运行查看详情。</div>
  </div>

  <div class="grid g2" style="margin-top:16px">
    <div class="card">
      <h2>逐请求明细</h2>
      <div style="max-height:520px;overflow:auto"><table id="reqTable"></table></div>
    </div>
    <div class="card">
      <h2>错误记录</h2>
      <div id="errList" class="muted">—</div>
      <h2 style="margin-top:18px">GPU 监控</h2>
      <div id="gpuBox" class="muted">—</div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>模型输出</h2>
    <div class="row">
      <label class="muted">提示词编号</label>
      <select id="outSel"></select>
      <button onclick="loadOutput()">查看输出</button>
      <span class="muted" id="outMeta"></span>
    </div>
    <pre id="outBox">选择一项查看模型输出原文。</pre>
  </div>
</main>
<script>
const fmt=(v,d=1)=>v==null?'—':Number(v).toFixed(d);
let IDX={runs:[]}, CUR=null, RUNS={};

async function j(u){const r=await fetch(u);return r.ok?r.json():null;}

function setClock(){document.getElementById('clock').textContent=new Date().toLocaleString('zh-CN');}
setInterval(setClock,1000);setClock();

async function loadService(){
  const s=await j('/api/service');
  const el=document.getElementById('statusCards');
  if(!s||!s.health){el.innerHTML=`<div class="card"><h2>服务状态</h2>
    <div class="big bad">未运行</div><div class="note">llama-server 未响应 /health</div></div>`;
    document.getElementById('svcinfo').textContent='服务未运行';return;}
  const h=s.health||{},p=s.props||{},sl=s.slots||[];
  document.getElementById('svcinfo').textContent=
    `模型 ${p.model_path?p.model_path.split('/').pop():'—'} · slot ${sl.length||p.total_slots||'—'} · 每 slot ctx ${sl[0]?sl[0].n_ctx:'—'}`;
  el.innerHTML=`
   <div class="card"><h2>服务状态</h2><div class="big ok">就绪</div>
     <div class="note">/health = ${h.status||'—'}</div></div>
   <div class="card"><h2>并发 slot</h2><div class="big">${sl.length||p.total_slots||'—'}</div>
     <div class="kv"><span>每 slot 上下文</span><span>${sl[0]?sl[0].n_ctx:'—'}</span></div>
     <div class="kv"><span>总上下文</span><span>${s.n_ctx_total??'—'}</span></div>
     <div class="kv"><span>KV 共享</span><span>${s.kv_unified??'—'}</span></div></div>
   <div class="card"><h2>KV 容量（token）</h2>
     <div class="big">${s.kv_tokens??'—'}</div>
     <div class="kv"><span>KV 类型 K/V</span><span>${s.kv_type??'—'}</span></div>
     <div class="kv"><span>KV 显存</span><span>${s.kv_mib??'—'} MiB</span></div>
     <div class="kv"><span>主机 prompt cache</span><span>${s.cache_ram??'—'}</span></div></div>
   <div class="card"><h2>显存</h2>
     <div class="big">${s.gpu_mem_used??'—'}<span class="muted" style="font-size:14px"> / ${s.gpu_mem_total??'—'} MiB</span></div>
     <div class="kv"><span>GPU</span><span style="color:var(--dim)">${s.gpu_name??'—'}</span></div>
     <div class="kv"><span>温度 / 功耗</span><span>${s.gpu_temp??'—'}°C / ${s.gpu_power??'—'}W</span></div></div>`;
}

async function loadIndex(){
  IDX=await j('/api/index')||{runs:[]};
  const sel=document.getElementById('runSel');sel.innerHTML='';
  IDX.runs.slice().reverse().forEach(r=>{
    const o=document.createElement('option');o.value=r.run_id;
    o.textContent=`${r.run_id}  ·  ${r.service_config||''}  ·  并发 ${r.concurrency??'—'}  ·  成功率 ${r.success_rate==null?'—':(r.success_rate*100).toFixed(0)+'%'}`;
    sel.appendChild(o);});
  renderCompare();
}

async function loadRun(){
  const id=document.getElementById('runSel').value;if(!id)return;
  CUR=id;
  const d=await j('/api/run/'+encodeURIComponent(id));
  if(!d){document.getElementById('runSummary').textContent='无法读取该运行';return;}
  const o=d.overall||{};
  document.getElementById('runmeta').textContent=
    `${d.service_config} · ${d.load_id} · 并发 ${d.concurrency} · ${d.repeats} 轮 · 输出上限 ${d.output_limit}`;
  document.getElementById('runSummary').innerHTML=`
    <div class="grid g4" style="margin-top:10px">
      <div><div class="muted">成功率</div><div class="big">${o.success??'—'}/${o.total_requests??'—'}</div></div>
      <div><div class="muted">TTFT 中位 / P95</div><div class="big">${fmt(o.ttft_ms?.median,0)} / ${fmt(o.ttft_ms?.p95,0)}<span class="muted" style="font-size:13px"> ms</span></div></div>
      <div><div class="muted">TPOT（服务端）</div><div class="big">${fmt(o.tpot_server_ms?.mean)}<span class="muted" style="font-size:13px"> ms</span></div></div>
      <div><div class="muted">解码吞吐</div><div class="big">${fmt(o.throughput?.decode_tokens_per_s)}<span class="muted" style="font-size:13px"> tok/s</span></div></div>
      <div><div class="muted">完成耗时 中位 / P95</div><div class="big">${fmt(o.total_ms?.median,0)} / ${fmt(o.total_ms?.p95,0)}<span class="muted" style="font-size:13px"> ms</span></div></div>
      <div><div class="muted">输出 token</div><div class="big">${o.tokens?.output_total??'—'}</div></div>
      <div><div class="muted">重试请求</div><div class="big">${o.retried_requests??0}</div></div>
      <div><div class="muted">墙钟</div><div class="big">${fmt(o.wall_clock_s)}<span class="muted" style="font-size:13px"> s</span></div></div>
    </div>`;
  renderReqs(d.requests||[]);renderErrs(d.requests||[]);renderGpu(d.gpu||{});
  const sel=document.getElementById('outSel');sel.innerHTML='';
  (d.outputs||[]).forEach(p=>{const o=document.createElement('option');
    o.value=p.prompt_id;o.textContent=p.prompt_id;sel.appendChild(o);});
  document.getElementById('outBox').textContent='选择一项查看模型输出原文。';
}

function renderReqs(rs){
  const t=document.getElementById('reqTable');
  let h=`<thead><tr><th>请求 ID</th><th>任务</th><th>轮</th><th>状态</th>
    <th class="num">输入 tok</th><th class="num">输出 tok</th>
    <th class="num">TTFT ms</th><th class="num">TPOT ms</th><th class="num">完成 ms</th>
    <th class="num">重试</th></tr></thead><tbody>`;
  rs.forEach(r=>{h+=`<tr><td>${r.request_id||'—'}</td><td>${r.prompt_id||''}</td>
    <td class="num">${r.round??''}</td><td><span class="pill ${r.status}">${r.status}</span></td>
    <td class="num">${r.prompt_tokens??'—'}</td><td class="num">${r.output_tokens??'—'}</td>
    <td class="num">${fmt(r.ttft_ms,0)}</td><td class="num">${fmt(r.tpot_server_ms)}</td>
    <td class="num">${fmt(r.total_ms,0)}</td><td class="num">${r.retry_count??0}</td></tr>`;});
  t.innerHTML=h+'</tbody>';
}

function renderErrs(rs){
  const bad=rs.filter(r=>r.status!=='success');
  const el=document.getElementById('errList');
  if(!bad.length){el.innerHTML='<span class="ok">本次运行无失败请求</span>';return;}
  el.innerHTML=bad.map(r=>`<div style="padding:6px 0;border-bottom:1px solid var(--line)">
    <span class="pill ${r.status}">${r.status}</span>
    <b>${r.prompt_id}</b> <span class="muted">${r.request_id}</span>
    <div class="muted" style="margin-top:3px">${r.error_type||''} ${(r.error_message||'').slice(0,220)}</div>
    ${r.retry_count?`<div class="muted">已重试 ${r.retry_count} 次，按逻辑请求计数一次</div>`:''}
  </div>`).join('');
}

function renderGpu(g){
  const el=document.getElementById('gpuBox');
  if(!g||!g.samples){el.textContent='无 GPU 采样数据';return;}
  el.innerHTML=`<div class="kv"><span>采样点数</span><span>${g.samples}</span></div>
    <div class="kv"><span>显存峰值</span><span>${g.mem_used_mib_max??'—'} MiB</span></div>
    <div class="kv"><span>显存基线</span><span>${g.mem_used_mib_min??'—'} MiB</span></div>
    <div class="kv"><span>利用率均值</span><span>${fmt(g.util_pct_mean)}%</span></div>
    <div class="kv"><span>温度峰值</span><span>${g.temp_c_max??'—'} °C</span></div>
    <div class="kv"><span>功耗峰值</span><span>${fmt(g.power_w_max,0)} W</span></div>
    <div class="muted" style="margin-top:6px">${g.csv_path||''}</div>`;
}

async function loadOutput(){
  const pid=document.getElementById('outSel').value;if(!pid||!CUR)return;
  const d=await j(`/api/output/${encodeURIComponent(CUR)}/${encodeURIComponent(pid)}`);
  document.getElementById('outBox').textContent=d&&d.content?d.content:'（无输出）';
  document.getElementById('outMeta').textContent=d&&d.meta?
    `输入 ${d.meta.prompt_tokens??'—'} tok · 输出 ${d.meta.output_tokens??'—'} tok · TTFT ${fmt(d.meta.ttft_ms,0)} ms`:'';
}

function renderCompare(){
  // 按 (service_config, load_id) 聚合，形成基线/优化对比
  const rows=IDX.runs.filter(r=>r.summary_path&&r.concurrency!=null);
  if(!rows.length){document.getElementById('compare').innerHTML='<span class="muted">暂无运行数据</span>';return;}
  const groups={};
  rows.forEach(r=>{const k=r.load_id+'|'+r.concurrency;
    (groups[k]=groups[k]||[]).push(r);});
  let h=`<table><thead><tr><th>负载</th><th>并发</th><th>服务配置</th>
    <th class="num">运行数</th><th class="num">成功率</th><th class="num">TTFT 中位 ms</th>
    <th class="num">TTFT P95 ms</th><th class="num">解码 tok/s</th>
    <th class="num">输出 tok</th></tr></thead><tbody>`;
  Object.keys(groups).sort().forEach(k=>{
    const [load,conc]=k.split('|');
    const byCfg={};groups[k].forEach(r=>(byCfg[r.service_config]=byCfg[r.service_config]||[]).push(r));
    Object.keys(byCfg).sort().forEach(cfg=>{
      const rs=byCfg[cfg];
      const avg=f=>{const v=rs.map(f).filter(x=>x!=null);
        return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;};
      const best=cfg==='baseline'?'':' style="background:#12261a"';
      h+=`<tr${best}><td>${load}</td><td class="num">${conc}</td><td><b>${cfg}</b></td>
        <td class="num">${rs.length}</td>
        <td class="num">${fmt(avg(r=>r.success_rate)*100,0)}%</td>
        <td class="num">${fmt(avg(r=>r.ttft_p50_ms),0)}</td>
        <td class="num">${fmt(avg(r=>r.ttft_p95_ms),0)}</td>
        <td class="num">${fmt(avg(r=>r.decode_tokens_per_s))}</td>
        <td class="num">${fmt(avg(r=>r.output_tokens),0)}</td></tr>`;});
  });
  document.getElementById('compare').innerHTML=h+'</tbody></table>';
  document.getElementById('comparenote').textContent=
    '绿色行为优化配置。数值为该 (负载, 并发) 下多次运行的平均值；点击下方“运行记录”可查看每轮明细与原始数据路径。';
}

loadService();loadIndex();setInterval(loadService,15000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    service_config_id = "opt_final"

    def log_message(self, fmt, *args):  # 安静模式
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    # ---------------- 服务状态 ----------------
    def _service_status(self) -> dict:
        cfg = load_service_config(self.service_config_id)
        client = LlamaClient(cfg.base_url, timeout_s=5)
        health = client.health()
        props = client.props() or {}
        slots = client.slots() or []

        kv_tokens = kv_type = kv_mib = kv_unified = None
        log = RESULTS_DIR / "service_logs" / f"{cfg.config_id}.log"
        if log.is_file():
            text = log.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "kv_unified" in line:
                    try:
                        kv_unified = line.split("kv_unified = ")[1].strip().strip("'")
                    except Exception:
                        pass
                if "KV self size" in line or "KV buffer size" in line:
                    kv_mib = line.split("=")[-1].strip()
            import re
            m = re.findall(r"n_ctx_slot = (\d+)", text)
            if m:
                kv_tokens = int(m[-1]) * (len(slots) or props.get("total_slots") or 1)

        # KV 精度、字段与显存优先取 kvlog 的结构化证据
        kvp = RESULTS_DIR / "kv_capacity.json"
        if kvp.is_file():
            try:
                for rec in json.loads(kvp.read_text(encoding="utf-8")):
                    if rec.get("config_id") == cfg.config_id:
                        kv_type = f"{rec.get('kv_type_k')} / {rec.get('kv_type_v')}"
                        kv_mib = rec.get("kv_total_mib")
                        kv_tokens = rec.get("total_tokens") or kv_tokens
                        kv_unified = rec.get("kv_unified", kv_unified)
                        break
            except Exception:
                pass

        extra = self._gpu_snapshot()
        return {
            "health": health, "props": props, "slots": slots,
            "kv_tokens": kv_tokens, "kv_type": kv_type, "kv_mib": kv_mib,
            "kv_unified": kv_unified,
            "n_ctx_total": (slots[0].get("n_ctx") * len(slots)) if slots else None,
            "cache_ram": "已关闭（--cache-ram 0）",
            **extra,
        }

    @staticmethod
    def _gpu_snapshot() -> dict:
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8, check=False).stdout.strip()
            if out:
                p = [x.strip() for x in out.splitlines()[0].split(",")]
                return {"gpu_name": p[0], "gpu_mem_used": p[1], "gpu_mem_total": p[2],
                        "gpu_temp": p[3], "gpu_power": p[4]}
        except Exception:
            pass
        return {}

    # ---------------- 运行数据 ----------------
    def _run_detail(self, run_id: str) -> dict | None:
        d = RUNS_DIR / run_id
        if not d.is_dir():
            return None
        summary = _json_file(d / "summary.json", {}) or {}
        reqs = read_jsonl(d / "requests.jsonl")
        # 逐请求精简，避免响应过大
        slim = []
        for r in reqs:
            slim.append({k: r.get(k) for k in (
                "request_id", "prompt_id", "round", "status", "prompt_tokens",
                "cached_tokens", "output_tokens", "ttft_ms", "tpot_server_ms",
                "total_ms", "retry_count", "error_type", "error_message",
                "submit_seq", "attempt_count", "reasoning_chars")})
        outs = []
        od = OUTPUTS_DIR / run_id
        if od.is_dir():
            outs = [{"prompt_id": p.stem} for p in sorted(od.glob("*.md"))]
        summary["requests"] = slim
        summary["outputs"] = outs
        return summary

    def _output(self, run_id: str, prompt_id: str) -> dict:
        p = OUTPUTS_DIR / run_id / f"{prompt_id}.md"
        if not p.is_file():
            return {"content": None}
        text = p.read_text(encoding="utf-8")
        meta = {}
        if text.startswith("<!--"):
            head, _, body = text.partition("-->")
            text = body.lstrip()
            for kv in head.replace("<!--", "").split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    meta[k] = v
        return {"content": text, "meta": meta}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/index":
            idx = _json_file(INDEX_PATH, {"runs": []})
            self._json(idx or {"runs": []})
        elif path == "/api/service":
            try:
                self._json(self._service_status())
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, 500)
        elif path.startswith("/api/run/"):
            run_id = path[len("/api/run/"):]
            d = self._run_detail(run_id)
            self._json(d) if d else self._json({"error": "not found"}, 404)
        elif path.startswith("/api/output/"):
            parts = path[len("/api/output/"):].split("/")
            if len(parts) == 2:
                self._json(self._output(parts[0], parts[1]))
            else:
                self._json({"error": "bad request"}, 400)
        else:
            self._send(b"not found", "text/plain; charset=utf-8", 404)


def serve(config_id: str = "opt_final", host: str = "127.0.0.1", port: int = 8000) -> None:
    Handler.service_config_id = config_id
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"前端已启动： http://{host}:{port}/  （服务配置 {config_id}）", flush=True)
    httpd.serve_forever()
