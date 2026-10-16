# 测试结果

本目录保存全部原始记录。所有汇总数字都可以由 `requests.jsonl` 重新计算：
`python3 run.py tables` 会从原始记录生成 [`report_tables.md`](report_tables.md)。

## 目录结构

```text
results/
├─ index.json                # 全部运行的索引（前端与报告统一入口）
├─ report_tables.md          # 自动生成的报告数据表
├─ env.json                  # 环境与模型校验值（含模型 SHA256）
├─ kv_capacity.json          # 各服务配置的 KV 容量与显存日志证据
├─ capacity_<配置>.json      # 容量边界验证：逐步加长输入，定位实测上限
├─ matrix_meta.json          # 对比矩阵的硬件/模型/任务清单元数据
├─ runs/
│  └─ <run_id>/
│     ├─ summary.json        # 汇总：overall / rounds / aggregate / gpu
│     └─ requests.jsonl      # 逐请求原始记录
├─ outputs/<run_id>/<编号>.md # 模型输出原文（含一行元数据注释）
├─ checks/<run_id>.json      # 输出效果检查结果
├─ gpu/<run_id>.csv          # GPU 监控采样
└─ service_logs/*.log        # llama-server 启动与运行日志
```

## `requests.jsonl` 字段说明

每行是一个**逻辑请求**的 JSON 记录。

### 标识与追溯

| 字段 | 含义 |
| --- | --- |
| `run_id` | 运行编号，与 `results/runs/<run_id>/` 目录对应 |
| `request_id` | 逻辑请求编号，格式 `<run_id>-<提示词编号>-<序号>`；重试共用同一编号 |
| `submit_seq` | 该运行内的提交顺序号（1 起） |
| `round` | 重复轮次（1 起） |
| `prompt_id` / `prompt_title` | 规定任务编号与标题 |
| `task_kind` | `纯提示词` / `文档` / `源文件` |
| `file_name` / `file_sha256` | 关联文件名称与 SHA256（纯提示词任务为 `null`） |
| `service_config` | 服务配置编号（`baseline` / `opt_fa` / `opt_kv` / `opt_final`） |
| `concurrency` / `output_limit` | 本次运行的并发数与输出上限 |

### 时间点与时延

| 字段 | 含义 |
| --- | --- |
| `t_submit` | 客户端提交时刻（本地时区 ISO 8601，毫秒） |
| `t_first_content` | 首个非空生成内容时刻 |
| `t_done` | 流结束时刻 |
| `ttft_ms` | 提交 → 首个有效生成内容，含排队与传输 |
| `total_ms` | 提交 → 流结束 |
| `tpot_server_ms` | 服务端 TPOT = `predicted_ms / predicted_n`，仅生成阶段，排除首 token |
| `tpot_client_ms` | 客户端 TPOT = `(total_ms - ttft_ms) / (output_tokens - 1)` |
| `server_prompt_ms` / `server_gen_ms` | 服务端提示词处理耗时 / 生成耗时 |
| `planned_submit_at` | 计划提交时刻（用于观察到达偏差） |

### token 计数

| 字段 | 含义 |
| --- | --- |
| `prompt_tokens` | 聊天模板处理后的输入 token 数 = `prompt_n + cache_n` |
| `cached_tokens` | 被缓存复用的输入 token 数（服务端 `timings.cache_n`） |
| `output_tokens` | 生成 token 数（服务端 `timings.predicted_n`，**不数流式块**） |
| `chunk_count` | 流式数据块数，仅供对照，不是 token 数 |
| `timings` / `usage` | 服务端返回的原始计量字段，便于复核 |
| `reasoning_chars` / `content_chars` | 思考内容与正文字符数 |
| `content` / `reasoning` | 输出正文与思考内容原文 |

### 状态与重试

| 字段 | 含义 |
| --- | --- |
| `status` | `success` / `failed` / `timeout` / `cancelled` |
| `error_type` / `error_message` | 错误类型与信息 |
| `finish_reason` | `stop` / `length`（`length` 表示达到输出上限被截断） |
| `retry_count` | 重试次数（`attempt_count - 1`） |
| `attempts` | 每次尝试的摘要（尝试序号、状态、错误、TTFT、完成耗时） |

**计数口径**：`total_requests` 为逻辑请求数；重试在 `attempts` 中体现，
汇总时按逻辑请求计数一次，`retried_requests` 单独统计，避免重复计数。

## `summary.json` 结构

| 键 | 含义 |
| --- | --- |
| `overall` | 全部轮次合并后的汇总（成功率、TTFT/完成耗时均值与 P50/P95、TPOT、吞吐） |
| `rounds` | 每一轮的汇总，用于观察波动 |
| `aggregate` | 多轮聚合：各指标的 mean / min / max / `spread_pct` |
| `gpu` | 运行期间的 GPU 采样统计（显存峰值/基线、利用率均值、温度与功耗峰值） |

## `kv_capacity.json` 字段说明

`n_slots`、`n_ctx_slot`、`kv_unified` 来自启动日志
`initializing, n_slots = …, n_ctx_slot = …, kv_unified = '…'`；
`kv_total_mib`、`kv_type_k/v`、`kv_seqs` 来自 `llama_kv_cache: size = … MiB (… cells, … layers, …/… seqs), K (…): … MiB, V (…): … MiB`；
`cuda_model_mib` / `cuda_compute_mib` 来自 `load_tensors` 与 `sched_reserve` 行；
`bytes_per_token_*` 与 `kv_mib_computed_*` 由 GGUF 架构参数独立推算，用于与日志实测交叉验证。

临时模型、构建产物与环境放在被 `.gitignore` 管理的位置；本目录保留可复核的原始记录。
