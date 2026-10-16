# ANTLab Infra 基础阶段 · 开发说明

本目录是基础阶段考核的完整开发区域：本地 GPU 推理服务、性能测试与结果存储、
前端展示、报错记录与服务恢复。

---

## 1. 项目与应用场景

**服务对象**：澄禾工作室（虚构），共 8 人，同时在线 1—4 人。
**业务需求**：办公文档处理（邮件、会议纪要、制度与条款问答）与辅助编程
（代码审查、缺陷修复、脚本编写）。
**负载构成**：10 个规定任务中，办公类 6 个（P01、P02、D01、D02、D03、D04），
编程类 4 个（P03、P04、C01、C02）。
**优化目标**：在保证规定任务全部可服务的前提下，降低 TTFT、提升吞吐，
并扩大 KV 容量（以 token 为单位）。

场景定义见 [`configs/scenario.json`](configs/scenario.json)。

---

## 2. 环境与模型（复现信息）

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU，12226 MiB |
| 运行环境 | WSL2 / Ubuntu 24.04 |
| 驱动 | 596.36 |
| CUDA | 13.2（`nvcc` 13.2.86） |
| 构建工具 | CMake 3.28.3 / GCC 13.3.0 |
| 推理引擎 | llama.cpp `llama-server`，源码编译，CUDA 后端，`CMAKE_CUDA_ARCHITECTURES=120` |
| llama.cpp commit | 见 `results/env.json` 的 `llama_commit` |
| 模型 | `Qwen3-8B-Q4_K_M.gguf`（Qwen 官方 GGUF，ModelScope 获取） |
| 模型 SHA256 | 见 `results/env.json` 的 `model_sha256` |
| 量化格式 | Q4_K_M |
| 关键要求 | 模型全部计算层与 KV 缓存驻留 GPU；`--cache-ram 0` 关闭主机 RAM prompt cache |

一键采集环境与模型校验值：

```bash
python3 run.py env      # 写入 results/env.json
```

---

## 3. 系统结构与模块

```
development/
├─ run.py                     # 入口（在 WSL 内运行）
├─ src/antlab/
│  ├─ config.py               # 路径、服务配置、负载配置的数据类
│  ├─ prompts.py              # 解析规定测试任务，按「系统消息→资料区→任务区」组织输入
│  ├─ llm_client.py           # llama-server 客户端：流式、TTFT 采集、token 计数
│  ├─ service.py              # llama-server 进程管理、就绪确认、重启、GPU 监控
│  ├─ runner.py               # 固定负载执行器：并发、重复轮次、重试与去重计数
│  ├─ metrics.py              # TTFT/TPOT/吞吐/P95 计算与多轮聚合
│  ├─ store.py                # 运行编号、JSONL 原始记录、汇总与索引
│  ├─ checks.py               # 输出效果检查（办公事实/条款、代码执行与边界）
│  ├─ report.py               # 从原始记录生成报告数据表
│  ├─ web.py                  # 前端展示（标准库 http.server，零依赖）
│  └─ cli.py                  # 命令行入口
├─ configs/
│  ├─ service/                # baseline / opt_fa / opt_kv / opt_final 服务配置
│  ├─ loads/                  # bench_c1 / bench_c4 / quality 固定负载
│  └─ scenario.json           # 场景定义
├─ tests/                     # 单元测试与输出检查
├─ results/                   # 原始记录、模型输出、汇总、检查与图表数据
└─ reports/                   # 书面报告
```

**输入组织**：系统消息 → 用户消息中的资料区（关联文件全文）→ 任务区。
系统消息与任务正文均取自 `Basic Task/提示词与输入文档/测试提示词.md`，
解析时记录每个关联文件的 SHA256 与实际输入 token 数。

**时间口径**（详见报告第 5 节）：

- TTFT：客户端提交 → 首个非空生成内容，含排队与传输
- TPOT（服务端）：`predicted_ms / predicted_n`，仅生成阶段，排除首 token
- TPOT（客户端）：`(total_ms - ttft_ms) / (output_tokens - 1)`
- 吞吐：生成 token 总数 / 服务端生成耗时之和；另给成功请求数 / 墙钟时间
- 完成耗时：客户端提交 → 流结束，给出中位数与 P95

**请求状态与计数口径**：

| 状态 | 含义 |
| --- | --- |
| `success` | 正常完成并取得生成内容 |
| `failed` | 收到错误响应或连接失败 |
| `timeout` | 超过请求超时上限 |
| `cancelled` | 提交前检测到服务中断，未发送 |

每个逻辑请求有唯一 `request_id`；重试记录在同一 request_id 的 `attempts` 中，
汇总按逻辑请求计数一次，`retry_count` 单独统计，避免重复计数。

---

## 4. 运行方法

全部命令在 WSL 的 Ubuntu 24.04 中执行，工作目录为本目录。

```bash
# 查看解析到的规定任务与输入规模
python3 run.py tasks

# 启动 / 停止 / 查询服务
python3 run.py serve baseline
python3 run.py status baseline
python3 run.py stop baseline

# 单次固定负载测试
python3 run.py bench baseline bench_c4

# 对比矩阵（服务配置:负载）
python3 run.py matrix baseline:bench_c1 baseline:bench_c4 \
                     opt_final:bench_c1 opt_final:bench_c4

# 10 个规定任务的完整输出采集
python3 run.py quality opt_final

# 异常恢复测试（强杀 → 检测 → 重启 → 就绪确认 → 恢复后请求）
python3 run.py recover opt_final

# KV 容量验证（逐步加长输入，定位容量上限）
python3 run.py capacity baseline
python3 run.py capacity opt_final
python3 run.py kvlog            # KV 配置的日志证据

# 环境与模型校验值
python3 run.py env

# 输出效果检查
python3 run.py check <run_id>

# 从原始记录重新生成报告数据表
python3 run.py tables

# 前端展示
python3 run.py web opt_final 8000      # http://127.0.0.1:8000/
```

一键重跑全部实验：`bash scripts/run_experiments.sh`。

---

## 5. 前端展示

`python3 run.py web opt_final 8000` 后访问 `http://127.0.0.1:8000/`，包含：

- **服务状态**：健康检查、slot 数、每 slot 上下文、KV 容量与精度、显存
- **基线与优化对比**：按（负载，并发）聚合多轮结果
- **运行记录**：逐请求明细（提交/首 token/完成、状态、重试）
- **错误记录**：失败、超时、取消请求及其错误类型
- **GPU 监控**：显存峰值、利用率、温度、功耗
- **模型输出**：按提示词编号查看输出原文

---

## 6. 报告与数据位置

| 内容 | 位置 |
| --- | --- |
| **最终报告** | [`reports/基础阶段考核报告.md`](reports/基础阶段考核报告.md) |
| 报告数据表（自动生成） | [`results/report_tables.md`](results/report_tables.md) |
| 逐请求原始记录 | `results/runs/<run_id>/requests.jsonl` |
| 单次运行汇总 | `results/runs/<run_id>/summary.json` |
| 模型输出原文 | `results/outputs/<run_id>/<prompt_id>.md` |
| 服务启动与运行日志 | `results/service_logs/*.log` |
| GPU 监控采样 | `results/gpu/<run_id>.csv` |
| 环境与模型校验值 | `results/env.json` |
| KV 容量证据 | `results/kv_capacity.json`、`results/capacity_*.json` |
| 异常恢复记录 | `results/runs/<run_id>/recovery.json` |
| 输出效果检查 | `results/checks/<run_id>.json` |

所有汇总数字均由 `python3 run.py tables` 从 `requests.jsonl` 重新计算，
报告与原始记录可相互复核。

---

## 7. 测试

```bash
python3 tests/test_units.py          # 单元测试：任务解析、输入组织、指标口径
python3 run.py check <run_id>        # 输出效果检查（含执行 C02 生成代码）
python3 run.py recover opt_final     # 异常恢复测试
```

---

## 8. AI 辅助开发说明

本项目在开发中使用 AI 辅助编码工具，使用范围与核验方式见报告第 11 节。
所有提示词解析结果、指标计算、性能数据与模型输出均由本目录的代码实际产生并保存为
原始记录，可逐条回查。
