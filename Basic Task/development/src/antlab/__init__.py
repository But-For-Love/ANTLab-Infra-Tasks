"""ANTLab Infra 基础阶段：本地 GPU 推理服务与性能测试系统。

模块划分
--------
config    : 路径与配置加载
prompts   : 规定测试任务的解析与输入组织
llm_client: llama-server 推理客户端（流式，采集 TTFT 与 token 数）
metrics   : TTFT / TPOT / 吞吐 / P95 等指标计算
store     : 运行编号与结果落盘（JSONL 原始记录 + JSON 汇总）
service   : llama-server 进程管理、健康检查、重启与就绪确认
runner    : 固定负载执行器（重复次数、并发数、重试）
web       : 前端展示
"""

__version__ = "1.0.0"
