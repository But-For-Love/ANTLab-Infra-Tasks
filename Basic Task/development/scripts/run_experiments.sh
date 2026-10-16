#!/usr/bin/env bash
# ============================================================
# 一键重跑全部实验（在 WSL 的 Ubuntu 24.04 中执行）
#
#   bash scripts/run_experiments.sh
#
# 顺序：
#   1. 对比矩阵：baseline 与 opt_final × 单并发/多并发 × 3 轮
#   2. 归因实验：opt_fa、opt_kv（多并发），用于分离 FA 与 KV 量化的作用
#   3. 输出质量采集：10 个规定任务的完整输出
#   4. 异常恢复测试
#   5. KV 容量日志证据
#   6. 容量探测：逐步加长输入，定位容量上限
#   7. 环境与模型校验值
#   8. 输出效果检查 + 报告数据表生成
# ============================================================
set -uo pipefail

DEV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEV" || exit 1
mkdir -p logs
export LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/root/llama.cpp/build/bin}"

log() { echo; echo "==================== $* ===================="; }
cleanup() { pkill -f llama-server 2>/dev/null; sleep 2; }

cleanup

log "1. 对比矩阵 baseline vs opt_final"
python3 run.py matrix \
  baseline:bench_c1 baseline:bench_c4 \
  opt_final:bench_c1 opt_final:bench_c4 2>&1 | tee logs/exp1_matrix.log

log "2. 归因实验 opt_fa / opt_kv（多并发）"
python3 run.py matrix opt_fa:bench_c4 opt_kv:bench_c4 2>&1 | tee logs/exp2_attribution.log

log "3. 输出质量采集（10 个规定任务）"
python3 run.py quality opt_final 2>&1 | tee logs/exp3_quality.log

log "4. 异常恢复测试"
python3 run.py recover opt_final 2>&1 | tee logs/exp4_recovery.log

cleanup

log "5. KV 容量日志证据"
python3 run.py kvlog 2>&1 | tee logs/exp5_kvlog.log

log "6. 容量探测"
python3 run.py capacity baseline 2>&1 | tee logs/exp6_capacity_baseline.log
python3 run.py capacity opt_final 2>&1 | tee logs/exp7_capacity_opt_final.log

log "7. 环境与模型校验值"
python3 run.py env 2>&1 | tee logs/exp8_env.log

log "8. 输出效果检查与报告数据表"
QUALITY_RUN="$(ls -t results/outputs 2>/dev/null | head -1)"
if [ -n "${QUALITY_RUN:-}" ]; then
  python3 run.py check "$QUALITY_RUN" 2>&1 | tee logs/exp9_check.log
fi
python3 run.py tables 2>&1 | tee logs/exp10_tables.log

cleanup
echo
echo "#################### 全部实验完成 ####################"
echo "报告数据表： results/report_tables.md"
echo "模型输出：   results/outputs/${QUALITY_RUN:-<run_id>}/"
