#!/usr/bin/env bash
# ============================================================================
# 剩余 3 组：R1 / R3 / R4，**串行**跑，只用 GPU 0-3
# ----------------------------------------------------------------------------
# 用户要求只占 0-3，所以不再两组并行。串行反而更干净：
#   没有邻居抢 CPU/IO，稳态波动只来自 batch 组成本身。
# R4 预期 OOM（MBS8 + 不开 CCE 的 logits），OOM 不终止队列。
# ============================================================================
set -uo pipefail
BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b"
Q="${BASE}/results/queue.log"

run() {
  local cfg="$1" port="$2"
  echo "[$(date +%H:%M:%S)] 开始 ${cfg}  GPU=0,1,2,3（串行）" >> "$Q"
  MASTER_PORT="$port" bash "${BASE}/scripts/run_mfu_4gpu.sh" "$cfg" "0,1,2,3" \
    > "${BASE}/results/${cfg}.launch.log" 2>&1
  echo "[$(date +%H:%M:%S)] 结束 ${cfg}  exit=$?" >> "$Q"
}

echo "===== 串行队列(仅 GPU 0-3)启动 $(date) =====" >> "$Q"
run r1_mbs1_z3_gpugc 29761
run r3_mbs4_z2       29762
run r4_mbs8_z2       29763
echo "===== 串行队列结束 $(date) =====" >> "$Q"
