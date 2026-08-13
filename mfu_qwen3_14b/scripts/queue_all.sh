#!/usr/bin/env bash
# ============================================================================
# 14B 4卡 MFU 实验队列：5 组，8 张卡两两并行跑，共 3 波
# ----------------------------------------------------------------------------
#   波1: R0(GPU 0-3, port 29751)  ‖  R2(GPU 4-7, port 29752)
#   波2: R1(GPU 0-3)              ‖  R3(GPU 4-7)
#   波3: R4(GPU 0-3)      ← 预期 OOM，单独跑，OOM 不阻塞后续统计
#
# 并行安全性：GPU 独占（每组各 4 张，不重叠），CPU/IO 共享。
#   8B 那五组并行实测稳态 cv 全 <0.2%，并行未造成可见干扰；
#   本轮同样会在汇总时做 cv 体检，cv >2% 的组要串行重跑确认。
#
# 每组内部仍走 run_mfu_4gpu.sh 的 GPU 自检（占用 >2GB 拒绝启动，不抢占）。
# 某组失败（含 OOM）不终止队列，继续跑下一组。
# ============================================================================
set -uo pipefail
BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b"
Q="${BASE}/results/queue.log"
mkdir -p "${BASE}/results"

run() {  # run <cfg> <gpus> <port>
  local cfg="$1" gpus="$2" port="$3"
  echo "[$(date +%H:%M:%S)] 开始 ${cfg}  GPU=${gpus}" >> "$Q"
  MASTER_PORT="$port" bash "${BASE}/scripts/run_mfu_4gpu.sh" "$cfg" "$gpus" \
    > "${BASE}/results/${cfg}.launch.log" 2>&1
  echo "[$(date +%H:%M:%S)] 结束 ${cfg}  exit=$?" >> "$Q"
}

echo "===== 队列启动 $(date) =====" >> "$Q"

# ---- 波1 ----
run r0_mbs1_z3_unsloth "0,1,2,3" 29751 &
P1=$!
run r2_mbs2_z2         "4,5,6,7" 29752 &
P2=$!
wait $P1 $P2

# ---- 波2 ----
run r1_mbs1_z3_gpugc   "0,1,2,3" 29753 &
P3=$!
run r3_mbs4_z2         "4,5,6,7" 29754 &
P4=$!
wait $P3 $P4

# ---- 波3：预期 OOM 的一组，单独跑，独占以排除显存竞争这个混淆 ----
run r4_mbs8_z2         "0,1,2,3" 29755

echo "===== 队列结束 $(date) =====" >> "$Q"
