#!/usr/bin/env bash
# ============================================================================
# A4 → A5 串行队列：严格同批量对照（拆开「批量效应」与「关GC效应」）
# ----------------------------------------------------------------------------
# 背景：A/A2 名义 GBS=32 但那是「行数」。打包后每行平均装 4.72 条真实样本，
#   实际每步喂 150.9 条，是 R3 等不打包组（32 条）的 4.72 倍。所以 A2 的
#   38.38% 里混着「批量变大」的贡献，与 R3 的对照不公平。
#
# 本队列把 GAS 从 8 降到 2 → 每步 8 行 × 4.72 = 37.7 条样本，落回 32 附近。
#   A4 = 打包 + 关GC + GAS2
#   A5 = 打包 + 开GC + GAS2
# 配上已有的 A2(关GC,GAS8) 和 A(开GC,GAS8)，构成完整 2×2：
#
#            GAS=8(151条)   GAS=2(37.7条)
#   关GC        A2 38.38%      A4 = ?
#   开GC        A  30.58%      A5 = ?
#
#   A4 vs A5  → 同批量下关GC值多少（可与 R3 公平对照的那个数）
#   A2 vs A4  → 纯批量效应（关GC条件下）
#   A  vs A5  → 纯批量效应（开GC条件下）
#
# ★ 权重落 /root 本地盘，跑完立即删。框架里 trainer.save_model() 硬编码，
#   save_strategy:"no" 关不掉它。共享盘曾被 193 GB 权重写满过。
# ============================================================================
set -uo pipefail
BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b"
Q="${BASE}/results/queue_a4a5.log"
SCRATCH="/root/scratch_saves"

run() {
  local cfg="$1" gpus="$2" port="$3"
  echo "[$(date +%H:%M:%S)] 开始 ${cfg}  GPU=${gpus}" >> "$Q"
  MASTER_PORT="$port" bash "${BASE}/scripts/run_mfu_4gpu.sh" "$cfg" "$gpus" \
    > "${BASE}/results/${cfg}.launch.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] 结束 ${cfg}  exit=${rc}" >> "$Q"
  if [ -d "${SCRATCH}/${cfg}" ]; then
    local sz=$(du -sh "${SCRATCH}/${cfg}" 2>/dev/null | cut -f1)
    rm -rf "${SCRATCH}/${cfg}" && echo "[$(date +%H:%M:%S)] 已删权重 ${cfg} (${sz})" >> "$Q"
  fi
}

mkdir -p "$SCRATCH"
echo "===== A4→A5 同批量对照队列启动 $(date) =====" >> "$Q"
run a4_neatpack_nogc_gas2 "0,1,2,3" 29791
run a5_neatpack_gc_gas2   "0,1,2,3" 29792
echo "===== A4→A5 同批量对照队列结束 $(date) =====" >> "$Q"
