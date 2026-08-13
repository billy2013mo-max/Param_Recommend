#!/usr/bin/env bash
# ============================================================================
# D → C → A 串行队列（用户指定顺序）
# ----------------------------------------------------------------------------
#   D  4卡 ZeRO-2 关梯度检查点 MBS1   GPU 0-3   —— 验证"关GC只能到MBS1且悬"的估算
#   C  8卡 ZeRO-3 关梯度检查点 MBS2   GPU 0-7   —— 唯一能关GC还保住MBS2的组合
#   A  4卡 ZeRO-2 开 neat_packing     GPU 0-3   —— 去掉 24.8% 的补零浪费
#
# 串行跑：没有邻居抢 CPU/IO，稳态波动只来自 batch 组成本身。
# D 预期可能 OOM（显存模型预测 132.1 GiB vs 安全线 133.4），OOM 不终止队列。
# ============================================================================
set -uo pipefail
BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b"
Q="${BASE}/results/queue_dca.log"

run() {
  local cfg="$1" gpus="$2" port="$3"
  echo "[$(date +%H:%M:%S)] 开始 ${cfg}  GPU=${gpus}" >> "$Q"
  MASTER_PORT="$port" bash "${BASE}/scripts/run_mfu_4gpu.sh" "$cfg" "$gpus" \
    > "${BASE}/results/${cfg}.launch.log" 2>&1
  echo "[$(date +%H:%M:%S)] 结束 ${cfg}  exit=$?" >> "$Q"
}

echo "===== D→C→A 串行队列启动 $(date) =====" >> "$Q"
run d_nogc_mbs1_z2      "0,1,2,3"         29771
run c_nogc_mbs2_z3_8gpu "0,1,2,3,4,5,6,7" 29772
run a_neatpack_z2       "0,1,2,3"         29773
echo "===== D→C→A 串行队列结束 $(date) =====" >> "$Q"
