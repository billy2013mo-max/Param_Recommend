#!/usr/bin/env bash
# ============================================================================
# A2 → A3 串行队列（在 A 组 30.58% 基础上各改一个变量）
# ----------------------------------------------------------------------------
#   A2  打包 + 关梯度检查点，cutoff 4096   —— 省掉整个重算前向，预期 +5~15 点
#                                            风险：打包后每行满 4095，可能 OOM
#   A3  打包 + cutoff 8192（生效 8191）    —— 装填率已 97.3%，预期仅 +1~2 点
#                                            显存预测 117.2 GiB，余量 16.2
#
# 与 A 组逐键核对过，除上述单一变量 + output_dir 外完全一致。
#
# ★ 权重落 /root（本地盘 3.4 TB 余量），跑完立即删。
#   共享盘 /wanqing-develop 只剩 284 GB，上一轮 7 组权重吃掉 193 GB 把盘撑满过。
#   框架里 trainer.save_model() 是硬编码的，save_strategy:"no" 关不掉它，
#   所以只能靠跑完就删。删除只针对本队列自己的 output_dir，不碰别的。
# ============================================================================
set -uo pipefail
BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b"
Q="${BASE}/results/queue_a2a3.log"
SCRATCH="/root/scratch_saves"

run() {
  local cfg="$1" gpus="$2" port="$3"
  echo "[$(date +%H:%M:%S)] 开始 ${cfg}  GPU=${gpus}" >> "$Q"
  MASTER_PORT="$port" bash "${BASE}/scripts/run_mfu_4gpu.sh" "$cfg" "$gpus" \
    > "${BASE}/results/${cfg}.launch.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] 结束 ${cfg}  exit=${rc}" >> "$Q"
  # 立刻回收权重：只删本组自己那个 scratch 目录
  if [ -d "${SCRATCH}/${cfg}" ]; then
    local sz=$(du -sh "${SCRATCH}/${cfg}" 2>/dev/null | cut -f1)
    rm -rf "${SCRATCH}/${cfg}" && echo "[$(date +%H:%M:%S)] 已删权重 ${cfg} (${sz})" >> "$Q"
  fi
}

mkdir -p "$SCRATCH"
echo "===== A2→A3 串行队列启动 $(date) =====" >> "$Q"
run a2_neatpack_nogc     "0,1,2,3" 29781
run a3_neatpack_cut8192  "0,1,2,3" 29782
echo "===== A2→A3 串行队列结束 $(date) =====" >> "$Q"
