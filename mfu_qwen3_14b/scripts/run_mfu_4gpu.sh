#!/usr/bin/env bash
# ============================================================================
# Qwen3-14B 全参 SFT · 4 卡 · per-step MFU 实验启动器
# ----------------------------------------------------------------------------
# 用法:  bash scripts/run_mfu_4gpu.sh <配置名> "<GPU列表逗号分隔>"
#   例:  bash scripts/run_mfu_4gpu.sh r1_mbs2_z2 "0,1,2,3"
#
# 与 8B 单卡版的关键区别：
#   ★ 不开 CCE、不开 LoRAFusion（用户明确要求测"不加这两项优化"能到多少）
#     → ENABLE_CCE / ENABLE_FUSED_LORA 一律不 export，venv 钩子读不到即 no-op
#   ★ 4 卡 torchrun，计时钩子每个 rank 各写一份 clean_timing_rank{N}.jsonl
#
# 严格遵循《MFU 实验统计原则》：
#   - per-step 计时由 scripts/sitecustomize.py 注入，边界见该文件注释
#   - T_start 在启动训练命令**之前**记录（§5.1），T_end 在进程退出之后
#
# GPU 纪律：只用空闲卡，绝不抢占。每张卡启动前自检，占用 >2GB 直接拒绝启动。
# ============================================================================
set -uo pipefail

CFG="${1:?用法: bash scripts/run_mfu_4gpu.sh <配置名> \"0,1,2,3\"}"
GPUS="${2:-0,1,2,3}"

BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b"
VENV="/fine-tuning-launcher/.venv"
YAML="${BASE}/configs/${CFG}.yaml"
[ -f "$YAML" ] || { echo "找不到配置: $YAML"; exit 1; }

TS=$(date +"%Y%m%d_%H%M%S")
OUT="${BASE}/results/${CFG}/run_${TS}"
mkdir -p "$OUT"
LOG="${OUT}/train.log"

# ---- GPU 自检：绝不抢占别人的任务（fail-closed，读不到状态也拒绝）----
echo "=== GPU 自检 (${GPUS}) ==="
NGPU=0
for G in ${GPUS//,/ }; do
  USED=$(nvidia-smi -i "$G" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo NA)
  echo "  GPU ${G} 已用显存: ${USED} MiB"
  if [ "$USED" = "NA" ]; then
    echo "  ✗ 读不到 GPU ${G} 状态，拒绝启动"; exit 1
  fi
  if [ "$USED" -gt 2000 ]; then
    echo "  ✗ GPU ${G} 上有别人的任务(${USED} MiB > 2000)，拒绝启动（不抢占）"; exit 1
  fi
  NGPU=$((NGPU + 1))
done
echo "  ✓ ${NGPU} 张卡均空闲，可用"

export CUDA_VISIBLE_DEVICES="$GPUS"

# ---- FA3 前置断言：yaml 写了 fa3 但装不上会被 LLaMA-Factory 静默降级成 eager，
#      只在日志里留一行 warning。那样测出来的 MFU 是另一个口径，必须提前挡住。----
grep -q "^flash_attn: fa3" "$YAML" || { echo "  ✗ ${CFG} 没写 flash_attn: fa3，拒绝启动"; exit 1; }
"${VENV}/bin/python" -c "
import sys
from transformers.utils import is_flash_attn_3_available
sys.exit(0 if is_flash_attn_3_available() else 1)
" 2>/dev/null || { echo "  ✗ FA3 不可用，拒绝启动（否则会静默降级成 eager，口径不对）"; exit 1; }
echo "  ✓ FA3 可用"

# ---- env ----
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export NNODES=1
export NPROC_PER_NODE="$NGPU"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29751}"
export PATH="${VENV}/bin:${PATH}"
NVLIB=$("${VENV}/bin/python" -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(glob.glob(b+'/*/lib'))))" 2>/dev/null || echo "")
export LD_LIBRARY_PATH="${NVLIB}:${LD_LIBRARY_PATH:-}"

# ★ 明确不开 CCE / LoRAFusion —— 这是本轮实验的前提条件，不要加回来
unset ENABLE_CCE
unset ENABLE_FUSED_LORA

# torchrun 默认把 OMP_NUM_THREADS 设成 1，会把 CPU 侧 dataloader / Adam 掐死在单核。
# 本实验不用 CPU offload，影响小，但没有理由留着 1。4 卡各分 12 个物理核。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

export MFU_TIMING_OUT="${OUT}/clean_timing.jsonl"   # 实际写 clean_timing_rank{N}.jsonl
# 本目录 scripts/ 置最前 → 我们的合并 sitecustomize 生效（它会 exec venv 那份）
export PYTHONPATH="${BASE}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

cp "$YAML" "${OUT}/effective_config.yaml"

echo "=============================================="
echo " 配置=${CFG}   GPU=${GPUS} (${NGPU} 卡)"
echo " 输出=${OUT}"
echo " CCE=off  LoRAFusion=off  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "=============================================="

# ---- T_start：必须在启动训练命令之前记录（原则 §5.1）----
"${VENV}/bin/python" - "$OUT" <<'PYEOF'
import json, sys, time
out = sys.argv[1]
json.dump({"t_start_monotonic": time.monotonic(), "t_start_epoch": time.time()},
          open(f"{out}/e2e_start.json", "w"), indent=2)
PYEOF
T0=$(date +%s.%N)

llamafactory-cli train "${OUT}/effective_config.yaml" 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}

# ---- 事后核对：日志里出现降级 warning 就把本组标记为口径不合格 ----
if grep -q "FlashAttention-3 is not installed" "${LOG}"; then
  echo "  ✗✗ 本组 FA3 被静默降级，结果不可用" | tee -a "${LOG}"
  echo '{"invalid":"fa3_downgraded"}' > "${OUT}/INVALID.json"
fi

# ---- T_end：所有训练进程退出之后 ----
T1=$(date +%s.%N)
"${VENV}/bin/python" - "$OUT" "$T0" "$T1" "$RC" <<'PYEOF'
import json, sys
out, t0, t1, rc = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
json.dump({"wall_time_s": t1 - t0, "train_exit_code": rc,
           "note": "T_start 在 launcher 启动训练命令前记录, T_end 在进程退出后记录"},
          open(f"{out}/e2e_end.json", "w"), indent=2)
print(f"\n=== E2E wall time: {t1-t0:.4f}s  exit={rc} ===")
PYEOF

echo "=== 产物目录: ${OUT} ==="
exit $RC
