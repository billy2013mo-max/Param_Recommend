#!/usr/bin/env bash
# ============================================================================
# Qwen3-8B 单卡 FA3 · per-step MFU 实验启动器
# ----------------------------------------------------------------------------
# 用法:  bash scripts/run_mfu.sh <配置名> [GPU]
#        配置名 ∈ configs/*.yaml 的文件名(不含 .yaml)
#
# 严格遵循《MFU 实验统计原则》：
#   - per-step 计时由 scripts/sitecustomize.py 注入，边界见该文件注释
#   - T_start 在启动训练命令**之前**记录（§5.1 要求），写入 e2e_start.json
#   - T_end 在训练进程退出后记录
#
# GPU 纪律：只用空闲卡，绝不抢占。启动前自检，占用 >2GB 直接拒绝启动。
# ============================================================================
set -uo pipefail

CFG="${1:?用法: bash scripts/run_mfu.sh <配置名> [GPU]}"
GPU="${2:-0}"

BASE="/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_8b"
VENV="/fine-tuning-launcher/.venv"
LORAFUSION_REPO="/wanqing-develop/chengjin/LoRAFusion"
YAML="${BASE}/configs/${CFG}.yaml"
[ -f "$YAML" ] || { echo "找不到配置: $YAML"; exit 1; }

TS=$(date +"%Y%m%d_%H%M%S")
OUT="${BASE}/results/${CFG}/run_${TS}"
mkdir -p "$OUT"
LOG="${OUT}/train.log"

export CUDA_VISIBLE_DEVICES="$GPU"

# ---- GPU 自检：绝不抢占别人的任务 ----
echo "=== GPU 自检 (GPU ${GPU}) ==="
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo NA)
echo "  GPU ${GPU} 已用显存: ${USED} MiB"
if [ "$USED" = "NA" ]; then
  echo "  ✗ 读不到 GPU ${GPU} 状态，拒绝启动"; exit 1
fi
if [ "$USED" -gt 2000 ]; then
  echo "  ✗ GPU ${GPU} 上有别人的任务(${USED} MiB > 2000)，拒绝启动（不抢占）"; exit 1
fi
echo "  ✓ GPU ${GPU} 空闲，可用"

# ---- env：CCE + LoRAFusion + per-step 计时钩子 ----
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PATH="${VENV}/bin:${PATH}"
NVLIB=$("${VENV}/bin/python" -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(glob.glob(b+'/*/lib'))))" 2>/dev/null || echo "")
export LD_LIBRARY_PATH="${NVLIB}:${LD_LIBRARY_PATH:-}"

export ENABLE_CCE=1
export ENABLE_FUSED_LORA=1
export LORAFUSION_FORWARD_USE_TMA=0
export LORAFUSION_BACKWARD_USE_TMA=0
export LORAFUSION_REPO

export MFU_TIMING_OUT="${OUT}/clean_timing.jsonl"
# 本目录 scripts/ 置最前 → 我们的合并 sitecustomize 生效（它会 exec venv 那份）
export PYTHONPATH="${BASE}/scripts:${LORAFUSION_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

cp "$YAML" "${OUT}/effective_config.yaml"

echo "=============================================="
echo " 配置=${CFG}   GPU=${GPU}"
echo " 输出=${OUT}"
echo " ENABLE_CCE=${ENABLE_CCE}  ENABLE_FUSED_LORA=${ENABLE_FUSED_LORA}"
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
