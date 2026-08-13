#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <tag> <yaml> <master_port> [physical_gpu_ids]" >&2
  exit 2
fi

ROOT="/wanqing-develop/luowenjing/Param_Recommend/qwen3_14b_2gpu_ab_20260730"
VENV="/fine-tuning-launcher/.venv"
TANGCAN_SRC="/wanqing-models/tangcan/llama-factory-local/llama-factory/src"
TAG="$1"
YAML="$2"
MASTER_PORT_VALUE="$3"
PHYSICAL_GPU_IDS="${4:-4,5}"
LOG="$ROOT/logs/${TAG}.log"
GPU_LOG="$ROOT/logs/${TAG}_gpu.csv"
META="$ROOT/logs/${TAG}_meta.txt"

mkdir -p "$ROOT/logs" "$ROOT/outputs"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_IDS"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$TANGCAN_SRC:${PYTHONPATH:-}"
export ENABLE_CCE=1
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=48
export MASTER_PORT="$MASTER_PORT_VALUE"
unset FLOPS_MONITOR NSYS_STEP_WINDOW PIN_SHM ENABLE_FUSED_LORA

{
  echo "tag=$TAG"
  echo "yaml=$YAML"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "master_port=$MASTER_PORT"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$META"

echo "timestamp,index,memory_used_mib,utilization_gpu_percent,power_draw_w" > "$GPU_LOG"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits |
      awk -F', ' -v ids="$PHYSICAL_GPU_IDS" '
        BEGIN {
          count = split(ids, parts, ",")
          for (i = 1; i <= count; i++) selected[parts[i]] = 1
        }
        selected[$2] { print }
      ' >> "$GPU_LOG"
    sleep 2
  done
) &
MONITOR_PID=$!

stop_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap stop_monitor EXIT

set +e
"$VENV/bin/llamafactory-cli" train "$YAML" 2>&1 | tee "$LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e

stop_monitor
trap - EXIT
{
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=$TRAIN_RC"
} >> "$META"

exit "$TRAIN_RC"
