#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <tag> <yaml> <master_port>" >&2
  exit 2
fi

ROOT="/wanqing-develop/luowenjing/Param_Recommend/qwen3_14b_4gpu_rank_validation_20260730"
VENV="/fine-tuning-launcher/.venv"
TANGCAN_SRC="/wanqing-models/tangcan/llama-factory-local/llama-factory/src"
TAG="$1"
YAML="$2"
MASTER_PORT_VALUE="$3"
PHYSICAL_GPU_IDS="4,5,6,7"
LOG="$ROOT/logs/${TAG}.log"
GPU_LOG="$ROOT/logs/${TAG}_gpu.csv"
META="$ROOT/logs/${TAG}_meta.txt"

mkdir -p "$ROOT/logs" "$ROOT/outputs"

if ! nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits |
  awk -F', ' '
    $1 >= 4 && $1 <= 7 {
      if (($2 + 0) > 1024 || ($3 + 0) > 10) busy = 1
    }
    END { exit busy ? 1 : 0 }
  '; then
  echo "GPU 4,5,6,7 are not idle; refusing to overlap another job." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_IDS"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT/runtime_overlay:$TANGCAN_SRC:${PYTHONPATH:-}"
export ENABLE_CCE=1
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export MASTER_PORT="$MASTER_PORT_VALUE"
export PARAM_RECOMMEND_BENCHMARK_NO_SAVE=1
unset FLOPS_MONITOR NSYS_STEP_WINDOW PIN_SHM ENABLE_FUSED_LORA

{
  echo "tag=$TAG"
  echo "yaml=$YAML"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "master_port=$MASTER_PORT"
  echo "benchmark_no_save=$PARAM_RECOMMEND_BENCHMARK_NO_SAVE"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "$YAML"
} > "$META"

echo "timestamp,index,memory_used_mib,utilization_gpu_percent,power_draw_w" > "$GPU_LOG"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits |
      awk -F', ' '
        $2 >= 4 && $2 <= 7 { print }
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
  du -sh "$ROOT/outputs/$TAG" 2>/dev/null || true
} >> "$META"

exit "$TRAIN_RC"
