#!/usr/bin/env bash
set -u

ROOT="/wanqing-develop/luowenjing/Param_Recommend/qwen3_14b_4gpu_rank_validation_20260730"
RUN_ONE="$ROOT/run_one.sh"

# Reverse predicted order makes the predicted winner run last, which avoids
# giving it an unfair cold-GPU advantage.
declare -a RUNS=(
  "rank3_z3_gc_mbs8_ga4|$ROOT/rank3_z3_gc_mbs8_ga4.yaml|29673"
  "rank2_z2_gc_mbs8_ga4|$ROOT/rank2_z2_gc_mbs8_ga4.yaml|29672"
  "rank1_z3_gc_mbs16_ga2|$ROOT/rank1_z3_gc_mbs16_ga2.yaml|29671"
)

OVERALL_RC=0
for entry in "${RUNS[@]}"; do
  IFS='|' read -r tag yaml port <<< "$entry"
  echo "[rank-validation] starting $tag"
  if ! bash "$RUN_ONE" "$tag" "$yaml" "$port"; then
    echo "[rank-validation] $tag failed" >&2
    OVERALL_RC=1
  fi
  echo "[rank-validation] finished $tag"
done

exit "$OVERALL_RC"
