#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CODE_ROOT="$PROJECT_ROOT"
DEFAULT_OUTPUT_ROOT="$PROJECT_ROOT/revision/round2/hrstt"
if [[ -f "$PROJECT_ROOT/fupsi/train.py" ]]; then
  DEFAULT_CODE_ROOT="$PROJECT_ROOT/fupsi"
  DEFAULT_OUTPUT_ROOT="$PROJECT_ROOT/results/round2/hrstt"
fi
CODE_ROOT="${1:-$DEFAULT_CODE_ROOT}"
OUTPUT_ROOT="${2:-$DEFAULT_OUTPUT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT_DIR}/hrstt_reimplementation.py"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

run_dataset() {
  local dataset="$1"
  for seed in 2024 2025 2026; do
    local extra_args=()
    if [[ "${dataset}" == "TaxiBJ_P4" && "${seed}" == "2024" ]]; then
      extra_args=(--save-predictions)
    fi
    "${PYTHON_BIN}" -u "${SCRIPT}" \
      --code-root "${CODE_ROOT}" \
      --dataset "${dataset}" \
      --data-prefix MainSeed \
      --seed "${seed}" \
      --epochs 300 \
      --batch-size 32 \
      --learning-rate 0.0001 \
      --halving-interval 20 \
      --lambda-coarse 0.01 \
      --feature-dim 64 \
      --residual-blocks 4 \
      --transformer-layers 2 \
      --transformer-heads 4 \
      --output-root "${OUTPUT_ROOT}" \
      "${extra_args[@]}" \
      > "${LOG_ROOT}/${dataset}_seed${seed}.log" 2>&1
  done
}

for dataset in TaxiBJ_P1 TaxiBJ_P2 TaxiBJ_P3 TaxiBJ_P4 BikeNYC; do
  run_dataset "${dataset}" &
  echo "$!" > "${LOG_ROOT}/${dataset}.pid"
done

wait
echo "All HRSTT reimplementation runs completed."
