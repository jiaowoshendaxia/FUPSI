#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CODE_ROOT="$PROJECT_ROOT"
DEFAULT_OUTPUT_ROOT="$PROJECT_ROOT/revision/round2/inverse_order"
if [[ -f "$PROJECT_ROOT/fupsi/train.py" ]]; then
  DEFAULT_CODE_ROOT="$PROJECT_ROOT/fupsi"
  DEFAULT_OUTPUT_ROOT="$PROJECT_ROOT/results/round2/inverse_order"
fi
CODE_ROOT="${1:-$DEFAULT_CODE_ROOT}"
OUTPUT_ROOT="${2:-$DEFAULT_OUTPUT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT_DIR}/inverse_order_reimplementation.py"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

for seed in 2024 2025 2026; do
  extra_args=()
  if [[ "${seed}" == "2024" ]]; then
    extra_args=(--save-predictions)
  fi
  "${PYTHON_BIN}" -u "${SCRIPT}" \
    --code-root "${CODE_ROOT}" \
    --dataset TaxiBJ_P4 \
    --data-prefix MainSeed \
    --seed "${seed}" \
    --sr-epochs 300 \
    --prediction-epochs 300 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --halving-interval 20 \
    --output-root "${OUTPUT_ROOT}" \
    "${extra_args[@]}" \
    > "${LOG_ROOT}/TaxiBJ_P4_seed${seed}.log" 2>&1 &
  echo "$!" > "${LOG_ROOT}/TaxiBJ_P4_seed${seed}.pid"
done

wait
echo "All inverse-order TaxiBJ P4 runs completed."
