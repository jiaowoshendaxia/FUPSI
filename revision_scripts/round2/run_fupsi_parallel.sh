#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CODE_ROOT="$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/fupsi/train.py" ]]; then
  DEFAULT_CODE_ROOT="$PROJECT_ROOT/fupsi"
fi
ROOT="${1:-$DEFAULT_CODE_ROOT}"
PYTHON="${PYTHON:-python}"
NAMESPACE="${NAMESPACE:-ResidualMainE300P5}"
RUNNER="$SCRIPT_DIR/../run_main_seed_task_queue.py"
TRAIN_SCRIPT="main_seed_train_residual.py"
TEST_SCRIPT="main_seed_test_residual.py"

mkdir -p \
  "$ROOT/data" \
  "$ROOT/revision/statistics/$NAMESPACE" \
  "$ROOT/revision_main_seed_logs"

for dataset in TaxiBJ_P1 TaxiBJ_P2 TaxiBJ_P3 TaxiBJ_P4 BikeNYC; do
  source_alias="MainSeed_${dataset}"
  target_alias="${NAMESPACE}_${dataset}"
  if [[ ! -e "$ROOT/data/$target_alias" ]]; then
    ln -s "$ROOT/data/$source_alias" "$ROOT/data/$target_alias"
  fi

  status_dir="$ROOT/revision/statistics/$NAMESPACE/$dataset"
  mkdir -p "$status_dir"
  worker_log="$status_dir/worker.log"

  nohup "$PYTHON" -u "$RUNNER" \
    --code-root "$ROOT" \
    --datasets "$source_alias" \
    --seeds 2024,2025,2026 \
    --stages pretrain,train,test \
    --epochs 300 \
    --batch-size 32 \
    --namespace "$NAMESPACE" \
    --train-script "$TRAIN_SCRIPT" \
    --test-script "$TEST_SCRIPT" \
    --lambda-adv 0 \
    --status-dir "$status_dir" \
    > "$worker_log" 2>&1 &

  printf '%s\t%s\t%s\n' "$dataset" "$!" "$worker_log"
done
