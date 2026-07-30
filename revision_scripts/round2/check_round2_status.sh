#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_ROOT="${1:-${PACKAGE_ROOT}/fupsi}"
ROUND2="${2:-${PACKAGE_ROOT}/results/round2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PACKAGE_ROOT" || exit 1

echo "STATUS"
cat "$ROUND2/orchestrator/status.txt" 2>/dev/null || echo "missing"

echo "COUNTS"
printf 'fupsi_metrics='
find "$CODE_ROOT/saved_model/to_stage/no_ext(r)" \
  -type f -path '*ResidualMainE300P5_*' -name test_metrics.csv \
  2>/dev/null | wc -l
printf 'fupsi_workers='
pgrep -af 'run_main_seed_task_queue.py.*ResidualMainE300P5' \
  2>/dev/null | grep -v pgrep | wc -l
printf 'sparse_rows='
"${PYTHON_BIN}" - "$ROUND2/sparse_pipeline/sparse_pipeline_seed_metrics.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(sum(1 for _ in csv.DictReader(path.open())) if path.exists() else 0)
PY
printf 'hrstt_metrics='
find "$ROUND2/hrstt" -type f -name test_metrics.csv 2>/dev/null | wc -l
printf 'inverse_metrics='
find "$ROUND2/inverse_order" -type f -name test_metrics.csv \
  2>/dev/null | wc -l
printf 'sr_metrics='
find "$ROUND2/sr_baselines" -type f -name test_metrics.csv \
  2>/dev/null | wc -l

echo "GPU"
nvidia-smi \
  --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>/dev/null || true

echo "PROCESSES"
pgrep -af \
  'train.py|main_seed_train_residual.py|main_seed_test_residual.py|run_main_seed_task_queue.py|evaluate_sparse_pipeline.py|hrstt_reimplementation.py|inverse_order_reimplementation.py|run_sr_baseline_adapter.py' \
  2>/dev/null | grep -v pgrep | head -30 || true

echo "RECENT_LOGS"
find "$CODE_ROOT/revision_main_seed_logs" \
  -type f -name 'ResidualMainE300P5_*.log' \
  -printf '%T@ %p\n' 2>/dev/null |
  sort -nr |
  head -8 |
  cut -d' ' -f2- |
  while read -r log_file; do
    latest="$(
      tr '\r' '\n' < "$log_file" |
      grep -E \
        '\[[0-9]+/300\]|\[Epoch [0-9]+/300\]|RMSE=|completed|time cost' |
      tail -n 1
    )"
    printf '%s\t%s\n' "$log_file" "${latest:-no parsed progress}"
  done

echo "ERRORS"
grep -RniE \
  'Traceback|CUDA out of memory|RuntimeError|Error:' \
  "$CODE_ROOT"/revision_main_seed_logs/ResidualMainE300P5_*.log \
  "$ROUND2"/orchestrator/*.log \
  2>/dev/null | tail -20 || true
