#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
ROUND2="$ROOT/revision/round2"
LOG_ROOT="$ROUND2/orchestrator"
STATUS="$LOG_ROOT/status.txt"

mkdir -p "$LOG_ROOT"

set_status() {
  printf '%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee "$STATUS"
}

count_files() {
  local root="$1"
  local name="$2"
  if [[ ! -d "$root" ]]; then
    printf '0\n'
    return 0
  fi
  find "$root" -type f -name "$name" 2>/dev/null | wc -l
}

set_status "waiting_for_fupsi"
while pgrep -f "run_main_seed_task_queue.py.*ResidualMainE300P5" >/dev/null; do
  sleep 300
done

fupsi_metrics="$(
  find "$ROOT/saved_model/to_stage/no_ext(r)" \
    -type f -path "*ResidualMainE300P5_*" -name test_metrics.csv 2>/dev/null |
    wc -l
)"
if [[ "$fupsi_metrics" -ne 15 ]]; then
  set_status "blocked_fupsi_metrics_${fupsi_metrics}_of_15"
  exit 2
fi
set_status "fupsi_complete_15_of_15"
"$PYTHON" "$SCRIPT_DIR/collect_fupsi_seed_metrics.py" \
  --code-root "$ROOT" \
  --namespace ResidualMainE300P5 \
  --output "$ROUND2/fupsi_seed_metrics.csv" \
  > "$LOG_ROOT/fupsi_collection.log" 2>&1
"$PYTHON" "$SCRIPT_DIR/evaluate_hamean.py" \
  --code-root "$ROOT" \
  --data-prefix MainSeed \
  --output "$ROUND2/hamean_seed_metrics.csv" \
  > "$LOG_ROOT/hamean.log" 2>&1
"$PYTHON" "$SCRIPT_DIR/test_sparse_completion_causality.py" \
  > "$LOG_ROOT/sparse_causality.log" 2>&1
set_status "sparse_causality_passed"

sparse_dir="$ROUND2/sparse_pipeline"
if [[ ! -f "$sparse_dir/sparse_pipeline_mean_std.csv" ]]; then
  set_status "running_sparse_pipeline"
  "$PYTHON" -u "$SCRIPT_DIR/evaluate_sparse_pipeline.py" \
    --code-root "$ROOT" \
    --data-prefix ResidualMainE300P5 \
    --model-prefix ResidualMainE300P5 \
    --datasets TaxiBJ_P1,TaxiBJ_P2,TaxiBJ_P3,TaxiBJ_P4,BikeNYC \
    --seeds 2024,2025,2026 \
    --rates 0,0.1,0.3,0.5,0.7 \
    --methods adaptive,no_completion \
    --output-dir "$sparse_dir" \
    > "$LOG_ROOT/sparse_pipeline.log" 2>&1
fi
sparse_rows="$(
  "$PYTHON" - "$sparse_dir/sparse_pipeline_seed_metrics.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(sum(1 for _ in csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else 0)
PY
)"
if [[ "$sparse_rows" -ne 150 ]]; then
  set_status "blocked_sparse_rows_${sparse_rows}_of_150"
  exit 3
fi
set_status "sparse_pipeline_complete_150_of_150"
"$PYTHON" "$SCRIPT_DIR/summarize_sparse_pipeline.py" \
  --input "$sparse_dir/sparse_pipeline_seed_metrics.csv" \
  --output-dir "$sparse_dir/analysis" \
  --expected-rows 150 \
  > "$LOG_ROOT/sparse_summary.log" 2>&1

hrstt_dir="$ROUND2/hrstt"
hrstt_metrics="$(count_files "$hrstt_dir" test_metrics.csv)"
if [[ "$hrstt_metrics" -lt 15 ]]; then
  set_status "running_hrstt"
  bash "$SCRIPT_DIR/run_hrstt_parallel.sh" \
    "$ROOT" "$hrstt_dir" \
    > "$LOG_ROOT/hrstt_launcher.log" 2>&1
fi
hrstt_metrics="$(count_files "$hrstt_dir" test_metrics.csv)"
if [[ "$hrstt_metrics" -ne 15 ]]; then
  set_status "blocked_hrstt_metrics_${hrstt_metrics}_of_15"
  exit 4
fi
"$PYTHON" "$SCRIPT_DIR/summarize_hrstt.py" \
  --input-root "$hrstt_dir" \
  --output-dir "$hrstt_dir/summary" \
  > "$LOG_ROOT/hrstt_summary.log" 2>&1
set_status "hrstt_complete_15_of_15"

inverse_dir="$ROUND2/inverse_order"
inverse_metrics="$(count_files "$inverse_dir" test_metrics.csv)"
if [[ "$inverse_metrics" -lt 3 ]]; then
  set_status "running_inverse_order"
  bash "$SCRIPT_DIR/run_inverse_order_p4.sh" \
    "$ROOT" "$inverse_dir" \
    > "$LOG_ROOT/inverse_order_launcher.log" 2>&1
fi
inverse_metrics="$(count_files "$inverse_dir" test_metrics.csv)"
if [[ "$inverse_metrics" -ne 3 ]]; then
  set_status "blocked_inverse_metrics_${inverse_metrics}_of_3"
  exit 5
fi
"$PYTHON" "$SCRIPT_DIR/summarize_inverse_order.py" \
  --fupsi "$ROUND2/fupsi_seed_metrics.csv" \
  --inverse-root "$inverse_dir" \
  --output-dir "$inverse_dir/summary" \
  > "$LOG_ROOT/inverse_order_summary.log" 2>&1
set_status "inverse_order_complete_3_of_3"
"$PYTHON" "$SCRIPT_DIR/generate_p4_order_visualization.py" \
  --code-root "$ROOT" \
  --namespace ResidualMainE300P5 \
  --inverse-root "$inverse_dir" \
  --seed 2024 \
  --output-dir "$ROUND2/visualization" \
  > "$LOG_ROOT/p4_visualization.log" 2>&1
set_status "p4_visualization_complete"

sr_dir="$ROUND2/sr_baselines"
sr_metrics="$(count_files "$sr_dir" test_metrics.csv)"
if [[ "$sr_metrics" -lt 30 ]]; then
  set_status "running_sr_baselines"
  bash "$SCRIPT_DIR/run_sr_baselines_parallel.sh" \
    "$ROOT" "$sr_dir" \
    > "$LOG_ROOT/sr_baseline_launcher.log" 2>&1
fi
sr_metrics="$(count_files "$sr_dir" test_metrics.csv)"
if [[ "$sr_metrics" -ne 30 ]]; then
  set_status "blocked_sr_metrics_${sr_metrics}_of_30"
  exit 6
fi
"$PYTHON" "$SCRIPT_DIR/summarize_sr_baselines.py" \
  --input-root "$sr_dir" \
  --fupsi "$ROUND2/fupsi_seed_metrics.csv" \
  --output-dir "$sr_dir/summary" \
  > "$LOG_ROOT/sr_baseline_summary.log" 2>&1
set_status "sr_baselines_complete_30_of_30"

"$PYTHON" "$SCRIPT_DIR/build_unified_main_statistics.py" \
  --fupsi "$ROUND2/fupsi_seed_metrics.csv" \
  --hrstt "$hrstt_dir/summary/hrstt_seed_metrics.csv" \
  --sr "$sr_dir/summary/sr_baseline_seed_metrics.csv" \
  --hamean "$ROUND2/hamean_seed_metrics.csv" \
  --output-dir "$ROUND2/main_statistics" \
  > "$LOG_ROOT/unified_statistics.log" 2>&1

set_status "running_complexity"
"$PYTHON" "$ROOT/revision_scripts/measure_sr_pipeline_complexity.py" \
  --code-root "$ROOT" \
  --cufar-root "$ROOT/revision/external_baselines/CUFAR" \
  --output-dir "$ROUND2/complexity" \
  > "$LOG_ROOT/complexity.log" 2>&1
complexity_rows="$(
  "$PYTHON" - "$ROUND2/complexity/sr_pipeline_complexity_detailed.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(sum(1 for _ in csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else 0)
PY
)"
if [[ "$complexity_rows" -ne 15 ]]; then
  set_status "blocked_complexity_rows_${complexity_rows}_of_15"
  exit 7
fi
set_status "complexity_complete_15_of_15"
set_status "all_remote_experiments_complete"
