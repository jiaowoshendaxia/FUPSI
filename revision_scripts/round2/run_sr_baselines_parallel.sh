#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${1:-${DEFAULT_ROOT}}"
OUTPUT_ROOT="${2:-${ROOT}/revision/round2/sr_baselines}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
CUFAR_ROOT="${CUFAR_ROOT:-${ROOT}/revision/external_baselines/CUFAR}"
FUPSI_NAMESPACE="${FUPSI_NAMESPACE:-ResidualMainE300P5}"
ADAPTER="${ROOT}/revision_scripts/run_sr_baseline_adapter.py"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

required=(
  "${ADAPTER}"
  "${CUFAR_ROOT}/model/UrbanFM.py"
  "${CUFAR_ROOT}/model/FODE.py"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
done

run_dataset() {
  local dataset="$1"
  local models="$2"
  local log_path="${LOG_ROOT}/${dataset}.log"
  "${PYTHON_BIN}" -u "${ADAPTER}" \
    --code-root "${ROOT}" \
    --cufar-root "${CUFAR_ROOT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --datasets "${dataset}" \
    --seeds 2024,2025,2026 \
    --models "${models}" \
    --epochs 100 \
    --fupsi-epochs 300 \
    --batch-size 64 \
    --log-every 5 \
    --fupsi-namespace "${FUPSI_NAMESPACE}" \
    >"${log_path}" 2>&1
}

pids=()
for dataset in \
  MainSeed_TaxiBJ_P1 \
  MainSeed_TaxiBJ_P2 \
  MainSeed_TaxiBJ_P3 \
  MainSeed_TaxiBJ_P4; do
  run_dataset "${dataset}" "UrbanFM,FODE" &
  pids+=("$!")
done
run_dataset "MainSeed_BikeNYC" "UrbanFM,FODE" &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one SR baseline worker failed; inspect ${LOG_ROOT}." >&2
  exit 1
fi

echo "Completed 30 SR baseline runs in ${OUTPUT_ROOT}."
