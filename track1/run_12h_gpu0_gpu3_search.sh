#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOURS="${HOURS:-12}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/track1/search_runs/${RUN_ID}}"

mkdir -p "${OUT_DIR}"
printf '%s\n' "${OUT_DIR}" > "${ROOT_DIR}/track1/search_runs/latest.txt"

{
  echo "launcher_pid=$$"
  echo "started_at=$(date -Is)"
  echo "hours=${HOURS}"
  echo "out_dir=${OUT_DIR}"
  echo "python=${PYTHON_BIN}"
} > "${OUT_DIR}/run.meta"

"${PYTHON_BIN}" "${ROOT_DIR}/track1/official/search_runner.py" \
  --gpu 0 \
  --worker gpu0_correctness \
  --profile correctness \
  --hours "${HOURS}" \
  --out-dir "${OUT_DIR}" \
  > "${OUT_DIR}/gpu0_worker.log" 2>&1 &
GPU0_PID=$!

"${PYTHON_BIN}" "${ROOT_DIR}/track1/official/search_runner.py" \
  --gpu 3 \
  --worker gpu3_benchmark \
  --profile benchmark \
  --hours "${HOURS}" \
  --out-dir "${OUT_DIR}" \
  > "${OUT_DIR}/gpu3_worker.log" 2>&1 &
GPU3_PID=$!

{
  echo "gpu0_pid=${GPU0_PID}"
  echo "gpu3_pid=${GPU3_PID}"
} >> "${OUT_DIR}/run.meta"

wait "${GPU0_PID}" "${GPU3_PID}" || true

"${PYTHON_BIN}" "${ROOT_DIR}/track1/official/summarize_search.py" "${OUT_DIR}" \
  > "${OUT_DIR}/summary.md" 2>> "${OUT_DIR}/summary.err" || true

echo "finished_at=$(date -Is)" >> "${OUT_DIR}/run.meta"
