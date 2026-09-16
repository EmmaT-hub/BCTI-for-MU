#!/usr/bin/env bash
# Submit only this script. Select exactly TWO sufficiently large GPUs.
set -Eeuo pipefail

trap 'status=$?; echo "[all-paper-baselines] FAILED line=${LINENO} exit=${status}" >&2; exit "${status}"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"

PROJECT_DIR="${PROJECT_DIR:-.}"
CONDA_ENV="${CONDA_ENV:-unlearn}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_BASE="${OUT_BASE:-outputs/50-targets/baselines/${RUN_ID}}"

export PROJECT_DIR CONDA_ENV RUN_ID

run_one() {
  local method="$1"
  export METHOD="${method}"
  export CONFIG="comp/BCTI-comparisons/config_${method}.yaml"
  export OUT_ROOT="${OUT_BASE}/${method}"

  echo "[all-paper-baselines] START method=${method} run=${RUN_ID}"
  bash "${PROJECT_DIR}/comp/BCTI-comparisons/run_paper_baseline.sh"
  echo "[all-paper-baselines] DONE method=${method} run=${RUN_ID}"
}

# Sequential by design: never append '&'. Each full-parameter 6.9B run uses both GPUs.
run_one ga
run_one grad_diff
run_one ga_kl
run_one npo
run_one simnpo

echo "[all-paper-baselines] ALL COMPLETE run=${RUN_ID} output=${OUT_BASE}"
