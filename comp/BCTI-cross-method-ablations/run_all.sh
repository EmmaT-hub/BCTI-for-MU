#!/usr/bin/env bash
# Single-GPU sequential launcher for the complete cross-method ablation matrix.
set -Eeuo pipefail
trap 'status=$?; echo "[cross-methods-all] FAILED at line ${LINENO} (exit ${status})" >&2; exit "$status"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"
PROJECT_DIR="${PROJECT_DIR:-.}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_BASE="${OUT_BASE:-outputs/50-targets/ablation/cross-methods/run-${RUN_TAG}}"
cd "$PROJECT_DIR"
echo "[cross-methods-all] output base: $PROJECT_DIR/$OUT_BASE"
OUT_ROOT="$OUT_BASE/bcti-locator-bcti-objective" bash comp/BCTI-cross-method-ablations/submit_bcti_bcti.sh
OUT_ROOT="$OUT_BASE/rome-locator-bcti-objective" bash comp/BCTI-cross-method-ablations/submit_rome_bcti.sh
OUT_ROOT="$OUT_BASE/bcti-locator-palu-objective" bash comp/BCTI-cross-method-ablations/submit_bcti_palu.sh
echo "[cross-methods-all] SUCCESS: $PROJECT_DIR/$OUT_BASE"
