#!/usr/bin/env bash
# One-click, single-GPU, sequential launcher for all four BCTI-main ablations.

set -Eeuo pipefail
trap 'status=$?; echo "[run-all-ablations] FAILED at line ${LINENO} (exit ${status})" >&2; exit "$status"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"

PROJECT_DIR="${PROJECT_DIR:-.}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_BASE="${OUT_BASE:-outputs/50-targets/ablation/internal/unified-audit-single-gpu-${RUN_TAG}}"

cd "$PROJECT_DIR"

echo "[run-all-ablations] single-GPU sequential mode"
echo "[run-all-ablations] output base: $PROJECT_DIR/$OUT_BASE"

echo "[run-all-ablations] 1/4 forward-only localization"
OUT_ROOT="$OUT_BASE/forward-only-localization" \
  bash comp/BCTI-baselines/submit_forward_only_localization.sh

echo "[run-all-ablations] 2/4 reverse-only localization"
OUT_ROOT="$OUT_BASE/reverse-only-localization" \
  bash comp/BCTI-baselines/submit_reverse_only_localization.sh

echo "[run-all-ablations] 3/4 causal-score only"
OUT_ROOT="$OUT_BASE/causal-score-only" \
  bash comp/BCTI-baselines/submit_causal_score_only.sh

echo "[run-all-ablations] 4/4 answer-objective only"
OUT_ROOT="$OUT_BASE/answer-objective-only" \
  bash comp/BCTI-baselines/submit_answer_objective_only.sh

echo "[run-all-ablations] SUCCESS: all four ablations completed"
echo "[run-all-ablations] results: $PROJECT_DIR/$OUT_BASE"
