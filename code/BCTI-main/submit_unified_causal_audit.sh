#!/usr/bin/env bash
# One directly submitted job, one manually allocated GPU.
#SBATCH --job-name=bcti_main_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --chdir=.
#SBATCH --output=./code/BCTI-main/%j.out
#SBATCH --error=./code/BCTI-main/%j.err

set -Eeuo pipefail
trap 'status=$?; echo "[BCTI-main] FAILED at line ${LINENO} (exit ${status})" >&2; exit "$status"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"

PROJECT_DIR="${PROJECT_DIR:-.}"
CONDA_ENV="${CONDA_ENV:-unlearn}"
CONFIG="${CONFIG:-code/BCTI-main/config_early_stop.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/50-targets/BCTI-main-unified-causal-audit}"

if [[ -n "${FORGET_IDS_OVERRIDE:-}" ]]; then
  read -r -a FORGET_IDS <<< "$FORGET_IDS_OVERRIDE"
else
  FORGET_IDS=(f001 f002 f003 f004 f005 f006 f007 f008 f009 f010 f011 f012 f013 f014 f015 f016 f017 f018 f019 f020 f021 f022 f023 f024 f025 f026 f027 f028 f029 f030 f031 f032 f033 f034 f035 f036 f037 f038 f039 f040 f041 f042 f043 f044 f045 f046 f047 f048 f049 f050)
fi
MODES=(pcgrad)

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$PROJECT_DIR"

RUN_IDS=("${FORGET_IDS[@]}")
RUN_MODES=("${MODES[@]}")
launch_kind="direct single-GPU sequential job"

required_files=(
  "$CONFIG"
  code/BCTI-main/run.py
  code/BCTI-main/lora.py
  code/BCTI-main/metrics.py
  code/BCTI-main/splits.py
  code/BCTI-main/__init__.py
  code/BCTI-main/preflight_splits.py
  code/BCTI-main/test_bcti_main.py
  code/common/causal_tracing.py
  code/common/tl_pythia_loader.py
  code/knowledge_group/pipeline.py
  code/single_fact/config.yaml
  code/knowledge_group/base_config.yaml
  code/data/facts_small.jsonl
)
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "[BCTI-main] missing required file: $PROJECT_DIR/$required" >&2
    exit 4
  fi
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[BCTI-main] conda was not found" >&2
  exit 6
fi
conda activate "$CONDA_ENV"

python -c 'import os, torch, pandas, yaml, transformers, transformer_lens; assert torch.cuda.is_available(), "CUDA is unavailable"; assert torch.cuda.device_count() >= 1, "no visible CUDA device"; print("[BCTI-main] torch:", torch.__version__); print("[BCTI-main] visible_gpus:", torch.cuda.device_count(), "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", "unset")); print("[BCTI-main] gpu0:", torch.cuda.get_device_name(0))'
python -m unittest discover -s code/BCTI-main -p 'test_bcti_main.py' -v
python code/BCTI-main/preflight_splits.py --config "$CONFIG"
python code/BCTI-main/run.py --help >/dev/null
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
fi

echo "[BCTI-main] launch=$launch_kind facts=${RUN_IDS[*]} modes=${RUN_MODES[*]} localization=fresh"
echo "[BCTI-main] stopping_strategy=causal-threshold config=$CONFIG output=$OUT_ROOT"
for forget_id in "${RUN_IDS[@]}"; do
  for mode in "${RUN_MODES[@]}"; do
    out_dir="$OUT_ROOT/$forget_id/$mode"
    if [[ -e "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "[BCTI-main] refusing to reuse any existing output: $PROJECT_DIR/$out_dir" >&2
      exit 5
    fi
    mkdir -p "$out_dir"
    echo "[BCTI-main] starting forget=$forget_id mode=$mode output=$out_dir"
    echo "[BCTI-main] tracing and localizing from scratch"
    python -u code/BCTI-main/run.py \
      --config "$CONFIG" \
      --forget-id "$forget_id" \
      --mode "$mode" \
      --out-dir "$out_dir"

    python - "$out_dir/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("version") != "BCTI-main":
    raise SystemExit(f"unexpected method version: {summary.get('version')}")
baseline_causal = summary["baseline_causal_objective"]["target_causal_score"]
edited_causal = summary["edited_causal_objective"]["target_causal_score"]
causal_drop = summary["causal_score_drop"]
audit = summary.get("test_causal_audit_reductions")
if not isinstance(audit, dict):
    raise SystemExit("missing method-independent test causal audit")
print(
    "[causal-audit] macro_drop="
    f"{audit.get('direction_macro_relative_drop')} "
    f"min_direction_drop={audit.get('min_direction_relative_drop')} "
    f"remaining_max={audit.get('remaining_max_direction_causal_score')}"
)
print(
    f"[BCTI-main] causal score {baseline_causal:.6f} -> {edited_causal:.6f}; "
    f"drop={causal_drop:.6f}"
)
if causal_drop <= 0.0:
    print(
        f"[BCTI-main] WARNING: selected checkpoint did not lower causal score: {sys.argv[1]}",
        file=sys.stderr,
    )
if not summary["test_status"]["evaluable"]:
    print(f"[BCTI-main] WARNING: test split is not evaluable: {sys.argv[1]}", file=sys.stderr)
elif not summary["test_status"]["feasible"]:
    if summary["test_status"]["strong_feasible"]:
        print(f"[BCTI-main] STRONG success (strict threshold not met): {sys.argv[1]}")
    else:
        print(f"[BCTI-main] WARNING: test constraints not met: {sys.argv[1]}", file=sys.stderr)
PY
    echo "[BCTI-main] completed forget=$forget_id mode=$mode"
  done
done

echo "[BCTI-main] completed launch=$launch_kind"
