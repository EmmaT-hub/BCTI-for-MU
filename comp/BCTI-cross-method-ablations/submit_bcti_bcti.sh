#!/usr/bin/env bash
# One directly submitted single-GPU job. Runs all fifty facts sequentially.
#SBATCH --job-name=bcti_ref
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --chdir=.
#SBATCH --output=./comp/BCTI-cross-method-ablations/bcti_ref-%j.out
#SBATCH --error=./comp/BCTI-cross-method-ablations/bcti_ref-%j.err

set -Eeuo pipefail
trap 'status=$?; echo "[cross-methods] FAILED at line ${LINENO} (exit ${status})" >&2; exit "$status"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"

PROJECT_DIR="${PROJECT_DIR:-.}"
CONDA_ENV="${CONDA_ENV:-unlearn}"
CONFIG="${CONFIG:-comp/BCTI-cross-method-ablations/config_bcti_bcti.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/50-targets/ablation/cross-methods/bcti-locator-bcti-objective}"
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
required_files=(
  "$CONFIG"
  comp/BCTI-cross-method-ablations/run.py
  comp/BCTI-cross-method-ablations/rome_localization.py
  comp/BCTI-cross-method-ablations/palu_loss.py
  comp/BCTI-cross-method-ablations/lora.py
  comp/BCTI-cross-method-ablations/metrics.py
  comp/BCTI-cross-method-ablations/splits.py
  comp/BCTI-cross-method-ablations/preflight_splits.py
  comp/BCTI-cross-method-ablations/test_cross_method.py
  code/common/causal_tracing.py
  code/common/tl_pythia_loader.py
  code/knowledge_group/pipeline.py
  code/single_fact/config.yaml
  code/knowledge_group/base_config.yaml
  code/data/facts.jsonl
)
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "[cross-methods] missing required file: $PROJECT_DIR/$required" >&2
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
  echo "[cross-methods] conda was not found" >&2
  exit 6
fi
conda activate "$CONDA_ENV"

python -c 'import os, torch, pandas, yaml, transformers, transformer_lens; assert torch.cuda.is_available(), "CUDA is unavailable"; print("[cross-methods] torch:", torch.__version__); print("[cross-methods] visible_gpus:", torch.cuda.device_count(), "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", "unset")); print("[cross-methods] gpu0:", torch.cuda.get_device_name(0))'
python -m unittest discover -s comp/BCTI-cross-method-ablations -p 'test_cross_method.py' -v
python comp/BCTI-cross-method-ablations/preflight_splits.py --config "$CONFIG"
python comp/BCTI-cross-method-ablations/run.py --help >/dev/null
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
fi

echo "[cross-methods] variant=bcti-locator-bcti-objective facts=${FORGET_IDS[*]} modes=${MODES[*]}"
echo "[cross-methods] config=$CONFIG output=$OUT_ROOT"
for forget_id in "${FORGET_IDS[@]}"; do
  for mode in "${MODES[@]}"; do
    out_dir="$OUT_ROOT/$forget_id/$mode"
    if [[ -e "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      if [[ -f "$out_dir/summary.json" ]]; then
        echo "[cross-methods] already complete; skipping $forget_id/$mode"
        continue
      fi
      echo "[cross-methods] partial output requires a new OUT_ROOT: $out_dir" >&2
      exit 5
    fi
    mkdir -p "$out_dir"
    echo "[cross-methods] starting forget=$forget_id mode=$mode output=$out_dir"
    python -u comp/BCTI-cross-method-ablations/run.py \
      --config "$CONFIG" \
      --forget-id "$forget_id" \
      --mode "$mode" \
      --out-dir "$out_dir"

    python - "$out_dir/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("version") != "BCTI-cross-methods-v1":
    raise SystemExit(f"unexpected method version: {summary.get('version')}")
print(
    "[cross-methods] completed",
    summary.get("ablation_name"),
    "localization=", summary.get("localization_policy"),
    "objective=", summary.get("forget_objective_policy"),
    "test_evaluable=", summary["test_status"]["evaluable"],
    "test_feasible=", summary["test_status"]["feasible"],
)
PY
  done
done

echo "[cross-methods] SUCCESS variant=bcti-locator-bcti-objective output=$OUT_ROOT"
