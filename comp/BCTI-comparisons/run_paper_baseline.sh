#!/usr/bin/env bash
set -Eeuo pipefail

trap 'status=$?; echo "[paper-baseline] FAILED line=${LINENO} exit=${status}" >&2; exit "${status}"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"

PROJECT_DIR="${PROJECT_DIR:-.}"
CONDA_ENV="${CONDA_ENV:-unlearn}"

: "${METHOD:?METHOD is required}"
: "${CONFIG:?CONFIG is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"

cd "${PROJECT_DIR}"

required_files=(
  "${CONFIG}"
  "comp/BCTI-comparisons/config_common.yaml"
  "comp/BCTI-comparisons/references.yaml"
  "comp/BCTI-comparisons/run.py"
  "comp/BCTI-comparisons/methods.py"
  "comp/BCTI-comparisons/metrics.py"
  "comp/BCTI-comparisons/splits.py"
  "code/single_fact/config.yaml"
  "code/knowledge_group/base_config.yaml"
  "code/data/facts.jsonl"
)
for file in "${required_files[@]}"; do
  [[ -f "${file}" ]] || { echo "[paper-baseline] missing: ${PROJECT_DIR}/${file}" >&2; exit 4; }
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[paper-baseline] conda not found" >&2
  exit 6
fi
conda activate "${CONDA_ENV}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -c 'import torch, transformers, pandas, yaml; assert torch.cuda.device_count() >= 2, "two visible GPUs are required"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
python "comp/BCTI-comparisons/run.py" --help >/dev/null

if [[ -n "${FORGET_IDS_OVERRIDE:-}" ]]; then
  read -r -a forget_ids <<< "$FORGET_IDS_OVERRIDE"
else
  forget_ids=(f001 f002 f003 f004 f005 f006 f007 f008 f009 f010 f011 f012 f013 f014 f015 f016 f017 f018 f019 f020 f021 f022 f023 f024 f025 f026 f027 f028 f029 f030 f031 f032 f033 f034 f035 f036 f037 f038 f039 f040 f041 f042 f043 f044 f045 f046 f047 f048 f049 f050)
fi
for forget_id in "${forget_ids[@]}"; do
  out_dir="${OUT_ROOT}/${forget_id}/${METHOD}"
  [[ ! -e "${out_dir}" ]] || { echo "[paper-baseline] output exists: ${out_dir}" >&2; exit 5; }
  mkdir -p "${out_dir}"
  python -u "comp/BCTI-comparisons/run.py" \
    --config "${CONFIG}" \
    --forget-id "${forget_id}" \
    --method "${METHOD}" \
    --out-dir "${out_dir}"
  [[ -s "${out_dir}/summary.json" ]] || { echo "[paper-baseline] summary missing: ${out_dir}" >&2; exit 7; }
  echo "[paper-baseline] complete method=${METHOD} fact=${forget_id}"
done
