#!/usr/bin/env bash
# Standalone NPO+LoRA comparison job. Do not add this launcher to run_all_comparisons.sh.
#SBATCH --job-name=npo_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# TransformerLens conversion briefly holds HF bf16, converted fp32, and TL weights.
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --chdir=.
#SBATCH --output=./comp/BCTI-comparisons/npo_lora_%j.out
#SBATCH --error=./comp/BCTI-comparisons/npo_lora_%j.err

set -Eeuo pipefail
trap 'status=$?; echo "[npo-lora] FAILED line=${LINENO} exit=${status}" >&2; exit "${status}"' ERR

# Conda activation may read PS1 even in a non-interactive shell with nounset enabled.
export PS1="${PS1:-}"

PROJECT_DIR="${PROJECT_DIR:-.}"
CONDA_ENV="${CONDA_ENV:-unlearn}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="${CONFIG:-comp/BCTI-comparisons/config_npo_lora.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/50-targets/baselines/${RUN_ID}/npo_lora}"
if [[ -n "${FORGET_IDS_OVERRIDE:-}" ]]; then
  read -r -a FORGET_IDS <<< "${FORGET_IDS_OVERRIDE}"
else
  FORGET_IDS=(f001 f002 f003 f004 f005 f006 f007 f008 f009 f010 f011 f012 f013 f014 f015 f016 f017 f018 f019 f020 f021 f022 f023 f024 f025 f026 f027 f028 f029 f030 f031 f032 f033 f034 f035 f036 f037 f038 f039 f040 f041 f042 f043 f044 f045 f046 f047 f048 f049 f050)
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Bound glibc arena replication during the CPU-heavy HF -> TransformerLens conversion.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cd "${PROJECT_DIR}"

# Deliberately excludes local papers: the server job needs code/config/data/model only.
required_files=(
  "${CONFIG}"
  comp/BCTI-comparisons/config_common.yaml
  comp/BCTI-comparisons/run_npo_lora.py
  comp/BCTI-comparisons/run.py
  comp/BCTI-comparisons/localization.py
  comp/BCTI-comparisons/lora.py
  comp/BCTI-comparisons/methods.py
  comp/BCTI-comparisons/methods_npo_lora.py
  comp/BCTI-comparisons/metrics.py
  comp/BCTI-comparisons/splits.py
  comp/BCTI-comparisons/references_npo_lora.yaml
  comp/BCTI-comparisons/test_npo_lora.py
  code/common/causal_tracing.py
  code/common/tl_pythia_loader.py
  code/single_fact/config.yaml
  code/knowledge_group/base_config.yaml
  code/data/facts.jsonl
)
for required in "${required_files[@]}"; do
  [[ -f "${required}" ]] || { echo "[npo-lora] missing: ${PROJECT_DIR}/${required}" >&2; exit 4; }
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
  echo "[npo-lora] conda not found" >&2
  exit 6
fi
conda activate "${CONDA_ENV}"

python -c 'import os, resource, torch, pandas, yaml, transformers, transformer_lens; assert torch.cuda.device_count() == 1, "select exactly one visible GPU"; limit = resource.getrlimit(resource.RLIMIT_AS)[0]; print("[npo-lora] gpu:", torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", "unset")); print("[npo-lora] slurm_mem_per_node_mb=", os.environ.get("SLURM_MEM_PER_NODE", "unset"), "rlimit_as=", limit)'
python -m unittest discover -s comp/BCTI-comparisons -p 'test_npo_lora.py' -v
python comp/BCTI-comparisons/run_npo_lora.py --help >/dev/null

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
fi

echo "[npo-lora] run=${RUN_ID} config=${CONFIG} output=${OUT_ROOT}"
for forget_id in "${FORGET_IDS[@]}"; do
  out_dir="${OUT_ROOT}/${forget_id}/npo_lora"
  [[ ! -e "${out_dir}" ]] || { echo "[npo-lora] output exists: ${out_dir}" >&2; exit 5; }
  mkdir -p "${out_dir}"
  echo "[npo-lora] START fact=${forget_id}"
  python -u comp/BCTI-comparisons/run_npo_lora.py \
    --config "${CONFIG}" \
    --forget-id "${forget_id}" \
    --out-dir "${out_dir}"
  [[ -s "${out_dir}/summary.json" ]] || { echo "[npo-lora] summary missing: ${out_dir}" >&2; exit 7; }
  [[ -s "${out_dir}/adapter_checkpoint.pt" ]] || { echo "[npo-lora] checkpoint missing: ${out_dir}" >&2; exit 8; }
  echo "[npo-lora] DONE fact=${forget_id}"
done

echo "[npo-lora] ALL COMPLETE run=${RUN_ID} output=${OUT_ROOT}"
