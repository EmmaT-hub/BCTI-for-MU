# BCTI for Machine Unlearning

BCTI is a codebase for targeted knowledge unlearning in causal language models. It combines bidirectional causal tracing, localized LoRA updates, retention constraints, and structured evaluation.

## Features

- Forward and reverse causal tracing for target localization
- Localized LoRA training on selected model layers
- Prompt-disjoint forget, retain, validation, and test splits
- Checkpoint selection with forgetting and retention constraints
- Baseline and ablation implementations
- Support for Pythia and compatible OLMo causal language models

## Repository Structure

```text
code/BCTI-main/                    Main BCTI training pipeline
code/common/                       Shared tracing and data utilities
code/data/facts.jsonl              Example fact dataset
comp/BCTI-baselines/               Internal ablations
comp/BCTI-comparisons/             Comparison methods
comp/BCTI-cross-method-ablations/  Cross-method ablations
olmo-BCTI/                         OLMo-compatible pipeline
```

## Installation

Python 3.11 and a CUDA-enabled PyTorch environment are recommended.

```bash
git clone https://github.com/EmmaT-hub/BCTI-for-MU.git
cd BCTI-for-MU

conda create -n bcti python=3.11 -y
conda activate bcti
pip install torch transformers transformer-lens peft pandas pyyaml tqdm matplotlib pytest
```

Install the PyTorch build appropriate for your CUDA environment. Model weights are not included.

## Configuration

Model, dataset, output, and training settings are defined in YAML files. The default dataset path is:

```text
code/data/facts.jsonl
```

Configure local model paths before running an experiment:

- `code/knowledge_group/base_config.yaml` for Pythia
- `olmo-BCTI/config_early_stop.yaml` for OLMo

All bundled project paths are relative to the repository root.

## Data Format

Each JSONL record contains the fields required by the split and evaluation pipeline, including a fact identifier, prompt, answer, subject, relation, and forget/retain label.

Validate the configured splits with:

```bash
python code/BCTI-main/preflight_splits.py \
  --config code/BCTI-main/config_early_stop.yaml
```

## Usage

Run one target directly:

```bash
python code/BCTI-main/run.py \
  --config code/BCTI-main/config_early_stop.yaml \
  --forget-id f001 \
  --mode pcgrad \
  --out-dir outputs/f001
```

Run the main Slurm launcher:

```bash
sbatch code/BCTI-main/submit_early_stop.sh
```

Run the OLMo launcher:

```bash
sbatch olmo-BCTI/submit_early_stop.sh
```

Run comparison and ablation suites:

```bash
bash comp/BCTI-baselines/run_all_ablations.sh
bash comp/BCTI-comparisons/run_all_comparisons.sh
bash comp/BCTI-cross-method-ablations/run_all.sh
```

NPO+LoRA has a separate launcher:

```bash
sbatch comp/BCTI-comparisons/submit_npo_lora.sh
```

## Evaluation Outputs

Each run writes its artifacts to the configured output directory. Depending on the experiment, outputs include:

- resolved configuration
- split manifests and audits
- causal tracing and layer scores
- training history
- validation and test metrics
- adapter checkpoints
- summary JSON files

## Tests

Run the available unit tests from the repository root:

```bash
python -m unittest discover -s code/BCTI-main -p "test_*.py"
python -m unittest discover -s comp/BCTI-baselines -p "test_*.py"
python -m unittest discover -s comp/BCTI-comparisons -p "test_*.py"
python -m unittest discover -s comp/BCTI-cross-method-ablations -p "test_*.py"
python -m unittest discover -s olmo-BCTI -p "test_*.py"
```
