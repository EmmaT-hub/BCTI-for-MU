# BCTI: Bidirectional Causal Tracing Intervention for LLM Unlearning

This repository contains the official implementation of **“BCTI: Bidirectional Causal Tracing Intervention for LLM Unlearning.”**

> Paper and OpenReview links will be added upon release.

## Overview

BCTI is a framework for knowledge unlearning in large language models. It first uses bidirectional causal tracing to locate computational paths associated with a target fact, then performs localized parameter updates while constraining changes to retained knowledge. The repository supports Pythia-6.9B and OLMo-7B and includes baseline and ablation experiments.

The pipeline consists of five stages:

1. Construct disjoint forget, retain, validation, and test splits.
2. Localize target knowledge with forward and reverse prompts.
3. Install LoRA adapters on causally relevant layers.
4. Select checkpoints using validation constraints.
5. Evaluate answer forgetting, knowledge retention, and causal-path reduction.

## Installation

Python 3.11 and a CUDA-enabled PyTorch environment are recommended:

```bash
git clone <repository-url>
cd BCTIcodes

conda create -n bcti python=3.11 -y
conda activate bcti

pip install torch transformers transformer-lens peft pandas pyyaml tqdm matplotlib pytest
```

Install the PyTorch build that matches your CUDA environment. Model weights are not distributed with this repository.

## Data and Models

The default fact dataset is located at:

```text
code/data/facts_small.jsonl
```

Each record contains a fact ID, prompt, answer, subject, relation, and forget/retain label. Validate the data splits before training:

```bash
python code/BCTI-main/preflight_splits.py \
  --config code/BCTI-main/config_early_stop.yaml
```

The current implementation supports:

- `EleutherAI/pythia-6.9b`
- `allenai/OLMo-7B` or a compatible local OLMo-7B checkpoint

Model, data, and output paths are configured through the corresponding YAML files and launch scripts. All bundled paths are relative to and remain within the repository root.

## Usage

### Run BCTI

Run the main Pythia experiment in a Slurm environment:

```bash
sbatch code/BCTI-main/submit_early_stop.sh
```

Run the OLMo-7B experiment:

```bash
sbatch olmo-BCTI/submit_early_stop.sh
```

To run a single target directly:

```bash
python code/BCTI-main/run.py \
  --config code/BCTI-main/config_early_stop.yaml \
  --forget-id f001 \
  --mode pcgrad \
  --out-dir outputs/f001
```

### Evaluation

The main experiment automatically performs validation, test evaluation, and causal auditing after checkpoint selection. To run the unified causal audit separately:

```bash
sbatch code/BCTI-main/submit_unified_causal_audit.sh
```

Results are written to the output directory specified by the configuration or launch script. Outputs include training logs, per-example metrics, causal-audit results, and `summary.json`.

## Reproducing the Results

Run the main BCTI experiment:

```bash
sbatch code/BCTI-main/submit_early_stop.sh
```

Run the internal ablations:

```bash
bash comp/BCTI-baselines/run_all_ablations.sh
```

Run the GA, GradDiff, GA+KL, NPO, and SimNPO baselines:

```bash
bash comp/BCTI-comparisons/run_all_comparisons.sh
```

Run the BCTI, ROME, and PALU cross-method ablations:

```bash
bash comp/BCTI-cross-method-ablations/run_all.sh
```

Run the independent causal audit:

```bash
sbatch code/BCTI-main/submit_unified_causal_audit.sh
```

## Citation

Please use the official BibTeX entry once the paper is released:

```bibtex
@inproceedings{bcti,
  title     = {BCTI: Bidirectional Causal Tracing Intervention for LLM Unlearning},
  author    = {To be updated},
  booktitle = {To be updated},
  year      = {To be updated}
}
```

## Acknowledgements

This project uses PyTorch, Hugging Face Transformers, TransformerLens, and PEFT. The comparison and ablation experiments cover related methods including ROME, NPO, SimNPO, and PALU. Reference information is provided in:

```text
comp/BCTI-comparisons/references.yaml
comp/BCTI-comparisons/references_npo_lora.yaml
```

We thank the authors and open-source communities behind these projects.

## License

A license file is not currently included. Add a `LICENSE` file and specify the applicable open-source license before public release.
