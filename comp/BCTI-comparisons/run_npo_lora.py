from __future__ import annotations

"""Run npo lora utilities."""

import argparse
from dataclasses import asdict
import gc
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
COMMON_ROOT = CODE_ROOT / "common"
for path in (COMMON_ROOT, CODE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_utils import load_config  # noqa: E402
from data_utils import load_facts, split_facts  # noqa: E402
from tl_pythia_loader import dtype_from_name  # noqa: E402
from localization import discover_causal_layers  # noqa: E402
from lora import (  # noqa: E402
    install_causal_lora,
    restore_adapters,
    snapshot_adapters,
    trainable_named_parameters,
)
from methods_npo_lora import rwku_npo_from_log_probabilities  # noqa: E402
from run import (  # noqa: E402
    answer_statistics,
    build_reference,
    cycle_batches,
    ensure_empty,
    evaluate,
    load_experiment_config,
    make_limits,
)
from metrics import score, summarize  # noqa: E402
from splits import build_splits, manifest, rebalance_forget_splits_by_knowledge  # noqa: E402

METHOD_VERSION = "rwku-npo-bcti-lora-v1"
METHOD = "npo_lora"
CITATION_KEY = "jin2024rwku"


def validate_config(cfg: dict) -> None:
    for section in (
        "pipeline_config", "data_split", "localization", "lora",
        "evaluation", "training", "implementation", "baseline",
    ):
        if section not in cfg:
            raise ValueError(f"config is missing {section!r}")
    if str(cfg["baseline"]["method"]) != METHOD:
        raise ValueError(f"baseline.method must be {METHOD!r}")
    if str(cfg["baseline"]["citation_key"]) != CITATION_KEY:
        raise ValueError(f"baseline.citation_key must be {CITATION_KEY!r}")
    if float(cfg["baseline"]["beta"]) <= 0.0:
        raise ValueError("baseline.beta must be positive")
    if float(cfg["baseline"].get("retain_tradeoff_weight", 1.0)) < 0.0:
        raise ValueError("baseline.retain_tradeoff_weight cannot be negative")
    if str(cfg["implementation"]["parameterization"]) != "causal_lora":
        raise ValueError("implementation.parameterization must be causal_lora")
    localization = cfg["localization"]
    if int(localization["layer_count"]) <= 0:
        raise ValueError("localization.layer_count must be positive")
    if int(localization["candidate_pool_size"]) < int(localization["layer_count"]):
        raise ValueError("candidate_pool_size must be at least layer_count")
    lora_cfg = cfg["lora"]
    if int(lora_cfg["rank"]) <= 0 or float(lora_cfg["alpha"]) <= 0.0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not list(lora_cfg["target_modules"]):
        raise ValueError("lora.target_modules cannot be empty")
    train = cfg["training"]
    for key in ("max_steps", "eval_every", "forget_batch_size"):
        if int(train[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(train["learning_rate"]) <= 0.0 or float(train["max_grad_norm"]) <= 0.0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if int(train.get("warmup_steps", 0)) != 0 or str(train.get("schedule")) != "constant":
        raise ValueError("BCTI alignment requires a constant schedule without warmup")


def load_model(base_cfg: dict):
    required = 1
    if torch.cuda.device_count() < required:
        raise RuntimeError("BCTI-aligned NPO+LoRA requires one visible CUDA GPU")
    model_cfg = base_cfg["model"]
    dtype = dtype_from_name(str(model_cfg.get("dtype", "bfloat16")))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["local_path"],
        torch_dtype=dtype,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["local_path"],
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = str(base_cfg["model"].get("device", "cuda:0"))
    model.to(device)
    model.config.use_cache = False
    return model, tokenizer, device


def npo_loss(model, tokenizer, facts, reference, device: str, beta: float):
    current = torch.stack(
        [answer_statistics(model, tokenizer, fact, device)[1] for fact in facts]
    )
    original = torch.tensor(
        [reference[(fact.prompt, fact.answer)]["sequence_log_probability"] for fact in facts],
        device=current.device,
        dtype=current.dtype,
    )
    return rwku_npo_from_log_probabilities(current, original, beta)


def aggressive_checkpoint_priority(row: dict, retain_tradeoff_weight: float) -> tuple:
    """Balance forgetting progress against retention damage."""
    weight = float(retain_tradeoff_weight)
    return (
        not bool(row["evaluable"]),
        float(row["strong_forget_violation"]) + weight * float(row["retain_violation"]),
        float(row["forget_violation"]) + weight * float(row["retain_violation"]),
        -float(row.get("forget_min_direction_relative_drop", 0.0)),
        -float(row.get("forget_direction_macro_relative_drop", 0.0)),
        float(row["forget_cut_mean"]),
        int(row["step"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_npo_lora.yaml")
    parser.add_argument("--forget-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    validate_config(cfg)
    ensure_empty(args.out_dir)
    references = yaml.safe_load((HERE / "references_npo_lora.yaml").read_text(encoding="utf-8"))
    citation = references[CITATION_KEY]
    (args.out_dir / "00_resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (args.out_dir / "00_reference.json").write_text(
        json.dumps({**citation, "citation_key": CITATION_KEY}, indent=2), encoding="utf-8"
    )

    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    pipeline_cfg = load_config(Path(cfg["pipeline_config"]))
    base_cfg = load_config(Path(pipeline_cfg["base_config"]))
    all_facts = load_facts(base_cfg["data"]["facts_path"])
    all_forget, all_retain = split_facts(all_facts)
    target = [fact for fact in all_forget if fact.id == args.forget_id]
    if len(target) != 1:
        raise ValueError(f"expected exactly one forget fact for {args.forget_id}")
    splits = build_splits(target, all_retain, cfg["data_split"])

    
    layers, trace, site_summary, layer_summary = discover_causal_layers(
        base_cfg, splits, cfg["localization"]
    )
    trace.to_csv(args.out_dir / "02_fit_causal_trace.csv", index=False, encoding="utf-8-sig")
    site_summary.to_csv(args.out_dir / "03_site_scores.csv", index=False, encoding="utf-8-sig")
    layer_summary.to_csv(args.out_dir / "04_layer_scores.csv", index=False, encoding="utf-8-sig")

    model, tokenizer, device = load_model(base_cfg)
    initial_facts = [
        *splits["fit_forget"], *splits["fit_retain"],
        *splits["validation_forget"], *splits["validation_retain"],
        *splits["test_forget"], *splits["test_retain"],
    ]
    reference = build_reference(model, tokenizer, initial_facts, device)
    forget_candidates = [
        *splits["fit_forget"], *splits["validation_forget"], *splits["test_forget"]
    ]
    evaluation = cfg["evaluation"]
    known = {
        (fact.prompt, fact.answer): (
            bool(reference[(fact.prompt, fact.answer)]["greedy_exact"])
            if bool(evaluation["require_base_greedy_exact"])
            else bool(reference[(fact.prompt, fact.answer)]["greedy_exact"])
            or float(reference[(fact.prompt, fact.answer)]["answer_score"])
            >= float(evaluation["min_forget_base_answer_score"])
        )
        for fact in forget_candidates
    }
    base_scores = {
        (fact.prompt, fact.answer): float(reference[(fact.prompt, fact.answer)]["answer_score"])
        for fact in forget_candidates
    }
    split_audit = rebalance_forget_splits_by_knowledge(
        splits,
        known,
        base_scores,
        min_fit_per_direction=int(cfg["data_split"]["min_known_fit_per_direction"]),
        min_eval_per_direction=int(cfg["data_split"]["min_known_eval_per_direction"]),
        fail_on_unmet_direction_quotas=bool(
            cfg["data_split"].get("fail_on_unmet_direction_quotas", False)
        ),
    )
    manifest(splits).to_csv(
        args.out_dir / "01_split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    (args.out_dir / "01_split_audit.json").write_text(
        json.dumps(split_audit, indent=2), encoding="utf-8"
    )

    fit_forget = list(splits["fit_forget"])
    if not fit_forget:
        raise RuntimeError("fit forget split is empty")
    lora_cfg = cfg["lora"]
    installed = install_causal_lora(
        model,
        layers,
        [str(value) for value in lora_cfg["target_modules"]],
        rank=int(lora_cfg["rank"]),
        alpha=float(lora_cfg["alpha"]),
        dropout=float(lora_cfg["dropout"]),
    )
    named_parameters = trainable_named_parameters(model)
    parameters = [parameter for _, parameter in named_parameters]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("LoRA adapter parameters must remain float32")
    if {str(parameter.device) for parameter in parameters} != {str(torch.device(device))}:
        raise RuntimeError("all LoRA parameters must be on the BCTI training device")
    pd.DataFrame([asdict(value) for value in installed]).to_csv(
        args.out_dir / "05_installed_adapters.csv", index=False, encoding="utf-8-sig"
    )

    train = cfg["training"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train["learning_rate"]),
        betas=tuple(float(value) for value in train["betas"]),
        eps=float(train["adam_epsilon"]),
        weight_decay=float(train["weight_decay"]),
    )
    stream = cycle_batches(fit_forget, int(train["forget_batch_size"]), seed + 101)
    validation_facts = splits["validation_forget"] + splits["validation_retain"]
    limits = make_limits(evaluation)
    history = []
    best_key = None
    best_step = None
    best_state = None
    dense_steps = int(train.get("early_dense_eval_steps", 0))

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(train["max_steps"]) + 1):
        model.train()
        loss = npo_loss(
            model, tokenizer, next(stream), reference, device, float(cfg["baseline"]["beta"])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(train["max_grad_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step > dense_steps and step % int(train["eval_every"]):
            continue
        frame = evaluate(model, tokenizer, validation_facts, reference, device, evaluation)
        summary = summarize(frame, float(evaluation["tail_fraction"]))
        status = score(summary, limits)
        row = {
            "step": step,
            "loss": float(loss.detach().item()),
            "forget_loss": float(loss.detach().item()),
            "retain_loss": 0.0,
            "learning_rate": float(train["learning_rate"]),
            **summary,
            **status,
        }
        history.append(row)
        key = aggressive_checkpoint_priority(
            row, float(cfg["baseline"].get("retain_tradeoff_weight", 1.0))
        )
        if best_key is None or key < best_key:
            best_key = key
            best_step = step
            best_state = snapshot_adapters(named_parameters)
        print(
            f"[npo_lora] step={step}/{train['max_steps']} "
            f"min_direction_drop={summary['forget_min_direction_relative_drop']:.4f} "
            f"retain_kl={summary['retain_kl_mean']:.4f}",
            flush=True,
        )

    if best_state is None or best_step is None:
        raise RuntimeError("no validation checkpoint was selected")
    restore_adapters(named_parameters, best_state)
    torch.save(
        {
            "version": METHOD_VERSION,
            "forget_id": args.forget_id,
            "best_validation_step": best_step,
            "selected_layers": layers,
            "adapter_info": [asdict(value) for value in installed],
            "state_dict": best_state,
        },
        args.out_dir / "adapter_checkpoint.pt",
    )

    summaries = {}
    for split_name, facts in {
        "fit": splits["fit_forget"] + splits["fit_retain"],
        "validation": validation_facts,
        "test": splits["test_forget"] + splits["test_retain"],
    }.items():
        frame = evaluate(model, tokenizer, facts, reference, device, evaluation)
        summaries[split_name] = summarize(frame, float(evaluation["tail_fraction"]))
        frame.to_csv(args.out_dir / f"{split_name}_eval.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(history).to_csv(
        args.out_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    trainable_count = sum(parameter.numel() for parameter in parameters)
    result = {
        "version": METHOD_VERSION,
        "method": METHOD,
        "forget_id": args.forget_id,
        "citation_key": CITATION_KEY,
        "reference": citation,
        "objective": "RWKU Appendix H.1 Eq. (3): -log sigmoid(-beta log(p_theta/p_ref))",
        "hyperparameters": cfg["baseline"],
        "selected_layers": layers,
        "fairness_protocol": {
            "same_model_and_initial_checkpoint_as_bcti": True,
            "same_data_builder_and_prompt_splits_as_bcti": True,
            "same_fit_only_causal_localization_as_bcti": True,
            "same_lora_rank_alpha_dropout_targets_and_layer_count_as_bcti": True,
            "same_optimizer_update_budget_and_adamw_settings_as_bcti": True,
            "same_evaluation_constraints_as_bcti": True,
            "test_used_for_localization_or_checkpoint_selection": False,
            "only_training_objective_changed_to_rwku_npo": True,
            "checkpoint_selection_uses_forget_retain_tradeoff": True,
        },
        "full_parameter_finetuning": False,
        "trainable_parameter_count": trainable_count,
        "best_validation_step": best_step,
        "training": train,
        "split_audit": split_audit,
        "fit_summary": summaries["fit"],
        "validation_summary": summaries["validation"],
        "test_summary": summaries["test"],
        "test_status": score(summaries["test"], limits),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
