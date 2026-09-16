from __future__ import annotations

"""Run utilities."""

import argparse
import gc
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
COMMON_ROOT = CODE_ROOT / "common"
for path in (COMMON_ROOT, CODE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_utils import load_config  # noqa: E402
from data_utils import Fact, load_facts, split_facts  # noqa: E402
from tl_pythia_loader import dtype_from_name  # noqa: E402
from methods import (  # noqa: E402
    METHODS,
    METHOD_SPECS,
    ga_from_nll,
    ga_kl_from_values,
    grad_diff_from_nll,
    npo_from_log_probabilities,
    simnpo_from_log_probabilities,
)
from metrics import Constraints, score, summarize, validation_priority  # noqa: E402
from splits import (  # noqa: E402
    build_splits,
    causal_direction,
    manifest,
    rebalance_forget_splits_by_knowledge,
)

METHOD_VERSION = "bcti-paper-baselines-v1"


class CPUOffloadAdamW(torch.optim.Optimizer):
    """Keep AdamW state on the CPU."""

    def __init__(self, params, *, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("CPUOffloadAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    
                    state["exp_avg"] = torch.zeros_like(parameter, device="cpu")
                    state["exp_avg_sq"] = torch.zeros_like(parameter, device="cpu")
                state["step"] += 1
                step = state["step"]
                grad = parameter.grad.detach().to("cpu", dtype=parameter.dtype)
                value = parameter.detach().to("cpu", dtype=parameter.dtype)
                first, second = state["exp_avg"], state["exp_avg_sq"]
                value.mul_(1.0 - group["lr"] * group["weight_decay"])
                first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = second.sqrt().div_((1.0 - beta2**step) ** 0.5).add_(group["eps"])
                value.addcdiv_(first, denom, value=-group["lr"] / (1.0 - beta1**step))
                parameter.copy_(value.to(parameter.device))
        return loss


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: Path) -> dict:
    method_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    common_name = method_cfg.pop("common_config", None)
    if not common_name:
        raise ValueError("method config must declare common_config")
    common_path = (path.parent / common_name).resolve()
    common_cfg = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    cfg = deep_merge(common_cfg, method_cfg)
    cfg["resolved_common_config"] = str(common_path)
    return cfg


def validate_config(cfg: dict, cli_method: str) -> None:
    method = str(cfg["baseline"]["method"])
    if method != cli_method or method not in METHODS:
        raise ValueError(f"method mismatch or unsupported method: config={method!r}, cli={cli_method!r}")
    if cfg["baseline"]["citation_key"] != METHOD_SPECS[method].citation_key:
        raise ValueError("baseline citation_key does not match the method registry")
    train = cfg["training"]
    for key in ("max_steps", "eval_every", "forget_batch_size", "retain_batch_size"):
        if int(train[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(train["max_steps"]) % int(train["eval_every"]):
        raise ValueError("max_steps must be divisible by eval_every")
    if float(train["learning_rate"]) <= 0 or float(train["max_grad_norm"]) <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if METHOD_SPECS[method].needs_retain and float(cfg["baseline"]["retain_weight"]) <= 0:
        raise ValueError(f"{method} requires a positive retain_weight")


def ensure_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")


def load_model(base_cfg: dict):
    model_cfg = base_cfg["model"]
    required = 2
    if torch.cuda.device_count() < required:
        raise RuntimeError(
            f"full-parameter Pythia-6.9B baselines require at least {required} visible GPUs; "
            f"found {torch.cuda.device_count()}"
        )
    dtype = dtype_from_name(str(model_cfg.get("dtype", "bfloat16")))
    kwargs = dict(
        torch_dtype=dtype,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        low_cpu_mem_usage=True,
        device_map="balanced",
    )
    model = AutoModelForCausalLM.from_pretrained(model_cfg["local_path"], **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["local_path"],
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    device = str(model.get_input_embeddings().weight.device)
    return model, tokenizer, device, dtype


def token_ids(tokenizer, text: str, device: str) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.numel() == 0:
        raise ValueError(f"empty tokenization for {text!r}")
    return ids.to(device)


def answer_logits(model, tokenizer, fact: Fact, device: str):
    prompt = token_ids(tokenizer, fact.prompt, device)
    answer = token_ids(tokenizer, fact.answer, device)
    combined = torch.cat((prompt, answer), dim=1)
    logits = model(input_ids=combined).logits.float()[0]
    start = prompt.shape[1] - 1
    return logits[start : start + answer.shape[1]], answer[0]


def answer_statistics(model, tokenizer, fact: Fact, device: str):
    logits, targets = answer_logits(model, tokenizer, fact, device)
    token_logp = F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return -token_logp.mean(), token_logp.sum(), targets.numel(), logits, targets


def reference_forward_kl(model, tokenizer, fact: Fact, reference: dict, device: str):
    _, _, _, logits, _ = answer_statistics(model, tokenizer, fact, device)
    current_logp = F.log_softmax(logits, dim=-1)
    ref = reference[(fact.prompt, fact.answer)]["distributions"].to(device).clamp_min(1e-30)
    return F.kl_div(current_logp, ref.log(), reduction="batchmean", log_target=True)


def greedy_exact(model, tokenizer, fact: Fact, device: str) -> bool:
    generated = token_ids(tokenizer, fact.prompt, device)
    targets = token_ids(tokenizer, fact.answer, device)[0]
    predicted = []
    for _ in range(targets.numel()):
        next_token = int(model(input_ids=generated).logits[:, -1].argmax(-1).item())
        predicted.append(next_token)
        generated = torch.cat(
            (generated, torch.tensor([[next_token]], device=generated.device)), dim=1
        )
    return predicted == targets.tolist()


def answer_metrics(distributions: torch.Tensor, targets: torch.Tensor) -> dict:
    rows = torch.arange(targets.numel(), device=targets.device)
    probs = distributions[rows, targets].clamp_min(1e-30)
    accuracy = float((distributions.argmax(-1) == targets).float().mean().item())
    return {
        "answer_score": float(torch.exp(probs.log().mean()).item()),
        "answer_nll": float(-probs.log().mean().item()),
        "answer_token_accuracy": accuracy,
    }


@torch.no_grad()
def build_reference(model, tokenizer, facts: list[Fact], device: str) -> dict:
    model.eval()
    cache = {}
    for fact in facts:
        key = (fact.prompt, fact.answer)
        if key in cache:
            continue
        _, sequence_logp, length, logits, targets = answer_statistics(
            model, tokenizer, fact, device
        )
        distributions = F.softmax(logits, dim=-1)
        cache[key] = {
            "distributions": distributions.cpu(),
            "targets": targets.cpu(),
            "sequence_log_probability": float(sequence_logp.item()),
            "response_length": int(length),
            "greedy_exact": greedy_exact(model, tokenizer, fact, device),
            **answer_metrics(distributions, targets),
        }
    return cache


@torch.no_grad()
def evaluate(model, tokenizer, facts, reference, device, evaluation) -> pd.DataFrame:
    model.eval()
    rows = []
    for fact in facts:
        _, _, _, logits, targets = answer_statistics(model, tokenizer, fact, device)
        distributions = F.softmax(logits, dim=-1)
        current = answer_metrics(distributions, targets)
        base = reference[(fact.prompt, fact.answer)]
        ref = base["distributions"].to(device).clamp_min(1e-30)
        cur = distributions.clamp_min(1e-30)
        retain_kl = float((ref * (ref.log() - cur.log())).sum(-1).mean().item())
        eligible = bool(base["greedy_exact"])
        if not bool(evaluation["require_base_greedy_exact"]):
            eligible = eligible or float(base["answer_score"]) >= float(
                evaluation["min_forget_base_answer_score"]
            )
        rows.append(
            {
                "id": fact.id,
                "type": fact.type,
                "direction": causal_direction(fact),
                "prompt": fact.prompt,
                "answer": fact.answer,
                "base_answer_score": float(base["answer_score"]),
                "cut_answer_score": current["answer_score"],
                "base_first_token_prob": float(
                    base["distributions"][0, base["targets"][0]].item()
                ),
                "cut_first_token_prob": float(distributions[0, targets[0]].item()),
                "base_answer_token_accuracy": float(base["answer_token_accuracy"]),
                "cut_answer_token_accuracy": current["answer_token_accuracy"],
                "base_greedy_exact": bool(base["greedy_exact"]),
                "cut_greedy_exact": greedy_exact(model, tokenizer, fact, device),
                "forget_eligible": eligible if fact.type == "forget" else False,
                "answer_score_drop": float(base["answer_score"]) - current["answer_score"],
                "retain_kl": retain_kl if fact.type == "retain" else 0.0,
            }
        )
    return pd.DataFrame(rows)


def cycle_batches(values: list[Fact], size: int, seed: int):
    if not values:
        raise ValueError("cannot batch an empty split")
    rng = random.Random(seed)
    order = list(values)
    cursor = len(order)
    while True:
        batch = []
        while len(batch) < size:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            take = min(size - len(batch), len(order) - cursor)
            batch.extend(order[cursor : cursor + take])
            cursor += take
        yield batch


def method_loss(method, model, tokenizer, forget, retain, reference, device, baseline):
    forget_stats = [answer_statistics(model, tokenizer, fact, device) for fact in forget]
    forget_nll = torch.stack([value[0] for value in forget_stats])
    forget_logp = torch.stack([value[1] for value in forget_stats])
    lengths = torch.tensor([value[2] for value in forget_stats], device=forget_logp.device)
    zero = torch.zeros((), device=forget_logp.device)

    if method == "ga":
        forget_component = ga_from_nll(forget_nll)
        return forget_component, forget_component, zero
    if method == "npo":
        ref_logp = torch.tensor(
            [reference[(fact.prompt, fact.answer)]["sequence_log_probability"] for fact in forget],
            device=forget_logp.device,
            dtype=forget_logp.dtype,
        )
        forget_component = npo_from_log_probabilities(
            forget_logp, ref_logp, baseline["beta"]
        )
        return forget_component, forget_component, zero

    retain_nll = torch.stack(
        [answer_statistics(model, tokenizer, fact, device)[0] for fact in retain]
    )
    retain_component = retain_nll.mean()
    if method == "grad_diff":
        total = grad_diff_from_nll(forget_nll, retain_nll, baseline["retain_weight"])
    elif method == "ga_kl":
        retain_kls = torch.stack(
            [reference_forward_kl(model, tokenizer, fact, reference, device) for fact in retain]
        )
        retain_component = retain_kls.mean()
        total = ga_kl_from_values(forget_nll, retain_kls, baseline["retain_weight"])
    elif method == "simnpo":
        forget_component = simnpo_from_log_probabilities(
            forget_logp, lengths, baseline["beta"], baseline["gamma"]
        )
        total = forget_component + float(baseline["retain_weight"]) * retain_component
        return total, forget_component, retain_component
    else:
        raise AssertionError(method)
    return total, -forget_nll.mean(), retain_component


def make_limits(evaluation: dict) -> Constraints:
    value = evaluation["constraints"]
    return Constraints(
        min_forget_relative_drop=float(value["min_forget_relative_drop"]),
        min_strong_forget_relative_drop=float(value["min_strong_forget_relative_drop"]),
        max_forget_worst_score=float(value["max_forget_worst_answer_score"]),
        max_forget_increase=float(value["max_forget_answer_score_increase"]),
        min_eligible_forget_prompts=int(value["min_eligible_forget_prompts"]),
        min_eligible_forget_directions=int(value["min_eligible_forget_directions"]),
        min_eligible_forget_prompts_per_direction=int(value["min_eligible_forget_prompts_per_direction"]),
        min_direction_relative_drop=float(value["min_direction_relative_drop"]),
        min_strong_direction_relative_drop=float(value["min_strong_direction_relative_drop"]),
        max_retain_kl_mean=float(value["retain_kl_mean_budget"]),
        max_retain_kl_tail=float(value["retain_kl_tail_budget"]),
        max_retain_kl_max=float(value["retain_kl_max_budget"]),
        max_retain_drop=float(value["retain_max_drop"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--forget-id", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    validate_config(cfg, args.method)
    ensure_empty(args.out_dir)
    references = yaml.safe_load((HERE / "references.yaml").read_text(encoding="utf-8"))
    citation_key = METHOD_SPECS[args.method].citation_key
    citation = references[citation_key]
    (args.out_dir / "00_resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (args.out_dir / "00_reference.json").write_text(
        json.dumps({**citation, "citation_key": citation_key}, indent=2),
        encoding="utf-8",
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

    model, tokenizer, device, dtype = load_model(base_cfg)
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
    fit_retain = list(splits["fit_retain"])
    spec = METHOD_SPECS[args.method]
    if not fit_forget or (spec.needs_retain and not fit_retain):
        raise RuntimeError("method-required training split is empty")

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train = cfg["training"]
    optimizer = CPUOffloadAdamW(
        model.parameters(),
        lr=float(train["learning_rate"]),
        betas=train["betas"],
        eps=float(train["adam_epsilon"]),
        weight_decay=float(train["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train["warmup_steps"]),
        num_training_steps=int(train["max_steps"]),
    )
    forget_stream = cycle_batches(fit_forget, int(train["forget_batch_size"]), seed + 101)
    retain_stream = cycle_batches(fit_retain, int(train["retain_batch_size"]), seed + 211)
    limits = make_limits(evaluation)
    validation_facts = splits["validation_forget"] + splits["validation_retain"]
    history = []
    best_key = None
    best_step = None

    with tempfile.TemporaryDirectory(prefix=f"bcti-{args.method}-") as temporary:
        best_dir = Path(temporary) / "best"
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, int(train["max_steps"]) + 1):
            model.train()
            forget_batch = next(forget_stream)
            retain_batch = next(retain_stream) if spec.needs_retain else []
            total, forget_component, retain_component = method_loss(
                args.method,
                model,
                tokenizer,
                forget_batch,
                retain_batch,
                reference,
                device,
                cfg["baseline"],
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % int(train["eval_every"]):
                continue
            frame = evaluate(model, tokenizer, validation_facts, reference, device, evaluation)
            summary = summarize(frame, float(evaluation["tail_fraction"]))
            status = score(summary, limits)
            row = {
                "step": step,
                "loss": float(total.detach().item()),
                "forget_loss": float(forget_component.detach().item()),
                "retain_loss": float(retain_component.detach().item()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                **summary,
                **status,
            }
            history.append(row)
            key = validation_priority(row)
            if best_key is None or key < best_key:
                best_key = key
                best_step = step
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                model.save_pretrained(best_dir, safe_serialization=True)
            print(
                f"[{args.method}] step={step}/{train['max_steps']} "
                f"min_direction_drop={summary['forget_min_direction_relative_drop']:.4f} "
                f"retain_kl={summary['retain_kl_mean']:.4f}",
                flush=True,
            )

        del optimizer, scheduler, model
        gc.collect()
        torch.cuda.empty_cache()
        if best_step is None:
            raise RuntimeError("no validation checkpoint was selected")
        model = AutoModelForCausalLM.from_pretrained(
            best_dir,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map="balanced",
        )
        model.config.use_cache = False
        device = str(model.get_input_embeddings().weight.device)

        summaries = {}
        for split_name, facts in {
            "fit": splits["fit_forget"] + splits["fit_retain"],
            "validation": validation_facts,
            "test": splits["test_forget"] + splits["test_retain"],
        }.items():
            frame = evaluate(model, tokenizer, facts, reference, device, evaluation)
            summaries[split_name] = summarize(frame, float(evaluation["tail_fraction"]))
            frame.to_csv(
                args.out_dir / f"{split_name}_eval.csv", index=False, encoding="utf-8-sig"
            )

    pd.DataFrame(history).to_csv(
        args.out_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "version": METHOD_VERSION,
        "method": args.method,
        "forget_id": args.forget_id,
        "citation_key": spec.citation_key,
        "reference": citation,
        "objective": spec.objective,
        "hyperparameters": cfg["baseline"],
        "fairness_protocol": {
            "same_model_and_initial_checkpoint": True,
            "same_data_builder_as_bcti": True,
            "same_prompt_splits_and_knowledge_rebalancing_as_bcti": True,
            "same_evaluation_and_constraints_as_bcti": True,
            "same_optimizer_update_budget_across_baselines": True,
            "test_used_for_checkpoint_selection": False,
            "contains_bcti_localization_or_causal_objective": False,
        },
        "full_parameter_finetuning": True,
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


if __name__ == "__main__":
    main()
