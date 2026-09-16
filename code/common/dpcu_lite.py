
from __future__ import annotations
"""Select activation cuts that suppress targets while protecting retained knowledge."""


import argparse

import json

from dataclasses import asdict, dataclass, replace

from pathlib import Path


import pandas as pd

import torch

import torch.nn.functional as F

from tqdm import tqdm


from activation_patching import PatchSpec, clean_value_from_cache, run_with_multi_patch

from circuit_breaker import (
    
    CausalUnit,
    
    RELATION_PATTERNS,
    
    build_causal_units,
    
    detector_probability,
    
    save_detector,
    
    train_target_detector,

)

from config_utils import ensure_dir, load_config

from data_utils import Fact, load_facts, neutral_prompt, split_facts

from eval_target_prob import answer_token_ids

from tl_pythia_loader import load_from_config

from visualization import plot_cut_eval, plot_strength_sweep



@dataclass(frozen=True)

class CutCandidate:
    """Store one cut candidate and its causal and retain-effect scores."""
    
    layer: int
    
    site: str
    
    score: float
    
    forget_restore: float = 0.0
    
    retain_restore: float = 0.0
    
    single_objective: float | None = None
    
    single_forget_drop: float | None = None
    
    single_retain_kl: float | None = None
    
    single_max_retain_prob_drop: float | None = None
    
    single_feasible: bool | None = None

    
    @property
    
    def spec(self) -> PatchSpec:
        
        """Convert this candidate into an activation patch specification."""
        
        return PatchSpec(layer=self.layer, site=self.site, token_pos=-1)



def cache_names_filter(name: str) -> bool:
    
    """Return whether a hook belongs in the activation cache."""
    
    return (
        
        name.endswith("hook_resid_pre")
        
        or name.endswith("hook_attn_out")
        
        or name.endswith("hook_mlp_out")
    
    )



@torch.no_grad()

def next_token_distribution(model, prompt: str) -> torch.Tensor:
    
    """Return the next-token distribution for a prompt."""
    
    tokens = model.to_tokens(prompt)
    
    logits = model(tokens)[:, -1, :].float()
    
    return F.softmax(logits, dim=-1).detach()



@torch.no_grad()

def target_first_prob(model, prompt: str, answer: str) -> float:
    
    """Return the probability of the target answer?s first token."""
    
    dist = next_token_distribution(model, prompt)
    
    target_id = int(answer_token_ids(model, answer)[0].item())
    
    return float(dist[0, target_id].item())



def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute KL divergence between two probability distributions."""
    
    p = p.float().clamp_min(eps)
    
    q = q.float().clamp_min(eps)
    
    return float((p * (p.log() - q.log())).sum().item())



@torch.no_grad()




def build_neutral_bank(
    model,
    facts: list[Fact],
    candidates: list[CutCandidate],
    strategy: str = "unknown_mean",
) -> dict[PatchSpec, torch.Tensor]:
    """Cache relation-matched neutral activations for candidate sites."""
    if not facts:
        raise ValueError("build_neutral_bank requires at least one control fact")
    if not candidates:
        raise ValueError("build_neutral_bank requires at least one candidate")
    strategy = str(strategy).strip().lower()
    if strategy not in {"unknown_mean", "unknown_median", "zero"}:
        raise ValueError(
            "dpcu.neutral_strategy must be one of: "
            "unknown_mean, unknown_median, zero"
        )
    control_facts = [fact for fact in facts if fact.type == "retain"]
    if strategy != "zero" and not control_facts:
        raise ValueError(
            f"{strategy} requires retain control facts; refusing to build a "
            "neutral bank from forget knowledge"
        )
    source_facts = facts[:1] if strategy == "zero" else control_facts
    
    specs = sorted({c.spec for c in candidates}, key=lambda x: (x.layer, x.site))
    
    values: dict[PatchSpec, list[torch.Tensor]] = {spec: [] for spec in specs}
    
    for fact in source_facts:
        
        prompt = neutral_prompt(fact)
        
        _, cache = model.run_with_cache(model.to_tokens(prompt), names_filter=cache_names_filter)
        
        for spec in specs:
            
            values[spec].append(clean_value_from_cache(cache, spec.layer, spec.site, spec.token_pos).detach().cpu())
    
    bank = {}
    for spec, samples in values.items():
        stacked = torch.stack(samples, dim=0)
        if strategy == "zero":
            bank[spec] = torch.zeros_like(stacked[0])
        elif strategy == "unknown_median":
            bank[spec] = stacked.median(dim=0).values
        else:
            bank[spec] = stacked.mean(dim=0)
    return bank



@torch.no_grad()

def run_cut_logits(
    
    model,
    
    prompt: str,
    
    cut_set: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    cut_strength: float = 1.0,

) -> torch.Tensor:
    """Run a prompt with selected activations replaced by neutral values."""
    
    tokens = model.to_tokens(prompt)
    
    replacements = {cand.spec: neutral_bank[cand.spec] for cand in cut_set if cand.spec in neutral_bank}
    
    if not replacements:
        
        return model(tokens)
    
    return run_with_multi_patch(model, tokens, replacements, strength=cut_strength)



@torch.no_grad()

def target_prob_with_cut(
    
    model,
    
    fact: Fact,
    
    cut_set: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    cut_strength: float = 1.0,

) -> float:
    
    """Measure target recall after applying a cut set."""
    
    logits = run_cut_logits(model, fact.prompt, cut_set, neutral_bank, cut_strength=cut_strength)
    
    target_id = int(answer_token_ids(model, fact.answer)[0].item())
    
    probs = F.softmax(logits[:, -1, :].float(), dim=-1)
    
    return float(probs[0, target_id].item())



@torch.no_grad()

def retain_metrics_with_cut(
    
    model,
    
    retain_facts: list[Fact],
    
    cut_set: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    cut_strength: float = 1.0,

) -> tuple[float, float]:
    """Measure probability and distribution shifts on retained facts."""
    
    kl_vals = []
    
    prob_drops = []
    
    for fact in retain_facts:
        
        base = next_token_distribution(model, fact.prompt)
        
        logits = run_cut_logits(model, fact.prompt, cut_set, neutral_bank, cut_strength=cut_strength)
        
        cut = F.softmax(logits[:, -1, :].float(), dim=-1).detach()
        
        target_id = int(answer_token_ids(model, fact.answer)[0].item())
        
        prob_drops.append(float(base[0, target_id].item() - cut[0, target_id].item()))
        
        kl_vals.append(kl_div(base, cut))
    
    mean_kl = sum(kl_vals) / max(len(kl_vals), 1)
    
    max_prob_drop = max(prob_drops) if prob_drops else 0.0
    
    return mean_kl, max_prob_drop



@torch.no_grad()




def estimate_path_gate_effects(
    
    model,
    
    forget_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    candidates: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    gate_strengths: list[float],
    
    min_forget_drop: float,
    
    max_retain_kl: float,
    
    max_retain_prob_drop: float,
    
    min_specificity: float,

) -> pd.DataFrame:
    """Estimate each path?s effect on detector-gate activations."""
    
    base_forget = {
        
        fact.id: target_first_prob(model, fact.prompt, fact.answer)
        
        for fact in forget_facts
    
    }
    
    strengths = sorted({float(value) for value in gate_strengths if float(value) > 0.0})
    
    if not strengths:
        
        raise ValueError("path_gate.intervention_strengths must contain a positive value")
    
    rows = []
    
    for candidate in tqdm(candidates, desc="path-gate effect estimation", leave=False):
        
        
        
        
        
        
        
        
        
        for gate_strength in strengths:
            
            cut_probs = {
                
                fact.id: target_prob_with_cut(
                    
                    model, fact, [candidate], neutral_bank, cut_strength=gate_strength
                
                )
                
                for fact in forget_facts
            
            }
            
            drops = [base_forget[fact_id] - cut_probs[fact_id] for fact_id in base_forget]
            
            forget_drop_mean = sum(drops) / max(len(drops), 1)
            
            forget_drop_positive_mean = sum(max(value, 0.0) for value in drops) / max(len(drops), 1)
            
            forget_max_increase = max([-value for value in drops], default=0.0)
            
            retain_kl, retain_max_drop = retain_metrics_with_cut(
                
                model,
                
                retain_facts,
                
                [candidate],
                
                neutral_bank,
                
                cut_strength=gate_strength,
            
            )
            
            retain_cost = retain_kl + max(retain_max_drop, 0.0)
            
            specificity = forget_drop_positive_mean / max(retain_cost, 1e-8)
            
            passes_forget_drop = forget_drop_mean >= min_forget_drop
            
            passes_retain_kl = retain_kl <= max_retain_kl
            
            passes_retain_prob_drop = retain_max_drop <= max_retain_prob_drop
            
            passes_specificity = specificity >= min_specificity
            
            eligible = all(
                
                (
                    
                    passes_forget_drop,
                    
                    passes_retain_kl,
                    
                    passes_retain_prob_drop,
                    
                    passes_specificity,
                
                )
            
            )
            
            rows.append(
                
                {
                    
                    "layer": candidate.layer,
                    
                    "site": candidate.site,
                    
                    "trace_score": candidate.score,
                    
                    "gate_strength": gate_strength,
                    
                    "forget_drop_mean": forget_drop_mean,
                    
                    "forget_positive_drop_mean": forget_drop_positive_mean,
                    
                    "forget_max_prob_increase": forget_max_increase,
                    
                    "retain_kl_mean": retain_kl,
                    
                    "retain_max_prob_drop": retain_max_drop,
                    
                    "retain_cost": retain_cost,
                    
                    "path_specificity": specificity,
                    
                    "passes_forget_drop": passes_forget_drop,
                    
                    "passes_retain_kl": passes_retain_kl,
                    
                    "passes_retain_prob_drop": passes_retain_prob_drop,
                    
                    "passes_specificity": passes_specificity,
                    
                    "gate_validated": eligible,
                
                }
            
            )
            
            if eligible:
                
                break
    
    return pd.DataFrame(rows).sort_values(
        
        ["gate_validated", "path_specificity", "forget_drop_mean"],
        
        ascending=[False, False, False],
    
    )



def select_gate_validated_candidates(
    
    candidates: list[CutCandidate],
    
    effects: pd.DataFrame,
    
    min_validated_candidates: int,

) -> list[CutCandidate]:
    
    """Keep candidates whose measured gate effect passes validation."""
    
    validated = effects[effects["gate_validated"]].copy()
    
    if len(validated) < min_validated_candidates:
        
        gate_columns = (
            
            "passes_forget_drop",
            
            "passes_retain_kl",
            
            "passes_retain_prob_drop",
            
            "passes_specificity",
        
        )
        
        pass_counts = ", ".join(
            
            f"{column}={int(effects[column].sum())}/{len(effects)}"
            
            for column in gate_columns
            
            if column in effects
        
        )
        
        closest = effects.copy()
        
        available_gate_columns = [column for column in gate_columns if column in closest]
        
        if available_gate_columns:
            
            closest["passed_gate_count"] = closest[available_gate_columns].sum(axis=1)
            
            closest = closest.sort_values(
                
                ["passed_gate_count", "path_specificity", "forget_drop_mean"],
                
                ascending=[False, False, False],
            
            )
        
        preview_columns = [
            
            column
            
            for column in (
                
                "layer",
                
                "site",
                
                "gate_strength",
                
                "forget_drop_mean",
                
                "retain_kl_mean",
                
                "retain_max_prob_drop",
                
                "path_specificity",
                
                "passed_gate_count",
            
            )
            
            if column in closest
        
        ]
        
        preview = closest.head(3)[preview_columns].to_dict(orient="records")
        
        raise RuntimeError(
            
            "Path-gate validation found too few retain-safe candidates: "
            
            f"{len(validated)} < {min_validated_candidates}. "
            
            f"Per-condition pass counts: {pass_counts}. "
            
            f"Closest candidates: {preview}. "
            
            "Inspect 03_path_gate_effects.csv and tune only the failing threshold(s); "
            
            "do not bypass the gate."
        
        )
    
    keys = {(int(row.layer), str(row.site)) for row in validated.itertuples(index=False)}
    
    return [candidate for candidate in candidates if (candidate.layer, candidate.site) in keys]






def load_candidates(
    
    trace_path: str | Path,
    
    top_k: int,
    
    site: str | None = None,
    
    retain_trace_penalty: float = 1.0,
    
    min_forget_restore: float = 0.0,
    
    min_forget_positive_fraction: float = 0.0,
    
    robust_quantile: float = 0.5,
    
    consistency_weight: float = 0.0,
    min_forget_base_prob: float = 0.0,
    restore_effect_epsilon: float = 0.0,

) -> list[CutCandidate]:
    """Load and rank candidates from tracing results."""
    
    df = pd.read_csv(trace_path)
    
    if site in {"", "all", "*"}:
        
        site = None
    
    if site:
        
        df = df[df["site"] == site]
    
    quantile = min(max(float(robust_quantile), 0.0), 1.0)
    
    keys = ["layer", "site"]
    
    forget_df = df[df["type"] == "forget"].copy()
    if "clean_prob" not in forget_df.columns:
        raise ValueError("causal tracing CSV is missing required column: clean_prob")
    forget_df = forget_df[
        forget_df["clean_prob"] >= float(min_forget_base_prob)
    ].copy()
    if forget_df.empty:
        raise RuntimeError(
            "No forget tracing rows meet dpcu.min_forget_base_prob. "
            "The selected fact/variants are not active in the base model."
        )
    positive_epsilon = max(float(restore_effect_epsilon), 0.0)
    
    retain_df = df[df["type"] == "retain"].copy()
    
    forget = forget_df.groupby(keys)["restore_effect"].agg(
        
        forget_mean="mean",
        
        forget_positive_fraction=lambda values: float(
            (values > positive_epsilon).mean()
        ),
    
    )
    
    forget["forget_robust"] = forget_df.groupby(keys)["restore_effect"].quantile(quantile)
    
    if "corruption" in forget_df and forget_df["corruption"].nunique() > 1:
        
        by_corruption = forget_df.groupby(keys + ["corruption"])["restore_effect"].mean()
        
        forget["corruption_positive_fraction"] = by_corruption.groupby(level=keys).apply(
            
            lambda values: float((values > positive_epsilon).mean())
        
        )
    
    else:
        
        forget["corruption_positive_fraction"] = forget["forget_positive_fraction"]
    
    retain = retain_df.groupby(keys)["restore_effect"].agg(retain_mean="mean")
    
    table = forget.join(retain, how="left").fillna({"retain_mean": 0.0}).reset_index()
    
    table["forget_signal"] = table["forget_robust"].clip(lower=0.0)
    
    table["retain_risk"] = table["retain_mean"].clip(lower=0.0)
    
    table["consistency"] = (
        
        table["forget_positive_fraction"] * table["corruption_positive_fraction"]
    
    )
    
    table["score"] = (
        
        table["forget_signal"]
        
        - retain_trace_penalty * table["retain_risk"]
        
        + consistency_weight * table["forget_signal"] * table["consistency"]
    
    )
    
    grouped = table[
        
        (table["forget_signal"] >= min_forget_restore)
        
        & (table["forget_positive_fraction"] >= min_forget_positive_fraction)
        
        & (table["score"] > 0.0)
    
    ].sort_values(
        
        ["score", "consistency", "forget_mean"],
        
        ascending=[False, False, False],
    
    ).head(top_k)
    
    if grouped.empty:
        
        raise RuntimeError(
            
            "Robust causal candidate filtering removed every path. "
            
            "Lower dpcu.min_forget_restore or dpcu.min_forget_positive_fraction."
        
        )
    
    return [
        
        CutCandidate(
            
            layer=int(row.layer),
            
            site=str(row.site),
            
            score=float(row.score),
            
            forget_restore=float(row.forget_mean),
            
            retain_restore=float(row.retain_mean),
        
        )
        
        for row in grouped.itertuples(index=False)
    
    ]



def hard_tail_mean(values: list[float], tail_fraction: float) -> float:
    """Average the largest values in a metric tail."""
    
    if not values:
        
        return 0.0
    
    fraction = min(max(tail_fraction, 0.0), 1.0)
    
    count = max(1, int(round(len(values) * fraction)))
    
    return sum(sorted(values, reverse=True)[:count]) / count



def active_forget_ids(base_forget: dict[str, float], min_base_prob: float) -> set[str]:
    """Return forget identifiers whose detector gate is active."""
    
    active = {fact_id for fact_id, prob in base_forget.items() if prob >= min_base_prob}
    
    return active or set(base_forget)



def forget_cut_stats(
    
    base_forget: dict[str, float],
    
    cut_probs: dict[str, float],
    
    min_base_prob: float,
    
    tail_fraction: float,

) -> dict[str, float]:
    """Summarize target suppression across forgotten facts."""
    
    active_ids = active_forget_ids(base_forget, min_base_prob)
    
    active_cut = [cut_probs[k] for k in active_ids if k in cut_probs]
    
    active_drop = [base_forget[k] - cut_probs[k] for k in active_ids if k in cut_probs]
    
    active_increase = [cut_probs[k] - base_forget[k] for k in active_ids if k in cut_probs]
    
    return {
        
        "mean_cut_prob": sum(active_cut) / max(len(active_cut), 1),
        
        "worst_cut_prob": max(active_cut) if active_cut else 0.0,
        
        "tail_cut_prob": hard_tail_mean(active_cut, tail_fraction),
        
        "mean_drop": sum(active_drop) / max(len(active_drop), 1),
        
        "max_prob_increase": max(max(active_increase), 0.0) if active_increase else 0.0,
        
        "active_count": float(len(active_ids)),
    
    }



def expand_fact_variants(facts: list[Fact], variants: list[str]) -> list[Fact]:
    """Generate evaluation variants for one fact."""
    
    expanded: list[Fact] = []
    
    use_factual = "factual" in variants
    
    use_paraphrase = "paraphrase" in variants
    
    use_alias = "alias" in variants
    
    for fact in facts:
        
        if use_factual:
            
            expanded.append(fact)
        
        if use_paraphrase:
            
            for idx, template in enumerate(RELATION_PATTERNS.get(fact.relation, [])):
                
                expanded.append(
                    
                    Fact(
                        
                        id=f"{fact.id}:paraphrase:{idx}",
                        
                        prompt=template.format(subject=fact.subject),
                        
                        answer=fact.answer,
                        
                        type=fact.type,
                        
                        subject=fact.subject,
                        
                        relation=fact.relation,
                        
                        wrong_answer=fact.wrong_answer,
                    
                    )
                
                )
        
        if use_alias and fact.subject:
            
            alias = fact.subject.replace("The ", "").replace("the ", "")
            
            if alias != fact.subject:
                
                expanded.append(
                    
                    Fact(
                        
                        id=f"{fact.id}:alias",
                        
                        prompt=fact.prompt.replace(fact.subject, alias),
                        
                        answer=fact.answer,
                        
                        type=fact.type,
                        
                        subject=fact.subject,
                        
                        relation=fact.relation,
                        
                        unknown_prompt=fact.unknown_prompt.replace(fact.subject, alias),
                        
                        wrong_answer=fact.wrong_answer,
                    
                    )
                
                )
    
    return expanded



@torch.no_grad()

def rerank_candidates_by_single_cut(
    
    model,
    
    forget_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    candidates: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    top_k: int,
    
    retain_kl_budget: float,
    
    retain_kl_weight: float,
    
    cut_size_penalty: float,
    
    max_retain_prob_drop: float,
    
    cut_strength: float,
    
    forget_worst_weight: float = 0.0,
    
    forget_tail_weight: float = 0.0,
    
    forget_tail_fraction: float = 0.4,
    
    min_forget_base_prob: float = 0.0,
    
    max_forget_prob_increase: float = 1.0,
    
    forget_increase_weight: float = 50.0,
    retain_prob_drop_weight: float = 0.0,

) -> list[CutCandidate]:
    """Rerank candidates by isolated forget and retain effects."""
    
    base_forget = {fact.id: target_first_prob(model, fact.prompt, fact.answer) for fact in forget_facts}
    
    ranked = []
    
    for cand in tqdm(candidates, desc=f"rerank strength={cut_strength:g}", leave=False):
        
        trial = [cand]
        
        cut_probs = {
            
            fact.id: target_prob_with_cut(model, fact, trial, neutral_bank, cut_strength=cut_strength)
            
            for fact in forget_facts
        
        }
        
        forget_stats = forget_cut_stats(base_forget, cut_probs, min_forget_base_prob, forget_tail_fraction)
        
        mean_cut_prob = forget_stats["mean_cut_prob"]
        
        worst_cut_prob = forget_stats["worst_cut_prob"]
        
        tail_cut_prob = forget_stats["tail_cut_prob"]
        
        mean_drop = forget_stats["mean_drop"]
        
        max_prob_increase = forget_stats["max_prob_increase"]
        
        retain_kl, max_retain_drop = retain_metrics_with_cut(
            
            model, retain_facts, trial, neutral_bank, cut_strength=cut_strength
        
        )
        
        penalty = 1000.0 * max(retain_kl - retain_kl_budget, 0.0)
        
        penalty += 1000.0 * max(max_retain_drop - max_retain_prob_drop, 0.0)
        
        penalty += forget_increase_weight * max(max_prob_increase - max_forget_prob_increase, 0.0)
        
        objective = (
            
            mean_cut_prob
            
            + forget_worst_weight * worst_cut_prob
            
            + forget_tail_weight * tail_cut_prob
            
            + retain_kl_weight * retain_kl
            + retain_prob_drop_weight * max(max_retain_drop, 0.0)
            
            + penalty
            
            + cut_size_penalty
        
        )
        
        feasible = (
            
            retain_kl <= retain_kl_budget
            
            and max_retain_drop <= max_retain_prob_drop
            
            and max_prob_increase <= max_forget_prob_increase
            
            and mean_drop > 0.0
        
        )
        
        ranked.append(
            
            replace(
                
                cand,
                
                single_objective=float(objective),
                
                single_forget_drop=float(mean_drop),
                
                single_retain_kl=float(retain_kl),
                
                single_max_retain_prob_drop=float(max_retain_drop),
                
                single_feasible=bool(feasible),
            
            )
        
        )
    
    ranked.sort(
        
        key=lambda cand: (
            
            not bool(cand.single_feasible),
            
            float(cand.single_objective if cand.single_objective is not None else float("inf")),
        
        )
    
    )
    
    return ranked[:top_k]






def greedy_cut_search(
    
    model,
    
    forget_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    candidates: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    max_cut_size: int,
    
    retain_kl_budget: float,
    
    target_prob_stop: float,
    
    retain_kl_weight: float,
    
    cut_size_penalty: float,
    
    max_retain_prob_drop: float,
    
    cut_strength: float,
    
    forget_worst_weight: float = 0.0,
    
    forget_tail_weight: float = 0.0,
    
    forget_tail_fraction: float = 0.4,
    
    min_forget_base_prob: float = 0.0,
    
    max_forget_prob_increase: float = 1.0,
    
    forget_increase_weight: float = 50.0,
    retain_prob_drop_weight: float = 0.0,
    pair_lookahead: bool = False,

) -> tuple[list[CutCandidate], list[dict]]:
    """Build a feasible cut set under retain constraints."""
    
    selected: list[CutCandidate] = []
    
    remaining = list(candidates)
    
    log: list[dict] = []
    
    base_forget = {fact.id: target_first_prob(model, fact.prompt, fact.answer) for fact in forget_facts}
    
    initial_stats = forget_cut_stats(base_forget, base_forget, min_forget_base_prob, forget_tail_fraction)
    
    current_mean_cut_prob = initial_stats["mean_cut_prob"]
    
    current_worst_cut_prob = initial_stats["worst_cut_prob"]
    
    current_tail_cut_prob = initial_stats["tail_cut_prob"]
    
    current_objective = (
        
        current_mean_cut_prob
        
        + forget_worst_weight * current_worst_cut_prob
        
        + forget_tail_weight * current_tail_cut_prob
    
    )
    
    min_improvement = 1e-6

    
    for step in range(max_cut_size):
        if len(selected) >= max_cut_size:
            break
        
        best = None
        
        best_rejected = None
        
        for cand in tqdm(
            
            remaining,
            
            desc=f"greedy step {step + 1}/{max_cut_size} strength={cut_strength:g}",
            
            leave=False,
        
        ):
            
            trial = selected + [cand]
            
            cut_probs = {
                
                fact.id: target_prob_with_cut(model, fact, trial, neutral_bank, cut_strength=cut_strength)
                
                for fact in forget_facts
            
            }
            
            forget_stats = forget_cut_stats(base_forget, cut_probs, min_forget_base_prob, forget_tail_fraction)
            
            mean_cut_prob = forget_stats["mean_cut_prob"]
            
            worst_cut_prob = forget_stats["worst_cut_prob"]
            
            tail_cut_prob = forget_stats["tail_cut_prob"]
            
            mean_drop = forget_stats["mean_drop"]
            
            max_prob_increase = forget_stats["max_prob_increase"]
            
            retain_kl, max_retain_drop = retain_metrics_with_cut(
                
                model, retain_facts, trial, neutral_bank, cut_strength=cut_strength
            
            )
            
            penalty = 1000.0 * max(retain_kl - retain_kl_budget, 0.0)
            
            penalty += 1000.0 * max(max_retain_drop - max_retain_prob_drop, 0.0)
            
            penalty += forget_increase_weight * max(max_prob_increase - max_forget_prob_increase, 0.0)
            
            objective = (
                
                mean_cut_prob
                
                + forget_worst_weight * worst_cut_prob
                
                + forget_tail_weight * tail_cut_prob
                
                + retain_kl_weight * retain_kl
                + retain_prob_drop_weight * max(max_retain_drop, 0.0)
                
                + penalty
                
                + cut_size_penalty * len(trial)
            
            )
            
            record = {
                
                "step": step,
                
                "candidate": asdict(cand),
                
                "mean_cut_prob": mean_cut_prob,
                
                "worst_cut_prob": worst_cut_prob,
                
                "tail_cut_prob": tail_cut_prob,
                
                "mean_forget_drop": mean_drop,
                
                "max_forget_prob_increase": max_prob_increase,
                
                "active_forget_count": forget_stats["active_count"],
                
                "retain_kl": retain_kl,
                
                "max_retain_prob_drop": max_retain_drop,
                
                "within_retain_budget": retain_kl <= retain_kl_budget,
                
                "within_retain_prob_drop_budget": max_retain_drop <= max_retain_prob_drop,
                
                "within_forget_increase_budget": max_prob_increase <= max_forget_prob_increase,
                
                "objective": objective,
                
                "cut_probs": cut_probs,
            
            }
            
            improves_forget = mean_cut_prob < current_mean_cut_prob - min_improvement
            
            improves_worst = worst_cut_prob < current_worst_cut_prob - min_improvement
            
            improves_tail = tail_cut_prob < current_tail_cut_prob - min_improvement
            
            improves_objective = objective < current_objective - min_improvement
            
            is_acceptable = (
                
                record["within_retain_budget"]
                
                and record["within_retain_prob_drop_budget"]
                
                and record["within_forget_increase_budget"]
                
                and (improves_forget or improves_worst or improves_tail)
                
                and improves_objective
            
            )
            
            if is_acceptable and (best is None or objective < best[0]):
                
                best = (objective, cand, record)
            
            if best_rejected is None or objective < best_rejected[0]:
                
                best_rejected = (objective, cand, record)
        
        
        
        if (
            best is None
            and pair_lookahead
            and len(selected) + 2 <= max_cut_size
            and len(remaining) >= 2
        ):
            best_pair = None
            for left_index, left in enumerate(remaining[:-1]):
                for right in remaining[left_index + 1:]:
                    trial = selected + [left, right]
                    cut_probs = {
                        fact.id: target_prob_with_cut(
                            model, fact, trial, neutral_bank,
                            cut_strength=cut_strength,
                        )
                        for fact in forget_facts
                    }
                    stats = forget_cut_stats(
                        base_forget, cut_probs, min_forget_base_prob,
                        forget_tail_fraction,
                    )
                    retain_kl, max_retain_drop = retain_metrics_with_cut(
                        model, retain_facts, trial, neutral_bank,
                        cut_strength=cut_strength,
                    )
                    max_increase = stats["max_prob_increase"]
                    penalty = 1000.0 * max(
                        retain_kl - retain_kl_budget, 0.0
                    )
                    penalty += 1000.0 * max(
                        max_retain_drop - max_retain_prob_drop, 0.0
                    )
                    penalty += forget_increase_weight * max(
                        max_increase - max_forget_prob_increase, 0.0
                    )
                    objective = (
                        stats["mean_cut_prob"]
                        + forget_worst_weight * stats["worst_cut_prob"]
                        + forget_tail_weight * stats["tail_cut_prob"]
                        + retain_kl_weight * retain_kl
                        + retain_prob_drop_weight
                        * max(max_retain_drop, 0.0)
                        + penalty
                        + cut_size_penalty * len(trial)
                    )
                    record = {
                        "step": step,
                        "selection_mode": "pair_lookahead",
                        "candidates": [asdict(left), asdict(right)],
                        "mean_cut_prob": stats["mean_cut_prob"],
                        "worst_cut_prob": stats["worst_cut_prob"],
                        "tail_cut_prob": stats["tail_cut_prob"],
                        "mean_forget_drop": stats["mean_drop"],
                        "max_forget_prob_increase": max_increase,
                        "active_forget_count": stats["active_count"],
                        "retain_kl": retain_kl,
                        "max_retain_prob_drop": max_retain_drop,
                        "within_retain_budget":
                            retain_kl <= retain_kl_budget,
                        "within_retain_prob_drop_budget":
                            max_retain_drop <= max_retain_prob_drop,
                        "within_forget_increase_budget":
                            max_increase <= max_forget_prob_increase,
                        "objective": objective,
                        "cut_probs": cut_probs,
                    }
                    improves = (
                        stats["mean_cut_prob"]
                        < current_mean_cut_prob - min_improvement
                        or stats["worst_cut_prob"]
                        < current_worst_cut_prob - min_improvement
                        or stats["tail_cut_prob"]
                        < current_tail_cut_prob - min_improvement
                    )
                    acceptable = (
                        record["within_retain_budget"]
                        and record["within_retain_prob_drop_budget"]
                        and record["within_forget_increase_budget"]
                        and improves
                        and objective < current_objective - min_improvement
                    )
                    if acceptable and (
                        best_pair is None or objective < best_pair[0]
                    ):
                        best_pair = (
                            objective, (left, right), record
                        )
            if best_pair is not None:
                best = best_pair

        if best is None:
            
            if best_rejected is not None:
                
                _, _, record = best_rejected
                
                record["selected"] = False
                
                if not record["within_retain_budget"]:
                    
                    record["stop_reason"] = "retain_kl_budget_exceeded"
                
                elif not record["within_retain_prob_drop_budget"]:
                    
                    record["stop_reason"] = "retain_prob_drop_budget_exceeded"
                
                elif not record["within_forget_increase_budget"]:
                    
                    record["stop_reason"] = "forget_prob_increase_budget_exceeded"
                
                elif record["mean_cut_prob"] >= current_mean_cut_prob - min_improvement:
                    
                    record["stop_reason"] = "no_forget_improvement"
                
                else:
                    
                    record["stop_reason"] = "no_objective_improvement"
                
                log.append(record)
            
            break
        
        objective, chosen, record = best
        chosen_candidates = (
            list(chosen) if isinstance(chosen, tuple) else [chosen]
        )
        
        record["selected"] = True
        
        selected.extend(chosen_candidates)
        
        current_mean_cut_prob = float(record["mean_cut_prob"])
        
        current_worst_cut_prob = float(record["worst_cut_prob"])
        
        current_tail_cut_prob = float(record["tail_cut_prob"])
        
        current_objective = float(objective)
        
        chosen_keys = {
            (candidate.layer, candidate.site)
            for candidate in chosen_candidates
        }
        remaining = [
            candidate for candidate in remaining
            if (candidate.layer, candidate.site) not in chosen_keys
        ]
        
        log.append(record)
        
        print(
            
            "selected "
            
            f"step={step + 1} paths="
            f"{[(c.layer, c.site) for c in chosen_candidates]} "
            
            f"mean_cut_prob={record['mean_cut_prob']:.6f} "
            
            f"worst_cut_prob={record['worst_cut_prob']:.6f} "
            
            f"tail_cut_prob={record['tail_cut_prob']:.6f} "
            
            f"forget_drop={record['mean_forget_drop']:.6f} "
            
            f"forget_max_increase={record['max_forget_prob_increase']:.6f} "
            
            f"retain_kl={record['retain_kl']:.6f} "
            
            f"retain_max_drop={record['max_retain_prob_drop']:.6f}",
            
            flush=True,
        
        )
        
        if record["mean_cut_prob"] <= target_prob_stop:
            
            record["stop_reason"] = "target_prob_stop_reached"
            
            break
    
    return selected, log



@torch.no_grad()




def evaluate_cut(
    
    model,
    
    facts: list[Fact],
    
    cut_set: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    cut_strength: float = 1.0,

) -> pd.DataFrame:
    """Evaluate suppression, retain damage, and cut feasibility."""
    
    rows = []
    
    for fact in facts:
        
        base = target_first_prob(model, fact.prompt, fact.answer)
        
        cut = target_prob_with_cut(model, fact, cut_set, neutral_bank, cut_strength=cut_strength)
        
        row = {
            
            "id": fact.id,
            
            "type": fact.type,
            
            "prompt": fact.prompt,
            
            "answer": fact.answer,
            
            "base_first_token_prob": base,
            
            "cut_first_token_prob": cut,
            
            "prob_drop": base - cut,
        
        }
        
        if fact.type == "retain":
            
            base_dist = next_token_distribution(model, fact.prompt)
            
            cut_logits = run_cut_logits(model, fact.prompt, cut_set, neutral_bank, cut_strength=cut_strength)
            
            cut_dist = F.softmax(cut_logits[:, -1, :].float(), dim=-1)
            
            row["retain_kl"] = kl_div(base_dist, cut_dist)
        
        rows.append(row)
    
    return pd.DataFrame(rows)



def fact_to_factual_unit(fact: Fact) -> CausalUnit:
    
    """Convert a fact into the detector?s canonical factual unit."""
    
    variant = "factual"
    
    if ":paraphrase:" in fact.id:
        
        variant = "paraphrase"
    
    elif fact.id.endswith(":alias"):
        
        variant = "alias"
    
    return CausalUnit(
        
        id=f"{fact.id}:runtime",
        
        fact_id=fact.id,
        
        split=fact.type,
        
        variant=variant,
        
        prompt=fact.prompt,
        
        answer=fact.answer,
        
        subject=fact.subject,
        
        relation=fact.relation,
        
        target_label=int(fact.type == "forget"),
    
    )



@torch.no_grad()

def evaluate_breaker_cut(
    
    model,
    
    facts: list[Fact],
    
    forget_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    detector,
    
    detector_threshold: float,
    
    cut_set: list[CutCandidate],
    
    neutral_bank: dict[PatchSpec, torch.Tensor],
    
    cut_strength: float = 1.0,

) -> pd.DataFrame:
    """Evaluate detector-gated conditional cuts."""
    
    rows = []
    
    for fact in facts:
        
        base = target_first_prob(model, fact.prompt, fact.answer)
        
        unit = fact_to_factual_unit(fact)
        
        breaker_prob = detector_probability(detector, unit, forget_facts, retain_facts)
        
        breaker_triggered = breaker_prob > detector_threshold
        
        cut = (
            
            target_prob_with_cut(model, fact, cut_set, neutral_bank, cut_strength=cut_strength)
            
            if breaker_triggered
            
            else base
        
        )
        
        row = {
            
            "id": fact.id,
            
            "type": fact.type,
            
            "prompt": fact.prompt,
            
            "answer": fact.answer,
            
            "breaker_prob": breaker_prob,
            
            "breaker_threshold": detector_threshold,
            
            "breaker_triggered": breaker_triggered,
            
            "base_first_token_prob": base,
            
            "cut_first_token_prob": cut,
            
            "prob_drop": base - cut,
        
        }
        
        if fact.type == "retain":
            
            base_dist = next_token_distribution(model, fact.prompt)
            
            if breaker_triggered:
                
                cut_logits = run_cut_logits(model, fact.prompt, cut_set, neutral_bank, cut_strength=cut_strength)
                
                cut_dist = F.softmax(cut_logits[:, -1, :].float(), dim=-1)
                
                row["retain_kl"] = kl_div(base_dist, cut_dist)
            
            else:
                
                row["retain_kl"] = 0.0
        
        rows.append(row)
    
    return pd.DataFrame(rows)



def summarize_eval(df: pd.DataFrame) -> dict:
    """Aggregate per-example records into summary metrics."""
    
    forget = df[df["type"] == "forget"]
    
    retain = df[df["type"] == "retain"]
    
    return {
        
        "forget_base_mean": float(forget["base_first_token_prob"].mean()) if not forget.empty else 0.0,
        
        "forget_cut_mean": float(forget["cut_first_token_prob"].mean()) if not forget.empty else 0.0,
        
        "forget_drop_mean": float(forget["prob_drop"].mean()) if not forget.empty else 0.0,
        
        "retain_base_mean": float(retain["base_first_token_prob"].mean()) if not retain.empty else 0.0,
        
        "retain_cut_mean": float(retain["cut_first_token_prob"].mean()) if not retain.empty else 0.0,
        
        "retain_drop_mean": float(retain["prob_drop"].mean()) if not retain.empty else 0.0,
        
        "retain_max_drop": float(retain["prob_drop"].max()) if not retain.empty else 0.0,
        
        "retain_kl_mean": float(retain["retain_kl"].mean()) if "retain_kl" in retain and not retain.empty else 0.0,
    
    }



def main() -> None:
    """Parse arguments and run the configured workflow."""
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default="code/knowledge_group/base_config.yaml")
    
    parser.add_argument("--trace", default="outputs/50-targets/causal_tracing.csv")
    
    parser.add_argument("--out-dir", default="outputs/50-targets/dpcu_lite")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    print("[dpcu] loading facts/model", flush=True)
    
    facts = load_facts(cfg["data"]["facts_path"])
    
    forget_facts, retain_facts = split_facts(facts)
    
    search_variants = list(cfg["dpcu"].get("search_variants", ["factual"]))
    
    search_forget_facts = expand_fact_variants(forget_facts, search_variants)
    
    search_retain_facts = expand_fact_variants(retain_facts, search_variants)
    
    model = load_from_config(cfg)

    
    dcfg = cfg["dpcu"]
    
    trace_top_k = int(dcfg.get("trace_top_k_paths", dcfg["top_k_paths"]))
    
    print(f"[dpcu] loading top {trace_top_k} candidates from {args.trace}", flush=True)
    
    candidates = load_candidates(
        
        args.trace,
        
        top_k=trace_top_k,
        
        site=dcfg.get("site"),
        
        retain_trace_penalty=float(dcfg.get("retain_trace_penalty", 1.0)),
        min_forget_base_prob=float(dcfg.get("min_forget_base_prob", 0.0)),
        restore_effect_epsilon=float(dcfg.get("restore_effect_epsilon", 0.0)),
    
    )
    
    print(f"[dpcu] loaded {len(candidates)} candidates; building neutral bank", flush=True)
    
    neutral_bank = build_neutral_bank(
        model,
        facts,
        candidates,
        strategy=dcfg.get("neutral_strategy", "unknown_mean"),
    )
    
    print(f"[dpcu] neutral bank ready with {len(neutral_bank)} patch specs", flush=True)
    
    trace_candidates = list(candidates)
    
    out_dir = ensure_dir(args.out_dir)
    
    if "cut_strength_sweep" in dcfg:
        
        cut_strengths = [float(x) for x in dcfg["cut_strength_sweep"]]
    
    else:
        
        cut_strengths = [float(dcfg.get("cut_strength", 1.0))]

    
    runs = []
    
    print(
        
        f"[dpcu] search variants={search_variants}; "
        
        f"forget units={len(search_forget_facts)} retain units={len(search_retain_facts)}",
        
        flush=True,
    
    )
    
    print(f"[dpcu] strength sweep: {cut_strengths}", flush=True)
    
    for idx, cut_strength in enumerate(cut_strengths, start=1):
        
        print(f"[dpcu] strength {idx}/{len(cut_strengths)} = {cut_strength:g}: reranking candidates", flush=True)
        
        ranked_candidates = rerank_candidates_by_single_cut(
            
            model=model,
            
            forget_facts=search_forget_facts,
            
            retain_facts=search_retain_facts,
            
            candidates=list(candidates),
            
            neutral_bank=neutral_bank,
            
            top_k=int(dcfg["top_k_paths"]),
            
            retain_kl_budget=float(dcfg["retain_kl_budget"]),
            
            retain_kl_weight=float(dcfg.get("retain_kl_weight", 0.25)),
            
            cut_size_penalty=float(dcfg.get("cut_size_penalty", 0.01)),
            
            max_retain_prob_drop=float(dcfg.get("max_retain_prob_drop", 0.10)),
            
            cut_strength=cut_strength,
            
            forget_worst_weight=float(dcfg.get("forget_worst_weight", 0.0)),
            
            forget_tail_weight=float(dcfg.get("forget_tail_weight", 0.0)),
            
            forget_tail_fraction=float(dcfg.get("forget_tail_fraction", 0.4)),
            
            min_forget_base_prob=float(dcfg.get("min_forget_base_prob", 0.0)),
            
            max_forget_prob_increase=float(dcfg.get("max_forget_prob_increase", 1.0)),
            
            forget_increase_weight=float(dcfg.get("forget_increase_weight", 50.0)),
        
        )
        
        print(f"[dpcu] strength {cut_strength:g}: greedy cut search", flush=True)
        
        cut_set, search_log = greedy_cut_search(
            
            model=model,
            
            forget_facts=search_forget_facts,
            
            retain_facts=search_retain_facts,
            
            candidates=ranked_candidates,
            
            neutral_bank=neutral_bank,
            
            max_cut_size=int(dcfg["max_cut_size"]),
            
            retain_kl_budget=float(dcfg["retain_kl_budget"]),
            
            target_prob_stop=float(dcfg["target_prob_stop"]),
            
            retain_kl_weight=float(dcfg.get("retain_kl_weight", 0.25)),
            
            cut_size_penalty=float(dcfg.get("cut_size_penalty", 0.01)),
            
            max_retain_prob_drop=float(dcfg.get("max_retain_prob_drop", 0.10)),
            
            cut_strength=cut_strength,
            
            forget_worst_weight=float(dcfg.get("forget_worst_weight", 0.0)),
            
            forget_tail_weight=float(dcfg.get("forget_tail_weight", 0.0)),
            
            forget_tail_fraction=float(dcfg.get("forget_tail_fraction", 0.4)),
            
            min_forget_base_prob=float(dcfg.get("min_forget_base_prob", 0.0)),
            
            max_forget_prob_increase=float(dcfg.get("max_forget_prob_increase", 1.0)),
            
            forget_increase_weight=float(dcfg.get("forget_increase_weight", 50.0)),
        
        )
        
        print(f"[dpcu] strength {cut_strength:g}: evaluating {len(cut_set)} selected cuts", flush=True)
        
        df = evaluate_cut(model, facts, cut_set, neutral_bank, cut_strength=cut_strength)
        
        variant_df = evaluate_cut(
            
            model,
            
            search_forget_facts + search_retain_facts,
            
            cut_set,
            
            neutral_bank,
            
            cut_strength=cut_strength,
        
        )
        
        summary = summarize_eval(df)
        
        variant_summary = summarize_eval(variant_df)
        
        forget_df = df[df["type"] == "forget"]
        
        variant_forget_df = variant_df[variant_df["type"] == "forget"]
        
        summary["forget_worst_cut_prob"] = (
            
            float(forget_df["cut_first_token_prob"].max()) if not forget_df.empty else 0.0
        
        )
        
        summary["forget_tail_cut_prob"] = hard_tail_mean(
            
            [float(x) for x in forget_df["cut_first_token_prob"].tolist()],
            
            float(dcfg.get("forget_tail_fraction", 0.4)),
        
        )
        
        summary["variant_forget_cut_mean"] = variant_summary["forget_cut_mean"]
        
        summary["variant_forget_drop_mean"] = variant_summary["forget_drop_mean"]
        
        summary["variant_forget_worst_cut_prob"] = (
            
            float(variant_forget_df["cut_first_token_prob"].max()) if not variant_forget_df.empty else 0.0
        
        )
        
        summary["variant_retain_kl_mean"] = variant_summary["retain_kl_mean"]
        
        summary["variant_retain_max_drop"] = variant_summary["retain_max_drop"]
        
        feasible = (
            
            variant_summary["retain_kl_mean"] <= float(dcfg["retain_kl_budget"])
            
            and variant_summary["retain_max_drop"] <= float(dcfg.get("max_retain_prob_drop", 0.10))
        
        )
        
        runs.append(
            
            {
                
                "cut_strength": cut_strength,
                
                "candidates": ranked_candidates,
                
                "cut_set": cut_set,
                
                "search_log": search_log,
                
                "eval": df,
                
                "variant_eval": variant_df,
                
                "summary": summary,
                
                "feasible": feasible,
            
            }
        
        )
        
        print(
            
            f"[dpcu] strength {cut_strength:g} done: "
            
            f"feasible={feasible} "
            
            f"forget_cut_mean={summary['forget_cut_mean']:.6f} "
            
            f"forget_drop_mean={summary['forget_drop_mean']:.6f} "
            
            f"forget_worst_cut_prob={summary['forget_worst_cut_prob']:.6f} "
            
            f"forget_tail_cut_prob={summary['forget_tail_cut_prob']:.6f} "
            
            f"variant_forget_cut_mean={summary['variant_forget_cut_mean']:.6f} "
            
            f"variant_forget_worst_cut_prob={summary['variant_forget_worst_cut_prob']:.6f} "
            
            f"retain_kl_mean={summary['retain_kl_mean']:.6f} "
            
            f"retain_max_drop={summary['retain_max_drop']:.6f}",
            
            flush=True,
        
        )

    
    runs.sort(
        
        key=lambda run: (
            
            not run["feasible"],
            
            run["summary"].get("forget_cut_mean", 0.0)
            
            + float(dcfg.get("forget_worst_weight", 0.0)) * run["summary"].get("forget_worst_cut_prob", 0.0)
            
            + float(dcfg.get("forget_tail_weight", 0.0)) * run["summary"].get("forget_tail_cut_prob", 0.0),
            
            run["summary"].get("variant_forget_cut_mean", 0.0),
            
            run["summary"].get("variant_forget_worst_cut_prob", 0.0),
        
        )
    
    )
    
    best_run = runs[0]
    
    static_df = best_run["eval"]
    
    variant_static_df = best_run["variant_eval"]
    
    print(
        
        f"[dpcu] best strength={best_run['cut_strength']:g}; saving static cut outputs",
        
        flush=True,
    
    )
    
    static_df.to_csv(out_dir / "static_path_cut_eval.csv", index=False, encoding="utf-8-sig")
    
    variant_static_df.to_csv(out_dir / "variant_path_cut_eval.csv", index=False, encoding="utf-8-sig")
    
    bcfg = cfg.get("breaker", {})
    
    print("[dpcu] training target detector", flush=True)
    
    detector_units = build_causal_units(forget_facts, retain_facts)
    
    detector, detector_report = train_target_detector(
        
        detector_units,
        
        forget_facts,
        
        retain_facts,
        
        lambda_fp=float(bcfg.get("lambda_fp", 2.0)),
        
        lambda_fn=float(bcfg.get("lambda_fn", 2.0)),
        
        lr=float(bcfg.get("learning_rate", 0.05)),
        
        epochs=int(bcfg.get("epochs", 300)),
    
    )
    
    detector_threshold = float(bcfg.get("threshold", 0.5))
    
    save_detector(detector, out_dir / "target_detector.pt", detector_report, detector_units)
    
    print("[dpcu] evaluating breaker-gated cut", flush=True)
    
    df = evaluate_breaker_cut(
        
        model=model,
        
        facts=facts,
        
        forget_facts=forget_facts,
        
        retain_facts=retain_facts,
        
        detector=detector,
        
        detector_threshold=detector_threshold,
        
        cut_set=best_run["cut_set"],
        
        neutral_bank=neutral_bank,
        
        cut_strength=best_run["cut_strength"],
    
    )
    
    df.to_csv(out_dir / "dpcu_lite_eval.csv", index=False, encoding="utf-8-sig")
    
    variant_breaker_df = evaluate_breaker_cut(
        
        model=model,
        
        facts=search_forget_facts + search_retain_facts,
        
        forget_facts=forget_facts,
        
        retain_facts=retain_facts,
        
        detector=detector,
        
        detector_threshold=detector_threshold,
        
        cut_set=best_run["cut_set"],
        
        neutral_bank=neutral_bank,
        
        cut_strength=best_run["cut_strength"],
    
    )
    
    variant_breaker_df.to_csv(out_dir / "variant_dpcu_lite_eval.csv", index=False, encoding="utf-8-sig")
    
    print("[dpcu] writing summary CSV/PNG/JSON outputs", flush=True)
    
    sweep_df = pd.DataFrame(
        
        [
            
            {
                
                "cut_strength": run["cut_strength"],
                
                "feasible": run["feasible"],
                
                **run["summary"],
            
            }
            
            for run in runs
        
        ]
    
    )
    
    sweep_df.to_csv(out_dir / "strength_sweep_summary.csv", index=False, encoding="utf-8-sig")
    
    plot_cut_eval(static_df, out_dir / "static_path_cut_eval.png", title="Static DPCU-lite cut effect")
    
    plot_cut_eval(df, out_dir / "dpcu_lite_eval.png", title="Breaker-gated DPCU-lite cut effect")
    
    plot_cut_eval(variant_static_df, out_dir / "variant_path_cut_eval.png", title="Variant static DPCU-lite cut effect")
    
    plot_cut_eval(variant_breaker_df, out_dir / "variant_dpcu_lite_eval.png", title="Variant breaker-gated DPCU-lite cut effect")
    
    plot_strength_sweep(sweep_df, out_dir / "strength_sweep_summary.png")
    
    (out_dir / "cut_set.json").write_text(
        
        json.dumps(
            
            {
                
                "trace_candidates": [asdict(x) for x in trace_candidates],
                
                "candidates": [asdict(x) for x in best_run["candidates"]],
                
                "cut_set": [asdict(x) for x in best_run["cut_set"]],
                
                "cut_strength": best_run["cut_strength"],
                
                "detector_threshold": detector_threshold,
                
                "detector_report": detector_report,
                
                "strength_sweep": [
                    
                    {
                        
                        "cut_strength": run["cut_strength"],
                        
                        "feasible": run["feasible"],
                        
                        "summary": run["summary"],
                        
                        "cut_set": [asdict(x) for x in run["cut_set"]],
                    
                    }
                    
                    for run in runs
                
                ],
                
                "search_log": best_run["search_log"],
            
            },
            
            indent=2,
            
            ensure_ascii=False,
        
        ),
        
        encoding="utf-8",
    
    )
    
    print(df.to_string(index=False))
    
    print(f"saved: {out_dir}")



if __name__ == "__main__":
    
    main()
