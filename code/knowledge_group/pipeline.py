
from __future__ import annotations
"""Search one global cut policy and summarize leakage by attack family and target."""


import argparse

import json
import math
import sys

from dataclasses import asdict

from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


import pandas as pd

import yaml


from causal_tracing import run_tracing_multi

from config_utils import ensure_dir, load_config

from data_utils import Fact, load_facts, split_facts

from dpcu_lite import (
    
    CutCandidate,
    
    build_neutral_bank,
    
    evaluate_cut,
    
    expand_fact_variants,
    
    greedy_cut_search,
    
    hard_tail_mean,
    
    load_candidates,
    
    estimate_path_gate_effects,
    
    rerank_candidates_by_single_cut,
    
    select_gate_validated_candidates,
    
    target_first_prob,

)

from tl_pythia_loader import load_from_config

from visualization import plot_causal_tracing, plot_cut_eval, plot_target_prob, plot_strength_sweep






def read_experiment_config(path: str | Path) -> tuple[dict, dict]:
    
    """Load the pipeline configuration and resolve inherited paths."""
    
    def merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    def load_with_parents(config_path: Path, stack: tuple[Path, ...] = ()) -> dict:
        resolved = config_path.resolve()
        if resolved in stack:
            chain = " -> ".join(str(item) for item in (*stack, resolved))
            raise ValueError(f"Circular config inheritance: {chain}")
        with resolved.open("r", encoding="utf-8") as handle:
            current = yaml.safe_load(handle) or {}
        parent_ref = current.get("extends")
        if not parent_ref:
            return current
        
        
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute() and not parent_path.exists():
            parent_path = resolved.parent / parent_path
        parent = load_with_parents(parent_path, (*stack, resolved))
        return merge(parent, current)

    experiment = load_with_parents(Path(path))
    
    base = load_config(experiment["base_config"])
    
    return experiment, base



def strip_answer(answer: str) -> str:
    
    """Normalize an answer before prompt concatenation."""
    
    return answer.strip()



def answer_for_text(text: str) -> str:
    
    """Format an answer consistently with its prompt."""
    
    return text if text.startswith(" ") else f" {text}"



def family_from_id(fact_id: str) -> str:
    
    """Extract the attack family from a fact identifier."""
    
    parts = fact_id.split(":")
    
    return parts[1] if len(parts) > 2 and parts[0].startswith(("f", "r")) else "factual"



def base_id(fact_id: str) -> str:
    
    """Remove variant suffixes from a fact identifier."""
    
    return fact_id.split(":", 1)[0]



def make_fact(
    
    fact: Fact,
    
    suffix: str,
    
    prompt: str,
    
    answer: str,
    
    relation: str | None = None,
    
    subject: str | None = None,
    
    fact_type: str | None = None,

) -> Fact:
    """Construct a validated fact record from source fields."""
    
    clean_answer = answer_for_text(answer.strip())
    
    return Fact(
        
        id=f"{fact.id}:{suffix}",
        
        prompt=prompt,
        
        answer=clean_answer,
        
        type=fact_type or fact.type,
        
        subject=subject if subject is not None else fact.subject,
        
        relation=relation if relation is not None else fact.relation,
        
        unknown_prompt=f"{prompt} unknown",
        
        wrong_answer=" unknown",
    
    )



def relation_attack_facts(fact: Fact) -> list[Fact]:
    """Generate relation-based attacks for selected targets."""
    
    subject = fact.subject
    
    answer = strip_answer(fact.answer)
    
    attacks: list[Fact] = []
    
    if fact.relation == "capital":
        
        attacks.extend(
            
            [
                
                make_fact(fact, "reverse:0", f"{answer} is the capital of", subject, relation="reverse_capital", subject=answer),
                
                make_fact(fact, "clue:0", f"The country whose capital is {answer} is", subject, relation="capital_clue", subject=answer),
                
                make_fact(fact, "clue:1", f"A country with capital city {answer} is", subject, relation="capital_clue", subject=answer),
                
                make_fact(fact, "multihop:0", f"The capital city of {subject}, also known as {answer}, is", answer),
                
                make_fact(fact, "multihop:1", f"If {answer} is the capital of {subject}, then the capital of {subject} is", answer),
            
            ]
        
        )
    
    elif fact.relation == "located_in":
        
        attacks.extend(
            
            [
                
                make_fact(fact, "reverse:0", f"{answer} is home to the", subject, relation="reverse_location", subject=answer),
                
                make_fact(fact, "clue:0", f"The city where the {subject} is located is", answer, relation="location_clue"),
                
                make_fact(fact, "clue:1", f"The {subject} can be found in", answer, relation="location_clue"),
                
                make_fact(fact, "multihop:0", f"Tourists visiting the {subject} are usually in", answer),
            
            ]
        
        )
    
    elif fact.relation == "currency":
        
        attacks.extend(
            
            [
                
                make_fact(fact, "reverse:0", f"The country that uses the {answer} is", subject, relation="reverse_currency", subject=answer),
                
                make_fact(fact, "clue:0", f"The currency used in {subject} is the", answer, relation="currency_clue"),
                
                make_fact(fact, "clue:1", f"People in {subject} pay with the", answer, relation="currency_clue"),
                
                make_fact(fact, "multihop:0", f"If shopping in {subject}, the local currency is the", answer),
            
            ]
        
        )
    
    return attacks



def neighbor_attack_facts(forget_facts: list[Fact]) -> list[Fact]:
    """Generate neighbor-based attacks for selected targets."""
    
    by_answer: dict[str, list[Fact]] = {}
    
    by_subject: dict[str, Fact] = {}
    
    for fact in forget_facts:
        
        by_answer.setdefault(strip_answer(fact.answer), []).append(fact)
        
        by_subject[fact.subject] = fact

    
    attacks: list[Fact] = []
    
    for fact in forget_facts:
        
        answer = strip_answer(fact.answer)
        
        if fact.relation == "capital":
            
            for located in by_answer.get(answer, []):
                
                if located.id == fact.id or located.relation != "located_in":
                    
                    continue
                
                attacks.append(
                    
                    make_fact(
                        
                        fact,
                        
                        f"neighbor:{located.id}:0",
                        
                        f"The city with the {located.subject} is also the capital of {fact.subject}. That city is",
                        
                        answer,
                        
                        relation="neighbor_bridge",
                    
                    )
                
                )
                
                attacks.append(
                    
                    make_fact(
                        
                        fact,
                        
                        f"neighbor:{located.id}:1",
                        
                        f"{located.subject} is in the capital city of {fact.subject}, which is",
                        
                        answer,
                        
                        relation="neighbor_bridge",
                    
                    )
                
                )
        
        if fact.relation == "located_in":
            
            city = answer
            
            for capital_fact in forget_facts:
                
                if capital_fact.relation == "capital" and strip_answer(capital_fact.answer) == city:
                    
                    attacks.append(
                        
                        make_fact(
                            
                            fact,
                            
                            f"neighbor:{capital_fact.id}:0",
                            
                            f"The capital of {capital_fact.subject} contains the {fact.subject}. That city is",
                            
                            city,
                            
                            relation="neighbor_bridge",
                        
                        )
                    
                    )
    
    return attacks






def build_leakage_facts(forget_facts: list[Fact], variants: list[str]) -> list[Fact]:
    
    """Build the complete leakage-evaluation set."""
    
    facts: list[Fact] = []
    
    if "factual" in variants or "paraphrase" in variants or "alias" in variants:
        
        facts.extend(expand_fact_variants(forget_facts, variants))
    
    enabled = set(variants)
    
    for fact in forget_facts:
        
        generated = relation_attack_facts(fact)
        
        if "reverse" not in enabled:
            
            generated = [x for x in generated if ":reverse:" not in x.id]
        
        if "clue" not in enabled:
            
            generated = [x for x in generated if ":clue:" not in x.id]
        
        if "multihop" not in enabled:
            
            generated = [x for x in generated if ":multihop:" not in x.id]
        
        facts.extend(generated)
    
    if "neighbor" in enabled:
        
        facts.extend(neighbor_attack_facts(forget_facts))

    
    deduped: dict[tuple[str, str], Fact] = {}
    
    for fact in facts:
        
        deduped[(fact.prompt, fact.answer)] = fact
    
    return list(deduped.values())



def baseline_eval(facts: list[Fact], model) -> pd.DataFrame:
    
    """Evaluate the unmodified model on the supplied facts."""
    
    rows = []
    
    for fact in facts:
        
        rows.append(
            
            {
                
                "id": fact.id,
                
                "base_id": base_id(fact.id),
                
                "family": family_from_id(fact.id),
                
                "type": fact.type,
                
                "prompt": fact.prompt,
                
                "answer": fact.answer,
                
                "subject": fact.subject,
                
                "relation": fact.relation,
                
                "first_token_prob": target_first_prob(model, fact.prompt, fact.answer),
            
            }
        
        )
    
    return pd.DataFrame(rows)



def summarize_eval(df: pd.DataFrame, tail_fraction: float) -> dict:
    
    """Aggregate per-example records into summary metrics."""
    
    forget = df[df["type"] == "forget"]
    
    retain = df[df["type"] == "retain"]
    
    values = [float(x) for x in forget["cut_first_token_prob"].tolist()]
    
    base_mean = float(forget["base_first_token_prob"].mean()) if not forget.empty else 0.0
    
    cut_mean = float(forget["cut_first_token_prob"].mean()) if not forget.empty else 0.0
    
    summary = {
        
        "forget_count": int(len(forget)),
        
        "retain_count": int(len(retain)),
        
        "forget_base_mean": base_mean,
        
        "forget_cut_mean": cut_mean,
        
        "forget_relative_drop": (base_mean - cut_mean) / base_mean if base_mean > 0 else 0.0,
        
        "forget_worst_cut_prob": max(values) if values else 0.0,
        
        "forget_tail_cut_prob": hard_tail_mean(values, tail_fraction),
        
        "forget_max_prob_increase": max(
            float((-forget["prob_drop"]).max()), 0.0
        ) if not forget.empty else 0.0,
        
        "retain_drop_mean": float(retain["prob_drop"].mean()) if not retain.empty else 0.0,
        
        "retain_max_drop": max(
            float(retain["prob_drop"].max()), 0.0
        ) if not retain.empty else 0.0,
        
        "retain_kl_mean": float(retain["retain_kl"].mean()) if "retain_kl" in retain and not retain.empty else 0.0,
    
    }
    
    return summary



def family_summary(df: pd.DataFrame) -> pd.DataFrame:
    
    """Aggregate evaluation metrics by attack family."""
    
    rows = []
    
    for (kind, family), group in df.groupby(["type", "family"]):
        
        rows.append(
            
            {
                
                "type": kind,
                
                "family": family,
                
                "count": int(len(group)),
                
                "base_mean": float(group["base_first_token_prob"].mean()),
                
                "cut_mean": float(group["cut_first_token_prob"].mean()),
                
                "worst_cut_prob": float(group["cut_first_token_prob"].max()),
                
                "drop_mean": float(group["prob_drop"].mean()),
                
                "max_prob_increase": max(
                    float((-group["prob_drop"]).max()), 0.0
                ),
            
            }
        
        )
    
    return pd.DataFrame(rows).sort_values(["type", "family"])



def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    
    """Aggregate evaluation metrics by target fact."""
    
    rows = []
    
    for (kind, root), group in df.groupby(["type", "base_id"]):
        
        rows.append(
            
            {
                
                "type": kind,
                
                "base_id": root,
                
                "count": int(len(group)),
                
                "base_mean": float(group["base_first_token_prob"].mean()),
                
                "cut_mean": float(group["cut_first_token_prob"].mean()),
                
                "worst_cut_prob": float(group["cut_first_token_prob"].max()),
                
                "drop_mean": float(group["prob_drop"].mean()),
                
                "max_prob_increase": max(
                    float((-group["prob_drop"]).max()), 0.0
                ),
            
            }
        
        )
    
    return pd.DataFrame(rows).sort_values(["type", "base_id"])






def policy_score(summary: dict, selection: dict, cut_size: int) -> tuple[bool, bool, float, float]:
    
    """Compute the scalar score used to compare global policies."""
    
    forget_acceptable = (
        
        summary["forget_relative_drop"] >= float(selection["min_forget_relative_drop"])
        
        and summary["forget_worst_cut_prob"] <= float(selection["max_forget_worst_cut_prob"])
        
        and summary["forget_max_prob_increase"] <= float(selection["max_forget_prob_increase"])
    
    )
    
    retain_acceptable = (
        
        summary["retain_kl_mean"] <= float(selection["retain_kl_budget"])
        
        and summary["retain_max_drop"] <= float(selection["retain_max_drop"])
    
    )
    
    feasible = forget_acceptable and retain_acceptable
    
    forget_score = (
        
        summary["forget_cut_mean"]
        
        + float(selection["worst_weight"]) * summary["forget_worst_cut_prob"]
        
        + float(selection["tail_weight"]) * summary["forget_tail_cut_prob"]
    
    )
    
    retain_risk = summary["retain_kl_mean"] + summary["retain_max_drop"]
    
    score = (
        
        forget_score
        
        + float(selection["retention_risk_weight"]) * retain_risk
        
        + float(selection["cut_size_penalty"]) * cut_size
    
    )
    
    violation = (
        
        max(float(selection["min_forget_relative_drop"]) - summary["forget_relative_drop"], 0.0)
        
        + max(summary["forget_worst_cut_prob"] - float(selection["max_forget_worst_cut_prob"]), 0.0)
        
        + max(summary["retain_kl_mean"] - float(selection["retain_kl_budget"]), 0.0)
        
        + max(summary["retain_max_drop"] - float(selection["retain_max_drop"]), 0.0)
        
        + max(summary["forget_max_prob_increase"] - float(selection["max_forget_prob_increase"]), 0.0)
    
    )
    
    return forget_acceptable, feasible, float(score), float(violation)



def policy_priority_key_legacy(run: dict) -> tuple:
    
    """Build the fallback policy-ordering key."""
    
    summary = run["summary"]
    
    return (
        
        not run["forget_acceptable"],
        
        not run["feasible"],
        
        summary["retain_max_drop"],
        
        summary["retain_kl_mean"],
        
        summary["retain_drop_mean"],
        
        run["violation"],
        
        summary["forget_worst_cut_prob"],
        
        summary["forget_cut_mean"],
        
        run["score"],
    
    )



def policy_priority_key(run: dict) -> tuple:
    """Build the constraint-aware policy-ordering key."""
    summary = run["summary"]
    selection = run.get("selection", {})

    retain_kl_budget = float(selection.get("retain_kl_budget", float("inf")))
    retain_drop_budget = float(selection.get("retain_max_drop", float("inf")))
    retain_violation = (
        max(summary["retain_kl_mean"] - retain_kl_budget, 0.0)
        / max(retain_kl_budget, 1e-12)
        + max(summary["retain_max_drop"] - retain_drop_budget, 0.0)
        / max(retain_drop_budget, 1e-12)
    )

    min_relative_drop = float(selection.get("min_forget_relative_drop", 0.0))
    max_worst_prob = float(
        selection.get("max_forget_worst_cut_prob", float("inf"))
    )
    max_increase = float(
        selection.get("max_forget_prob_increase", float("inf"))
    )
    forget_violation = (
        max(min_relative_drop - summary["forget_relative_drop"], 0.0)
        / max(min_relative_drop, 1e-12)
        + max(summary["forget_worst_cut_prob"] - max_worst_prob, 0.0)
        / max(max_worst_prob, 1e-12)
        + max(summary["forget_max_prob_increase"] - max_increase, 0.0)
        / max(max_increase, 1e-12)
    )

    return (
        retain_violation > 0.0,
        retain_violation,
        not run["forget_acceptable"],
        forget_violation,
        summary["forget_worst_cut_prob"],
        summary["forget_cut_mean"],
        summary["retain_max_drop"],
        summary["retain_kl_mean"],
        run["score"],
    )





def search_global_policy(
    
    model,
    
    forget_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    candidates: list[CutCandidate],
    
    neutral_bank,
    
    base_cfg: dict,
    
    experiment: dict,

) -> tuple[dict, list[dict]]:
    """Search candidate sites and strengths for one shared policy."""
    
    dcfg = base_cfg["dpcu"]
    
    selection = experiment["selection"]
    
    strengths = [float(x) for x in experiment.get("cut_strength_sweep", dcfg["cut_strength_sweep"])]
    minimal_cfg = experiment.get("minimal_path_search", {})
    minimal_enabled = bool(minimal_cfg.get("enabled", False))
    candidate_limit = int(
        minimal_cfg.get("candidate_limit", dcfg["top_k_paths"])
        if minimal_enabled else dcfg["top_k_paths"]
    )
    max_paths = int(
        minimal_cfg.get("max_paths", dcfg["max_cut_size"])
        if minimal_enabled else dcfg["max_cut_size"]
    )
    if candidate_limit < 1 or max_paths < 1:
        raise ValueError("minimal_path_search candidate_limit and max_paths must be positive")
    
    runs = []
    
    for index, strength in enumerate(strengths, start=1):
        
        print(
            
            f"[6.9_defence] global strength {index}/{len(strengths)} = {strength:g}",
            
            flush=True,
        
        )
        
        ranked = rerank_candidates_by_single_cut(
            
            model=model,
            
            forget_facts=forget_facts,
            
            retain_facts=retain_facts,
            
            candidates=list(candidates),
            
            neutral_bank=neutral_bank,
            
            top_k=min(int(dcfg["top_k_paths"]), candidate_limit),
            
            retain_kl_budget=float(selection["retain_kl_budget"]),
            
            retain_kl_weight=float(selection["retention_risk_weight"]),
            
            cut_size_penalty=float(selection["cut_size_penalty"]),
            
            max_retain_prob_drop=float(selection["retain_max_drop"]),
            
            cut_strength=strength,
            
            forget_worst_weight=float(selection["worst_weight"]),
            
            forget_tail_weight=float(selection["tail_weight"]),
            
            forget_tail_fraction=float(dcfg.get("forget_tail_fraction", 0.4)),
            
            min_forget_base_prob=float(dcfg.get("min_forget_base_prob", 0.0)),
            
            max_forget_prob_increase=float(dcfg.get("max_forget_prob_increase", 1.0)),
            
            forget_increase_weight=float(dcfg.get("forget_increase_weight", 50.0)),
            retain_prob_drop_weight=float(selection["retention_risk_weight"]),
        
        )
        
        cut_set, search_log = greedy_cut_search(
            
            model=model,
            
            forget_facts=forget_facts,
            
            retain_facts=retain_facts,
            
            candidates=ranked,
            
            neutral_bank=neutral_bank,
            
            max_cut_size=min(int(dcfg["max_cut_size"]), max_paths),
            
            retain_kl_budget=float(selection["retain_kl_budget"]),
            
            target_prob_stop=float(dcfg["target_prob_stop"]),
            
            retain_kl_weight=float(selection["retention_risk_weight"]),
            
            cut_size_penalty=float(selection["cut_size_penalty"]),
            
            max_retain_prob_drop=float(selection["retain_max_drop"]),
            
            cut_strength=strength,
            
            forget_worst_weight=float(selection["worst_weight"]),
            
            forget_tail_weight=float(selection["tail_weight"]),
            
            forget_tail_fraction=float(dcfg.get("forget_tail_fraction", 0.4)),
            
            min_forget_base_prob=float(dcfg.get("min_forget_base_prob", 0.0)),
            
            max_forget_prob_increase=float(dcfg.get("max_forget_prob_increase", 1.0)),
            
            forget_increase_weight=float(dcfg.get("forget_increase_weight", 50.0)),
            retain_prob_drop_weight=float(selection["retention_risk_weight"]),
            pair_lookahead=bool(minimal_cfg.get("pair_lookahead", False)),
        
        )
        
        if cut_set:
            
            eval_df = evaluate_cut(model, forget_facts + retain_facts, cut_set, neutral_bank, cut_strength=strength)
            
            for col, value in [("base_id", ""), ("family", "")]:
                
                if col not in eval_df:
                    
                    eval_df[col] = value
            
            eval_df["base_id"] = eval_df["id"].map(base_id)
            
            eval_df["family"] = eval_df["id"].map(family_from_id)
            
            summary = summarize_eval(eval_df, float(dcfg.get("forget_tail_fraction", 0.4)))
            
            forget_acceptable, feasible, score, violation = policy_score(summary, selection, len(cut_set))
        
        else:
            
            eval_df = None
            
            summary = None
            
            forget_acceptable = False
            
            feasible = False
            
            score = None
            
            violation = None
        
        runs.append(
            
            {
                
                "strength": strength,
                "selection": selection,
                
                "forget_acceptable": forget_acceptable,
                
                "feasible": feasible,
                
                "score": score,
                
                "violation": violation,
                
                "summary": summary,
                
                "shortlist": ranked,
                "cut_set": cut_set,
                
                "search_log": search_log,
                
                "eval": eval_df,
            
            }
        
        )
    
    valid = [run for run in runs if run["cut_set"]]
    
    if not valid:
        
        raise RuntimeError("No global cut set found")
    
    valid.sort(key=policy_priority_key)
    
    return valid[0], runs



def annotate_eval(df: pd.DataFrame) -> pd.DataFrame:
    
    """Attach family and target metadata to evaluation records."""
    
    result = df.copy()
    
    result["base_id"] = result["id"].map(base_id)
    
    result["family"] = result["id"].map(family_from_id)
    
    return result



def calibrate_group_strengths(
    
    model,
    
    leakage_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    cut_set: list[CutCandidate],
    
    neutral_bank,
    
    base_cfg: dict,
    
    experiment: dict,

) -> tuple[dict[str, dict], list[dict]]:
    """Calibrate cut strengths for selected candidate groups."""
    
    dcfg = base_cfg["dpcu"]
    
    selection = experiment["selection"]
    
    strengths = [float(x) for x in experiment.get("cut_strength_sweep", dcfg["cut_strength_sweep"])]
    
    tail_fraction = float(dcfg.get("forget_tail_fraction", 0.4))
    
    grouped: dict[str, list[Fact]] = {}
    
    for fact in leakage_facts:
        
        grouped.setdefault(base_id(fact.id), []).append(fact)

    
    selected: dict[str, dict] = {}
    
    all_runs: list[dict] = []
    
    for group_index, (group_id, group_facts) in enumerate(grouped.items(), start=1):
        
        group_runs = []
        
        print(
            
            f"[6.9_defence] calibrating group {group_index}/{len(grouped)} {group_id} "
            
            f"({len(group_facts)} leakage probes)",
            
            flush=True,
        
        )
        
        for strength in strengths:
            
            eval_df = annotate_eval(
                
                evaluate_cut(
                    
                    model,
                    
                    group_facts + retain_facts,
                    
                    cut_set,
                    
                    neutral_bank,
                    
                    cut_strength=strength,
                
                )
            
            )
            
            eval_df["policy_base_id"] = group_id
            
            eval_df["cut_strength"] = strength
            
            summary = summarize_eval(eval_df, tail_fraction)
            
            forget_acceptable, feasible, score, violation = policy_score(summary, selection, len(cut_set))
            
            run = {
                
                "group_id": group_id,
                
                "strength": strength,
                
                "forget_acceptable": forget_acceptable,
                
                "feasible": feasible,
                
                "score": score,
                
                "violation": violation,
                
                "summary": summary,
                
                "eval": eval_df,
            
            }
            
            group_runs.append(run)
            
            all_runs.append(run)
        
        group_runs.sort(key=policy_priority_key)
        
        selected[group_id] = group_runs[0]
        
        best = group_runs[0]
        
        print(
            
            f"[6.9_defence] selected group {group_id}: strength={best['strength']:g}, "
            
            f"feasible={best['feasible']}",
            
            flush=True,
        
        )
    
    return selected, all_runs



def json_run(run: dict) -> dict:
    
    """Convert one run result into JSON-compatible values."""
    
    return {
        
        "strength": run["strength"],
        
        "forget_acceptable": run["forget_acceptable"],
        
        "feasible": run["feasible"],
        
        "score": run["score"],
        
        "violation": run["violation"],
        
        "summary": run["summary"],
        
        "shortlist": [
            asdict(candidate) for candidate in run.get("shortlist", [])
        ],
        "cut_set": [asdict(candidate) for candidate in run["cut_set"]],
        
        "search_log": run["search_log"],
    
    }



def json_group_run(run: dict) -> dict:
    
    """Convert grouped run results into JSON-compatible values."""
    
    return {
        
        "group_id": run["group_id"],
        
        "strength": run["strength"],
        
        "forget_acceptable": run["forget_acceptable"],
        
        "feasible": run["feasible"],
        
        "score": run["score"],
        
        "violation": run["violation"],
        
        "summary": run["summary"],
    
    }






def main(
    default_config: str = "code/knowledge_group/base_config.yaml",
    require_forget_id: bool = False,
) -> None:
    """Parse arguments and run the configured workflow."""
    
    parser = argparse.ArgumentParser(description="Pythia-6.9B causal-path-gated defence experiment")
    
    parser.add_argument("--config", default=default_config)
    
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--forget-id",
        default="",
        help="Restrict the experiment to one base forget fact, for example f001.",
    )
    
    args = parser.parse_args()

    
    experiment, base_cfg = read_experiment_config(args.config)
    
    out_dir = ensure_dir(args.out_dir or experiment["out_dir"])
    
    if any(out_dir.iterdir()):
        
        raise RuntimeError(f"Output directory must be empty for a fresh run: {out_dir}")

    
    facts = load_facts(base_cfg["data"]["facts_path"])
    
    forget_facts, retain_facts = split_facts(facts)
    if require_forget_id and not args.forget_id:
        parser.error("--forget-id is required for the single-fact entry point")
    if args.forget_id:
        matched = [fact for fact in forget_facts if fact.id == args.forget_id]
        if len(matched) != 1:
            available = ", ".join(fact.id for fact in forget_facts)
            parser.error(
                f"unknown --forget-id {args.forget_id!r}; available IDs: {available}"
            )
        forget_facts = matched
    
    leakage_variants = list(experiment["leakage_eval"]["forget_variants"])
    
    retain_variants = list(experiment["leakage_eval"].get("retain_variants", ["factual", "paraphrase", "alias"]))
    
    leakage_facts = build_leakage_facts(forget_facts, leakage_variants)
    
    expanded_retain = expand_fact_variants(retain_facts, retain_variants)
    
    tracing_facts = leakage_facts + expanded_retain

    
    print("[6.9_defence] loading model", flush=True)
    
    model = load_from_config(base_cfg)

    
    print("[6.9_defence] 01 baseline first-token evaluation", flush=True)
    
    baseline = baseline_eval(tracing_facts, model)
    
    baseline.to_csv(out_dir / "01_leakage_baseline.csv", index=False, encoding="utf-8-sig")
    
    plot_target_prob(
        
        baseline[["id", "type", "first_token_prob"]],
        
        out_dir / "01_leakage_baseline.png",
        
        title="Leakage baseline first-token probability",
    
    )

    
    
    
    min_retain_search_prob = float(
        experiment["leakage_eval"].get("min_retain_base_prob_for_search", 0.0)
    )
    retain_search_ids = set(
        baseline.loc[
            (baseline["type"] == "retain")
            & (baseline["first_token_prob"] >= min_retain_search_prob),
            "id",
        ].tolist()
    )
    search_retain = [fact for fact in expanded_retain if fact.id in retain_search_ids]
    if not search_retain:
        raise RuntimeError(
            "No retain prompt passed leakage_eval.min_retain_base_prob_for_search; "
            "lower the threshold or improve the retain prompts."
        )
    print(
        f"[6.9_defence] retain search set: {len(search_retain)}/"
        f"{len(expanded_retain)} prompts (base probability >= "
        f"{min_retain_search_prob:g})",
        flush=True,
    )

    
    print("[6.9_defence] 02 causal tracing on leakage set", flush=True)
    
    corrupt_strategies = list(
        
        base_cfg["tracing"].get(
            
            "corrupt_strategies",
            
            [base_cfg["tracing"].get("corrupt_strategy", "subject_unknown")],
        
        )
    
    )
    
    tracing = run_tracing_multi(
        
        model,
        
        leakage_facts + search_retain,
        
        sites=list(base_cfg["tracing"]["sites"]),
        
        corrupt_strategies=corrupt_strategies,
        
        max_examples=base_cfg["tracing"].get("max_examples"),
    
    )
    
    trace_path = out_dir / "02_leakage_causal_tracing.csv"
    
    tracing.to_csv(trace_path, index=False, encoding="utf-8-sig")
    
    plot_causal_tracing(tracing, out_dir / "02_leakage_causal_tracing.png")

    
    dcfg = base_cfg["dpcu"]
    
    candidates = load_candidates(
        
        trace_path,
        
        top_k=int(dcfg.get("trace_top_k_paths", dcfg["top_k_paths"])),
        
        site=dcfg.get("site"),
        
        retain_trace_penalty=float(dcfg.get("retain_trace_penalty", 1.0)),
        
        min_forget_restore=float(dcfg.get("min_forget_restore", 0.0)),
        
        min_forget_positive_fraction=float(dcfg.get("min_forget_positive_fraction", 0.0)),
        
        robust_quantile=float(dcfg.get("robust_quantile", 0.5)),
        
        consistency_weight=float(dcfg.get("consistency_weight", 0.0)),
        min_forget_base_prob=float(dcfg.get("min_forget_base_prob", 0.0)),
        restore_effect_epsilon=float(dcfg.get("restore_effect_epsilon", 0.0)),
    
    )
    
    trace_candidate_count = len(candidates)
    
    print(f"[6.9_defence] 03 building neutral bank for {len(candidates)} candidates", flush=True)
    
    
    
    
    neutral_strategy = str(
        experiment.get(
            "neutral_strategy",
            dcfg.get("neutral_strategy", "unknown_mean"),
        )
    )
    neutral_strategy = neutral_strategy.strip().lower()
    configured_cut_strengths = [
        float(value)
        for value in experiment.get(
            "cut_strength_sweep", dcfg.get("cut_strength_sweep", [])
        )
    ]
    if not configured_cut_strengths or any(
        not math.isfinite(value) or value <= 0.0
        for value in configured_cut_strengths
    ):
        raise ValueError("cut_strength_sweep must contain finite positive values")
    if neutral_strategy == "zero":
        if any(value > 1.0 for value in configured_cut_strengths):
            raise ValueError(
                "zero neutral_strategy requires every cut strength <= 1.0; "
                "strength > 1 extrapolates past zero and is not zero ablation"
            )
    print(
        f"[6.9_defence] neutral replacement strategy: {neutral_strategy}",
        flush=True,
    )
    neutral_bank = build_neutral_bank(
        model,
        retain_facts,
        candidates,
        strategy=neutral_strategy,
    )

    
    
    
    
    
    
    
    gate_cfg = experiment.get("path_gate", {})
    
    configured_strengths = gate_cfg.get("intervention_strengths")
    
    if configured_strengths is None:
        
        configured_strengths = [gate_cfg.get("intervention_strength", 0.35)]
    
    gate_effects = estimate_path_gate_effects(
        
        model=model,
        
        forget_facts=leakage_facts,
        
        retain_facts=search_retain,
        
        candidates=candidates,
        
        neutral_bank=neutral_bank,
        
        gate_strengths=[float(value) for value in configured_strengths],
        
        min_forget_drop=float(gate_cfg.get("min_forget_drop", 0.005)),
        
        max_retain_kl=float(gate_cfg.get("max_retain_kl", 0.10)),
        
        max_retain_prob_drop=float(gate_cfg.get("max_retain_prob_drop", 0.05)),
        
        min_specificity=float(gate_cfg.get("min_specificity", 0.5)),
    
    )
    
    
    
    
    
    gate_effects["selected_for_minimal_path_search"] = False
    
    gate_effects.to_csv(out_dir / "03_path_gate_effects.csv", index=False, encoding="utf-8-sig")
    
    candidates = select_gate_validated_candidates(
        
        candidates,
        
        gate_effects,
        
        min_validated_candidates=int(gate_cfg.get("min_validated_candidates", 1)),
    
    )
    
    selected_gate_keys = {(candidate.layer, candidate.site) for candidate in candidates}
    
    gate_effects["selected_for_minimal_path_search"] = [
        
        (int(row.layer), str(row.site)) in selected_gate_keys
        
        for row in gate_effects.itertuples(index=False)
    
    ]
    
    gate_effects.to_csv(out_dir / "03_path_gate_effects.csv", index=False, encoding="utf-8-sig")
    
    print(
        
        f"[6.9_defence] path gate retained {len(candidates)} candidates for minimal-path search",
        
        flush=True,
    
    )

    
    best, runs = search_global_policy(
        
        model,
        
        leakage_facts,
        
        search_retain,
        
        candidates,
        
        neutral_bank,
        
        base_cfg,
        
        experiment,
    
    )
    
    print(
        
        f"[6.9_defence] selected global policy: strength={best['strength']:g}, "
        
        f"cuts={len(best['cut_set'])}, feasible={best['feasible']}",
        
        flush=True,
    
    )

    
    group_cfg = experiment.get("group_strength_optimization", {})
    
    if bool(group_cfg.get("enabled", True)):
        
        selected_groups, group_runs = calibrate_group_strengths(
            
            model,
            
            leakage_facts,
            
            expanded_retain,
            
            best["cut_set"],
            
            neutral_bank,
            
            base_cfg,
            
            experiment,
        
        )
        
        
        
        
        
        final_eval = pd.concat(
            
            [selected_groups[group_id]["eval"] for group_id in selected_groups],
            
            ignore_index=True,
        
        )
    
    else:
        
        selected_groups = {}
        
        group_runs = []
        
        final_eval = annotate_eval(
            
            evaluate_cut(
                
                model,
                
                leakage_facts + expanded_retain,
                
                best["cut_set"],
                
                neutral_bank,
                
                cut_strength=best["strength"],
            
            )
        
        )
        
        final_eval["policy_base_id"] = "global"
        
        final_eval["cut_strength"] = best["strength"]
    
    final_eval.to_csv(out_dir / "03_group_strength_leakage_eval.csv", index=False, encoding="utf-8-sig")
    
    plot_cut_eval(final_eval, out_dir / "03_group_strength_leakage_eval.png", title="Group-calibrated global-cut defence")

    
    by_family = family_summary(final_eval)
    
    by_target = target_summary(final_eval)
    
    by_family.to_csv(out_dir / "04_family_summary.csv", index=False, encoding="utf-8-sig")
    
    by_target.to_csv(out_dir / "05_target_summary.csv", index=False, encoding="utf-8-sig")

    
    sweep_rows = []
    
    for run in runs:
        
        
        if not run["summary"] or not run["cut_set"]:
            continue
        
        row = {
            
            "cut_strength": run["strength"],
            
            "cut_size": len(run["cut_set"]),
            
            "forget_acceptable": run["forget_acceptable"],
            
            "feasible": run["feasible"],
            
            "score": run["score"],
            
            "violation": run["violation"],
        
        }
        
        if run["summary"]:
            
            row.update(run["summary"])
        
        sweep_rows.append(row)
    
    sweep = pd.DataFrame(sweep_rows)
    
    sweep.to_csv(out_dir / "06_global_strength_sweep.csv", index=False, encoding="utf-8-sig")
    
    plot_strength_sweep(sweep, out_dir / "06_global_strength_sweep.png", title="Causal-path defence sweep")

    
    group_sweep_rows = [
        
        {
            
            "group_id": run["group_id"],
            
            "cut_strength": run["strength"],
            
            "forget_acceptable": run["forget_acceptable"],
            
            "feasible": run["feasible"],
            
            "score": run["score"],
            
            "violation": run["violation"],
            
            **run["summary"],
        
        }
        
        for run in group_runs
    
    ]
    
    group_sweep = pd.DataFrame(group_sweep_rows)
    if group_sweep.empty:
        group_sweep = pd.DataFrame(
            columns=["group_id", "cut_strength", "forget_acceptable", "feasible", "score", "violation"]
        )
    group_sweep.to_csv(
        
        out_dir / "07_group_strength_sweep.csv", index=False, encoding="utf-8-sig"
    
    )

    
    payload = {
        
        "fresh_run": True,
        
        "reads_previous_outputs": False,
        
        
        "resolved_experiment_config": experiment,
        "resolved_base_config": base_cfg,
        
        "base_config": experiment["base_config"],
        
        "leakage_variants": leakage_variants,
        
        "retain_variants": retain_variants,
        
        "corrupt_strategies": corrupt_strategies,
        
        "leakage_fact_count": len(leakage_facts),
        
        "retain_fact_count": len(expanded_retain),
        
        "retain_unique_count": len(expanded_retain),
        
        "retain_policy_exposure_count": int((final_eval["type"] == "retain").sum()),
        "retain_search_count": len(search_retain),
        "retain_search_min_base_prob": min_retain_search_prob,
        
        "trace_candidate_count": trace_candidate_count,
        
        "gate_selected_candidate_count": len(candidates),
        
        "path_gate_role": "causal_intervention_only",
        
        "path_gate_validated_count": int(gate_effects["gate_validated"].sum()),
        
        "path_gate_effects": gate_effects.to_dict(orient="records"),
        
        "final_summary": summarize_eval(final_eval, float(dcfg.get("forget_tail_fraction", 0.4))),
        
        "family_summary": by_family.to_dict(orient="records"),
        
        "target_summary": by_target.to_dict(orient="records"),
        
        "selected_policy": json_run(best),
        
        "strength_sweep": [json_run(run) for run in runs],
        
        "group_strength_optimization": bool(group_cfg.get("enabled", True)),
        
        "selected_group_strengths": {
            
            key: json_group_run(value) for key, value in selected_groups.items()
        
        },
        
        "group_strength_sweeps": [json_group_run(run) for run in group_runs],
        
        
        
        "meets_unlearning_target": bool(
            policy_score(
                summarize_eval(
                    final_eval,
                    float(dcfg.get("forget_tail_fraction", 0.4)),
                ),
                experiment["selection"],
                len(best["cut_set"]),
            )[1]
        ),
    
    }
    
    (out_dir / "summary.json").write_text(
        
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    
    )
    
    print(json.dumps(payload["final_summary"], indent=2), flush=True)
    
    print(f"[6.9_defence] done: {out_dir}", flush=True)



if __name__ == "__main__":
    
    main()
