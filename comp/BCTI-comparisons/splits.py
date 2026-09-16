from __future__ import annotations

"""Splits utilities."""

import math
from collections import defaultdict

import pandas as pd

from data_utils import Fact
from dpcu_lite import expand_fact_variants
from knowledge_group.pipeline import build_leakage_facts


def dedupe(facts: list[Fact]) -> list[Fact]:
    unique: dict[tuple[str, str], Fact] = {}
    for fact in facts:
        unique[(fact.prompt, fact.answer)] = fact
    return list(unique.values())


def prompt_key(value: str) -> str:
    """Return a normalized prompt key."""

    return " ".join(value.casefold().strip().split())


def without_overlap(facts: list[Fact], used: list[Fact]) -> list[Fact]:
    prompts = {prompt_key(fact.prompt) for fact in used}
    return [fact for fact in facts if prompt_key(fact.prompt) not in prompts]


def family(fact: Fact) -> str:
    for value in (
        "causal_probe_forward",
        "causal_probe_reverse",
        "paraphrase",
        "alias",
        "reverse",
        "clue",
        "multihop",
        "neighbor",
    ):
        if f":{value}" in fact.id:
            return value
    return "factual"


def base_id(fact: Fact) -> str:
    return fact.id.split(":", 1)[0]


def causal_direction(fact: Fact) -> str:
    """Determine the causal direction."""

    relation = str(fact.relation or "")
    if family(fact) in {"reverse", "causal_probe_reverse"}:
        return "reverse"
    if relation.startswith("reverse_") or relation == "capital_clue":
        return "reverse"
    return "forward"


def augment(facts: list[Fact], prefixes: list[str]) -> list[Fact]:
    if not prefixes:
        raise ValueError("fit_prompt_prefixes cannot be empty")
    rows: list[Fact] = []
    for prefix_index, prefix in enumerate(prefixes):
        for fact in facts:
            rows.append(
                Fact(
                    id=f"{fact.id}:fitprefix:{prefix_index}",
                    prompt=f"{prefix}{fact.prompt}",
                    answer=fact.answer,
                    type=fact.type,
                    subject=fact.subject,
                    relation=fact.relation,
                    unknown_prompt=fact.unknown_prompt,
                    wrong_answer=fact.wrong_answer,
                )
            )
    return dedupe(rows)


def normalized_text(value: str) -> str:
    return prompt_key(value)


def answer_leaks_into_prompt(fact: Fact) -> bool:
    answer = normalized_text(fact.answer)
    return bool(answer and answer in normalized_text(fact.prompt))


def _probe(
    base: Fact,
    family_name: str,
    index: int,
    prompt: str,
    answer: str,
    *,
    cue_subject: str | None = None,
) -> Fact:
    return Fact(
        id=f"{base.id}:{family_name}:{index}",
        prompt=prompt,
        answer=answer if answer.startswith(" ") else f" {answer}",
        type=base.type,
        
        
        
        subject=cue_subject if cue_subject is not None else base.subject,
        relation=base.relation,
        unknown_prompt=f"{prompt} unknown",
        wrong_answer=" unknown",
    )


def build_nonleaking_causal_probes(base_facts: list[Fact]) -> list[Fact]:
    """Build nonleaking causal probes."""

    rows: list[Fact] = []
    for fact in base_facts:
        subject = str(fact.subject).strip()
        answer = fact.answer.strip()
        if fact.relation == "capital":
            forward = [
                f"The administrative capital of {subject} is",
                f"{subject}'s capital city is",
                f"Name the capital of {subject}:",
                f"For {subject}, the national capital is",
                f"The capital city of {subject} is",
                f"{subject}'s capital is",
                f"{subject} has its capital in",
                f"The seat of government in {subject} is",
            ]
            reverse = [
                (f"{answer} serves as the capital of", subject),
                (f"The country governed from {answer} is", subject),
                (f"{answer} is the capital city of", subject),
                (f"The nation whose capital is {answer} is", subject),
                (f"Identify the country whose capital is {answer}:", subject),
                (f"{answer} is the seat of government of", subject),
                (f"The national capital {answer} belongs to", subject),
                (f"Which country has {answer} as its capital?", subject),
                (f"What country is {answer} the capital of?", subject),
                (f"Country with capital {answer}:", subject),
                (f"{answer} is the national capital of which country?", subject),
                (f"The country whose main capital city is {answer} is", subject),
            ]
        elif fact.relation == "located_in":
            forward = [
                f"The landmark {subject} is located in",
                f"Visitors can find the {subject} in",
                f"The city containing the {subject} is",
                f"A trip to the {subject} takes you to",
                f"The {subject} stands in",
                f"The location of the {subject} is",
                f"The {subject} can be visited in",
                f"To see the {subject}, travel to",
            ]
            reverse = [
                (f"A famous landmark in {answer} is the", subject),
                (f"{answer} is home to the landmark called the", subject),
                (f"One attraction located in {answer} is the", subject),
                (f"The landmark associated with {answer} is the", subject),
                (f"Name the landmark found in {answer}:", subject),
                (f"A visitor to {answer} can see the", subject),
                (f"The well-known monument situated in {answer} is the", subject),
                (f"Which landmark is located in {answer}? The", subject),
                (f"What landmark is {answer} known for? The", subject),
                (f"Which landmark can be found in {answer}? The", subject),
                (f"{answer} contains the famous", subject),
                (f"Landmark located in {answer}:", subject),
            ]
        elif fact.relation == "currency":
            forward = [
                f"The official currency used by {subject} is the",
                f"In {subject}, prices are denominated in",
                f"Residents of {subject} commonly pay in",
                f"The monetary unit of {subject} is the",
                f"{subject}'s currency is the",
                f"The currency of {subject} is the",
                f"People in {subject} use the",
                f"Money used in {subject} is the",
            ]
            reverse = [
                (f"The {answer} is the currency of", subject),
                (f"The nation associated with the {answer} is", subject),
                (f"A country that pays in {answer} is", subject),
                (f"The {answer} is used as money in", subject),
                (f"Name the country whose currency is the {answer}:", subject),
                (f"The {answer} serves as legal tender in", subject),
                (f"Which nation uses the {answer}?", subject),
                (f"Prices are denominated in {answer} in", subject),
                (f"What country uses the {answer} as currency?", subject),
                (f"Country using the {answer}:", subject),
                (f"The {answer} is the official currency of which nation?", subject),
                (f"The country that uses {answer} for payment is", subject),
            ]
        else:
            forward = []
            reverse = []
        rows.extend(
            _probe(fact, "causal_probe_forward", index, prompt, fact.answer)
            for index, prompt in enumerate(forward)
        )
        rows.extend(
            _probe(
                fact,
                "causal_probe_reverse",
                index,
                prompt,
                target,
                cue_subject=answer,
            )
            for index, (prompt, target) in enumerate(reverse)
        )
    leaked = [fact.id for fact in rows if answer_leaks_into_prompt(fact)]
    if leaked:
        raise RuntimeError(f"generated causal probes leak their target answer: {leaked}")
    return dedupe(rows)


def split_retain(
    retain_facts: list[Fact], variants: list[str]
) -> tuple[list[Fact], list[Fact], list[Fact]]:
    groups: dict[str, list[Fact]] = {}
    for fact in retain_facts:
        groups.setdefault(fact.relation or "__unknown_relation__", []).append(fact)
    fit: list[Fact] = []
    validation: list[Fact] = []
    test: list[Fact] = []
    for relation in sorted(groups):
        group = sorted(groups[relation], key=lambda item: item.id)
        count = len(group)
        if count >= 3:
            fit_count = 1 if count == 3 else max(2, math.ceil(count * 0.5))
            remaining = count - fit_count
            validation_count = max(1, math.ceil(remaining * 0.5))
            fit.extend(expand_fact_variants(group[:fit_count], variants))
            validation.extend(
                expand_fact_variants(group[fit_count:fit_count + validation_count], variants)
            )
            test.extend(expand_fact_variants(group[fit_count + validation_count:], variants))
        else:
            expanded = sorted(dedupe(expand_fact_variants(group, variants)), key=lambda item: item.id)
            if len(expanded) < 3:
                raise ValueError(f"relation {relation!r} has fewer than three prompt variants")
            fit_count = max(1, len(expanded) - 2)
            fit.extend(expanded[:fit_count])
            validation.append(expanded[fit_count])
            test.extend(expanded[fit_count + 1:])
    if not fit or not validation or not test:
        raise ValueError("retain split contains an empty partition")
    return dedupe(fit), dedupe(validation), dedupe(test)


def split_forget_holdout(
    base_forget: list[Fact],
    fit_forget: list[Fact],
    validation_variants: list[str],
    test_variants: list[str],
    *,
    add_causal_probes: bool,
    reject_answer_leakage: bool,
) -> tuple[list[Fact], list[Fact]]:
    """Split held-out forget examples."""

    variants = list(dict.fromkeys([*validation_variants, *test_variants]))
    if not variants:
        raise ValueError("forget holdout variants cannot be empty")
    pool = without_overlap(
        dedupe(build_leakage_facts(base_forget, variants)),
        fit_forget,
    )
    if add_causal_probes:
        pool = dedupe([*pool, *build_nonleaking_causal_probes(base_forget)])
    
    
    
    pool = without_overlap(pool, fit_forget)
    if reject_answer_leakage:
        pool = [fact for fact in pool if not answer_leaks_into_prompt(fact)]
    groups: dict[str, list[Fact]] = defaultdict(list)
    for fact in pool:
        groups[family(fact)].append(fact)
    validation: list[Fact] = []
    test: list[Fact] = []
    for family_name in sorted(groups):
        group = sorted(groups[family_name], key=lambda item: item.id)
        if len(group) == 1:
            (validation if len(validation) <= len(test) else test).append(group[0])
            continue
        validation.extend(group[::2])
        test.extend(group[1::2])
    if not validation or not test:
        raise ValueError("forget holdout split contains an empty partition")
    return dedupe(validation), dedupe(test)


def build_splits(base_forget: list[Fact], base_retain: list[Fact], cfg: dict) -> dict[str, list[Fact]]:
    raw_forget = dedupe(build_leakage_facts(base_forget, list(cfg["fit_forget_variants"])))
    validation_forget, test_forget = split_forget_holdout(
        base_forget,
        raw_forget,
        list(cfg["validation_forget_variants"]),
        list(cfg["test_forget_variants"]),
        add_causal_probes=bool(cfg.get("add_causal_eval_probes", True)),
        reject_answer_leakage=bool(cfg.get("reject_answer_leakage", True)),
    )
    raw_retain, validation_retain, test_retain = split_retain(
        base_retain, list(cfg["retain_variants"])
    )
    prefixes = [str(value) for value in cfg["fit_prompt_prefixes"]]
    fit_forget = augment(raw_forget, prefixes)
    fit_retain = augment(raw_retain, prefixes)
    trace_variants = list(cfg.get("trace_forget_variants", ["factual", "reverse"]))
    trace_forget = dedupe(
        build_leakage_facts(
            base_forget,
            trace_variants,
        )
    )
    
    
    
    target_relations = {fact.relation for fact in base_forget if fact.relation}
    trace_all_retain = bool(cfg.get("trace_all_retain_relations", False))
    fit_retain_bases = [
        fact
        for fact in raw_retain
        if family(fact) == "factual"
        and (trace_all_retain or fact.relation in target_relations)
    ]
    trace_retain = dedupe(build_leakage_facts(fit_retain_bases, trace_variants))
    if not trace_retain:
        raise ValueError(
            f"no fit retain facts match target relations: {sorted(target_relations)}"
        )
    result = {
        "fit_forget": fit_forget,
        "fit_retain": fit_retain,
        "trace_forget": trace_forget,
        "trace_retain": trace_retain,
        "validation_forget": validation_forget,
        "validation_retain": validation_retain,
        "test_forget": test_forget,
        "test_retain": test_retain,
    }
    roles = {
        "fit": fit_forget + fit_retain,
        "validation": validation_forget + validation_retain,
        "test": test_forget + test_retain,
    }
    prompt_sets = {
        name: {prompt_key(fact.prompt) for fact in facts}
        for name, facts in roles.items()
    }
    for left, right in (("fit", "validation"), ("fit", "test"), ("validation", "test")):
        overlap = prompt_sets[left] & prompt_sets[right]
        if overlap:
            raise RuntimeError(f"data leakage between {left} and {right}: {len(overlap)}")
    return result


def rebalance_forget_splits_by_knowledge(
    splits: dict[str, list[Fact]],
    known: dict[tuple[str, str], bool],
    base_scores: dict[tuple[str, str], float],
    *,
    min_fit_per_direction: int,
    min_eval_per_direction: int,
    fail_on_unmet_direction_quotas: bool = False,
) -> dict[str, object]:
    """Balance forget splits across knowledge groups."""

    names = ("fit", "validation", "test")
    split_keys = {name: f"{name}_forget" for name in names}
    pool = dedupe([fact for name in names for fact in splits[split_keys[name]]])
    missing = [fact.id for fact in pool if (fact.prompt, fact.answer) not in known]
    if missing:
        raise KeyError(f"missing base-knowledge status for holdout prompts: {missing}")
    missing_scores = [
        fact.id for fact in pool if (fact.prompt, fact.answer) not in base_scores
    ]
    if missing_scores:
        raise KeyError(f"missing base answer scores for holdout prompts: {missing_scores}")

    original = {
        (fact.prompt, fact.answer): name
        for name in names
        for fact in splits[split_keys[name]]
    }
    traced_prompts = {prompt_key(fact.prompt) for fact in splits["trace_forget"]}
    traced_source_ids = {fact.id for fact in splits["trace_forget"]}
    pinned_keys = {
        (fact.prompt, fact.answer)
        for fact in pool
        if prompt_key(fact.prompt) in traced_prompts
        or fact.id.split(":fitprefix:", 1)[0] in traced_source_ids
    }
    requirements = {
        "fit": int(min_fit_per_direction),
        "validation": int(min_eval_per_direction),
        "test": int(min_eval_per_direction),
    }
    assigned: dict[tuple[str, str], str] = {key: "fit" for key in pinned_keys}

    availability: dict[str, dict[str, int | bool]] = {}
    for direction in ("forward", "reverse"):
        known_facts = [
            fact for fact in pool
            if causal_direction(fact) == direction
            and bool(known[(fact.prompt, fact.answer)])
        ]
        pinned_known = [
            fact for fact in known_facts if (fact.prompt, fact.answer) in pinned_keys
        ]
        needed_fit = max(requirements["fit"] - len(pinned_known), 0)
        required_total = (
            len(pinned_known) + needed_fit
            + requirements["validation"] + requirements["test"]
        )
        availability[direction] = {
            "known_total": len(known_facts),
            "pinned_fit_known": len(pinned_known),
            "required_total": required_total,
            "quota_possible": len(known_facts) >= required_total,
        }
        if fail_on_unmet_direction_quotas and len(known_facts) < required_total:
            raise RuntimeError(
                f"direction {direction!r} cannot satisfy formal split quotas: "
                f"known={len(known_facts)} required={required_total} "
                f"(pinned_fit={len(pinned_known)})"
            )

        available = [
            fact for fact in known_facts
            if (fact.prompt, fact.answer) not in assigned
        ]
        
        
        
        
        for destination in ("fit", "validation", "test"):
            current = sum(
                causal_direction(fact) == direction
                and bool(known[(fact.prompt, fact.answer)])
                and assigned.get((fact.prompt, fact.answer)) == destination
                for fact in pool
            )
            need = max(requirements[destination] - current, 0)
            ranked = sorted(
                available,
                key=lambda fact: (
                    original[(fact.prompt, fact.answer)] != destination,
                    -float(base_scores[(fact.prompt, fact.answer)]),
                    family(fact),
                    fact.id,
                ),
            )
            chosen = ranked[:need]
            for fact in chosen:
                assigned[(fact.prompt, fact.answer)] = destination
            chosen_keys = {(fact.prompt, fact.answer) for fact in chosen}
            available = [
                fact for fact in available
                if (fact.prompt, fact.answer) not in chosen_keys
            ]

    for fact in pool:
        key = (fact.prompt, fact.answer)
        assigned.setdefault(key, original[key])

    for name in names:
        splits[split_keys[name]] = dedupe(
            [fact for fact in pool if assigned[(fact.prompt, fact.answer)] == name]
        )
        if not splits[split_keys[name]]:
            raise RuntimeError(f"direction repair emptied {name}_forget")

    counts = {
        name: {
            direction: sum(
                causal_direction(fact) == direction
                and bool(known[(fact.prompt, fact.answer)])
                for fact in splits[split_keys[name]]
            )
            for direction in ("forward", "reverse")
        }
        for name in names
    }
    partition_evaluable = {
        name: all(value >= requirements[name] for value in counts[name].values())
        for name in names
    }
    all_evaluable = all(partition_evaluable.values())
    if fail_on_unmet_direction_quotas and not all_evaluable:
        raise RuntimeError(
            f"formal direction quota repair failed unexpectedly: {counts}"
        )
    return {
        "known_direction_counts": counts,
        "direction_availability": availability,
        "requirements": requirements,
        "partition_directionally_evaluable": partition_evaluable,
        "all_partitions_directionally_evaluable": all_evaluable,
        "formal_direction_quotas_required": True,
        "failed_on_unmet_direction_quotas": bool(fail_on_unmet_direction_quotas),
        "traced_prompts_pinned_to_fit": True,
    }


def manifest(splits: dict[str, list[Fact]]) -> pd.DataFrame:
    rows = []
    for name, facts in splits.items():
        if name.startswith("trace_"):
            continue
        for fact in facts:
            rows.append(
                {
                    "split": name,
                    "id": fact.id,
                    "base_id": base_id(fact),
                    "family": family(fact),
                    "type": fact.type,
                    "relation": fact.relation,
                    "prompt": fact.prompt,
                    "answer": fact.answer,
                }
            )
    return pd.DataFrame(rows)
