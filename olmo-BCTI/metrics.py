from __future__ import annotations

"""Metrics utilities."""

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Constraints:
    min_forget_relative_drop: float
    min_strong_forget_relative_drop: float
    max_forget_worst_score: float
    max_forget_increase: float
    min_eligible_forget_prompts: int
    min_eligible_forget_directions: int
    min_eligible_forget_prompts_per_direction: int
    min_direction_relative_drop: float
    min_strong_direction_relative_drop: float
    max_retain_kl_mean: float
    max_retain_kl_tail: float
    max_retain_kl_max: float
    max_retain_drop: float


def hard_tail_mean(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * float(fraction)))
    return float(sum(sorted(values, reverse=True)[:count]) / count)


def _relative_drop(base_mean: float, cut_mean: float) -> float:
    return (base_mean - cut_mean) / base_mean if base_mean > 0.0 else 0.0



def causal_threshold_streak(
    previous: int, score_value: float, threshold: float | None
) -> int:
    """Count consecutive causal-threshold passes."""

    if threshold is None or float(score_value) > float(threshold):
        return 0
    return int(previous) + 1


def causal_answer_gate_pass(
    *,
    causal_score: float,
    causal_threshold: float | None,
    min_direction_relative_drop: float,
    required_min_direction_relative_drop: float,
    worst_answer_score: float,
    maximum_worst_answer_score: float,
) -> bool:
    """Test whether causal and answer thresholds pass."""

    return bool(
        causal_threshold is not None
        and float(causal_score) <= float(causal_threshold)
        and float(min_direction_relative_drop)
        >= float(required_min_direction_relative_drop)
        and float(worst_answer_score) <= float(maximum_worst_answer_score)
    )


def causal_answer_gate_streak(
    previous: int,
    *,
    causal_score: float,
    causal_threshold: float | None,
    min_direction_relative_drop: float,
    required_min_direction_relative_drop: float,
    worst_answer_score: float,
    maximum_worst_answer_score: float,
) -> int:
    """Count consecutive causal-answer gate passes."""

    if not causal_answer_gate_pass(
        causal_score=causal_score,
        causal_threshold=causal_threshold,
        min_direction_relative_drop=min_direction_relative_drop,
        required_min_direction_relative_drop=required_min_direction_relative_drop,
        worst_answer_score=worst_answer_score,
        maximum_worst_answer_score=maximum_worst_answer_score,
    ):
        return 0
    return int(previous) + 1


def causal_threshold_stop_reached(
    *, step: int, minimum_step: int, streak: int, patience: int,
    threshold: float | None,
) -> bool:
    """Test whether the causal early-stop streak is complete."""

    return bool(
        threshold is not None
        and int(step) >= int(minimum_step)
        and int(streak) >= int(patience)
    )


def causal_checkpoint_priority(row: dict) -> tuple[float, float, int]:
    """Rank checkpoints by causal and answer metrics."""

    return (
        float(row["causal_target_score"]),
        float(row["causal_direction_macro_score"]),
        int(row["step"]),
    )


def bottleneck_violation_tier(value: float, slack: float) -> float | int:
    """Classify the most severe constraint violation."""

    value = max(float(value), 0.0)
    slack = float(slack)
    if slack < 0.0:
        raise ValueError("checkpoint bottleneck slack must be non-negative")
    if slack == 0.0 or not math.isfinite(value):
        return value
    return int(math.ceil(value / slack - 1.0e-12))


def causal_gated_checkpoint_priority(
    row: dict,
    causal_threshold: float | None,
    checkpoint_bottleneck_slack: float = 0.0,
) -> tuple:
    """Rank checkpoints with causal-gate feasibility."""

    causal_key = causal_checkpoint_priority(row)
    causal_pass = bool(
        causal_threshold is not None
        and float(row["causal_target_score"]) <= float(causal_threshold)
    )
    if not causal_pass:
        return (1, *causal_key)
    forget_violation = float(row.get("checkpoint_forget_violation", float("inf")))
    retain_violation = float(row.get("checkpoint_retain_violation", float("inf")))
    bottleneck = max(forget_violation, retain_violation)
    return (
        0,
        not bool(row.get("checkpoint_evaluable", False)),
        not bool(row.get("checkpoint_feasible", False)),
        not bool(row.get("checkpoint_strong_feasible", False)),
        bottleneck_violation_tier(bottleneck, checkpoint_bottleneck_slack),
        -float(row.get("forget_min_direction_relative_drop", 0.0)),
        float(row.get("forget_worst_cut_score", float("inf"))),
        bottleneck,
        float(row.get("checkpoint_total_violation", float("inf"))),
        retain_violation,
        forget_violation,
        *causal_key,
    )


def summarize(frame: pd.DataFrame, tail_fraction: float) -> dict[str, float | int | bool]:
    forget = frame[frame["type"] == "forget"].copy()
    retain = frame[frame["type"] == "retain"].copy()
    if forget.empty or retain.empty:
        raise ValueError("evaluation requires both forget and retain rows")
    if "forget_eligible" not in forget:
        raise ValueError("evaluation frame is missing forget_eligible")

    eligible = forget[forget["forget_eligible"].astype(bool)]
    
    
    primary = eligible if not eligible.empty else forget
    base_mean = float(primary["base_answer_score"].mean())
    cut_mean = float(primary["cut_answer_score"].mean())
    cut_scores = primary["cut_answer_score"].astype(float).tolist()
    first_base_mean = float(primary["base_first_token_prob"].mean())
    first_cut_mean = float(primary["cut_first_token_prob"].mean())
    retain_kls = retain["retain_kl"].astype(float).tolist()
    direction_metrics: dict[str, float | int] = {}
    if "direction" in primary:
        direction_drops: list[float] = []
        direction_eligible_counts: list[int] = []
        for direction in ("forward", "reverse"):
            group = primary[primary["direction"] == direction]
            prefix = f"forget_{direction}"
            direction_metrics[f"{prefix}_eligible_count"] = int(len(group))
            if group.empty:
                direction_metrics[f"{prefix}_relative_drop"] = 0.0
                direction_metrics[f"{prefix}_cut_greedy_exact_rate"] = 0.0
                direction_metrics[f"{prefix}_worst_cut_score"] = 0.0
                continue
            group_base = float(group["base_answer_score"].mean())
            group_cut = float(group["cut_answer_score"].mean())
            group_drop = _relative_drop(group_base, group_cut)
            direction_drops.append(group_drop)
            direction_eligible_counts.append(int(len(group)))
            direction_metrics[f"{prefix}_relative_drop"] = group_drop
            direction_metrics[f"{prefix}_cut_greedy_exact_rate"] = float(
                group["cut_greedy_exact"].mean()
            )
            direction_metrics[f"{prefix}_worst_cut_score"] = float(
                group["cut_answer_score"].max()
            )
        direction_metrics["forget_direction_macro_relative_drop"] = (
            float(sum(direction_drops) / len(direction_drops))
            if direction_drops
            else 0.0
        )
        direction_metrics["forget_evaluable_direction_count"] = len(direction_drops)
        direction_metrics["forget_min_direction_eligible_count"] = min(
            direction_eligible_counts, default=0
        )
        direction_metrics["forget_min_direction_relative_drop"] = (
            float(min(direction_drops)) if direction_drops else 0.0
        )
    result = {
        "forget_total_count": int(len(forget)),
        "forget_eligible_count": int(len(eligible)),
        "forget_eligible_fraction": float(len(eligible) / len(forget)),
        "forget_evaluable": bool(not eligible.empty),
        "forget_base_mean": base_mean,
        "forget_cut_mean": cut_mean,
        "forget_relative_drop": _relative_drop(base_mean, cut_mean),
        "forget_worst_cut_score": max(cut_scores),
        "forget_tail_cut_score": hard_tail_mean(cut_scores, tail_fraction),
        "forget_max_score_increase": float(
            (primary["cut_answer_score"] - primary["base_answer_score"]).max()
        ),
        "forget_base_first_token_mean": first_base_mean,
        "forget_cut_first_token_mean": first_cut_mean,
        "forget_first_token_relative_drop": _relative_drop(first_base_mean, first_cut_mean),
        "forget_worst_cut_first_token_prob": float(primary["cut_first_token_prob"].max()),
        "forget_base_token_accuracy": float(primary["base_answer_token_accuracy"].mean()),
        "forget_cut_token_accuracy": float(primary["cut_answer_token_accuracy"].mean()),
        "forget_base_greedy_exact_rate": float(primary["base_greedy_exact"].mean()),
        "forget_cut_greedy_exact_rate": float(primary["cut_greedy_exact"].mean()),
        "retain_kl_mean": float(retain["retain_kl"].mean()),
        "retain_kl_tail": hard_tail_mean(retain_kls, tail_fraction),
        "retain_kl_max": max(retain_kls),
        "retain_drop_mean": float(retain["answer_score_drop"].mean()),
        "retain_max_drop": float(retain["answer_score_drop"].max()),
        "retain_base_token_accuracy": float(retain["base_answer_token_accuracy"].mean()),
        "retain_cut_token_accuracy": float(retain["cut_answer_token_accuracy"].mean()),
        "retain_base_greedy_exact_rate": float(retain["base_greedy_exact"].mean()),
        "retain_cut_greedy_exact_rate": float(retain["cut_greedy_exact"].mean()),
    }
    result.update(direction_metrics)
    return result


def _normalized_gap(value: float, limit: float) -> float:
    return max(float(value) - float(limit), 0.0) / max(float(limit), 1e-12)


def score(
    summary: dict[str, float | int | bool], limits: Constraints
) -> dict[str, float | bool]:
    eligible_count = int(summary["forget_eligible_count"])
    eligibility_gap = max(limits.min_eligible_forget_prompts - eligible_count, 0) / max(
        limits.min_eligible_forget_prompts, 1
    )
    eligible_directions = int(summary.get("forget_evaluable_direction_count", 1))
    direction_gap = max(
        limits.min_eligible_forget_directions - eligible_directions, 0
    ) / max(limits.min_eligible_forget_directions, 1)
    min_direction_count = int(
        summary.get("forget_min_direction_eligible_count", 0)
    )
    direction_count_gap = max(
        limits.min_eligible_forget_prompts_per_direction - min_direction_count,
        0,
    ) / max(limits.min_eligible_forget_prompts_per_direction, 1)
    forget_gaps = [
        float(eligibility_gap),
        float(direction_gap),
        float(direction_count_gap),
        max(limits.min_forget_relative_drop - float(summary["forget_relative_drop"]), 0.0)
        / max(limits.min_forget_relative_drop, 1e-12),
        max(
            limits.min_direction_relative_drop
            - float(summary.get("forget_min_direction_relative_drop", 0.0)),
            0.0,
        )
        / max(limits.min_direction_relative_drop, 1e-12),
        _normalized_gap(float(summary["forget_worst_cut_score"]), limits.max_forget_worst_score),
        _normalized_gap(float(summary["forget_max_score_increase"]), limits.max_forget_increase),
        float(summary["forget_cut_greedy_exact_rate"] > 0.0),
    ]
    strong_forget_gaps = [
        float(eligibility_gap),
        float(direction_gap),
        float(direction_count_gap),
        max(
            limits.min_strong_forget_relative_drop
            - float(summary["forget_relative_drop"]),
            0.0,
        )
        / max(limits.min_strong_forget_relative_drop, 1e-12),
        max(
            limits.min_strong_direction_relative_drop
            - float(summary.get("forget_min_direction_relative_drop", 0.0)),
            0.0,
        )
        / max(limits.min_strong_direction_relative_drop, 1e-12),
        _normalized_gap(
            float(summary["forget_worst_cut_score"]), limits.max_forget_worst_score
        ),
        _normalized_gap(
            float(summary["forget_max_score_increase"]), limits.max_forget_increase
        ),
        float(summary["forget_cut_greedy_exact_rate"] > 0.0),
    ]
    retain_gaps = [
        _normalized_gap(float(summary["retain_kl_mean"]), limits.max_retain_kl_mean),
        _normalized_gap(float(summary["retain_kl_tail"]), limits.max_retain_kl_tail),
        _normalized_gap(float(summary["retain_kl_max"]), limits.max_retain_kl_max),
        _normalized_gap(float(summary["retain_max_drop"]), limits.max_retain_drop),
    ]
    forget_violation = float(sum(forget_gaps))
    strong_forget_violation = float(sum(strong_forget_gaps))
    retain_violation = float(sum(retain_gaps))
    evaluable = (
        bool(summary["forget_evaluable"])
        and eligibility_gap == 0.0
        and direction_gap == 0.0
        and direction_count_gap == 0.0
    )
    forget_acceptable = evaluable and forget_violation == 0.0
    strong_forget_acceptable = evaluable and strong_forget_violation == 0.0
    retain_acceptable = retain_violation == 0.0
    return {
        "evaluable": evaluable,
        "direction_count_gap": float(direction_count_gap),
        "direction_coverage_acceptable": bool(direction_count_gap == 0.0),
        "forget_acceptable": forget_acceptable,
        "strong_forget_acceptable": strong_forget_acceptable,
        "retain_acceptable": retain_acceptable,
        "feasible": forget_acceptable and retain_acceptable,
        "strong_feasible": strong_forget_acceptable and retain_acceptable,
        "forget_violation": forget_violation,
        "strong_forget_violation": strong_forget_violation,
        "retain_violation": retain_violation,
        "total_violation": forget_violation + retain_violation,
    }


def validation_priority(row: dict) -> tuple:
    """Build the default checkpoint-ordering tuple."""

    selection_safe = bool(
        row.get("selection_retain_acceptable", row["retain_acceptable"])
    )
    formal_safe = bool(row["retain_acceptable"])
    if bool(row["feasible"]) and selection_safe:
        tier = 0  
    elif bool(row["strong_feasible"]) and selection_safe:
        tier = 1  
    elif bool(row["feasible"]):
        tier = 2  
    elif bool(row["strong_feasible"]):
        tier = 3  
    elif formal_safe:
        tier = 4  
    else:
        tier = 5
    return (
        not bool(row["evaluable"]),
        tier,
        -float(row.get("forget_min_direction_relative_drop", 0.0)),
        -float(row.get("forget_direction_macro_relative_drop", 0.0)),
        float(row["strong_forget_violation"]),
        float(row.get("selection_retain_margin_gap", 0.0)),
        float(row["total_violation"]),
        float(row["retain_violation"]),
        float(row["forget_violation"]),
        float(row["forget_cut_mean"]),
        int(row["step"]),
    )


def safe_validation_priority(row: dict) -> tuple:
    """Rank checkpoints while prioritizing retain safety."""

    return (
        not bool(row["evaluable"]),
        not bool(row["retain_acceptable"]),
        -float(row.get("forget_min_direction_relative_drop", 0.0)),
        -float(row.get("forget_direction_macro_relative_drop", 0.0)),
        float(row["strong_forget_violation"]),
        float(row.get("selection_retain_margin_gap", 0.0)),
        float(row["retain_violation"]),
        float(row["forget_violation"]),
        float(row["forget_cut_mean"]),
        int(row["step"]),
    )


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    """Return nondominated rows for the configured objectives."""

    mask = []
    for index, row in frame.iterrows():
        dominated = (
            (frame["forget_violation"] <= row["forget_violation"])
            & (frame["retain_violation"] <= row["retain_violation"])
            & (
                (frame["forget_violation"] < row["forget_violation"])
                | (frame["retain_violation"] < row["retain_violation"])
            )
        ).any()
        mask.append(not bool(dominated))
    return pd.Series(mask, index=frame.index, dtype=bool)
