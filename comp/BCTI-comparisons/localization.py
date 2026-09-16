from __future__ import annotations

"""Localization utilities."""

import gc

import pandas as pd
import torch

from causal_tracing import run_tracing_multi
from tl_pythia_loader import load_from_config

def normalized_restore_score(
    clean_prob, corrupt_prob, patched_prob, epsilon: float
):
    """Normalize restoration by the clean-to-corrupt gap."""

    if isinstance(clean_prob, torch.Tensor) or isinstance(corrupt_prob, torch.Tensor) or isinstance(patched_prob, torch.Tensor):
        reference = next(
            value
            for value in (clean_prob, corrupt_prob, patched_prob)
            if isinstance(value, torch.Tensor)
        )
        clean = torch.as_tensor(clean_prob, device=reference.device, dtype=reference.dtype)
        corrupt = torch.as_tensor(corrupt_prob, device=reference.device, dtype=reference.dtype)
        patched = torch.as_tensor(patched_prob, device=reference.device, dtype=reference.dtype)
        gap = clean - corrupt
        raw = (patched - corrupt) / gap.clamp_min(float(epsilon))
        return torch.where(gap > float(epsilon), raw.clamp(0.0, 1.0), torch.zeros_like(raw))
    gap = float(clean_prob) - float(corrupt_prob)
    if gap <= float(epsilon):
        return 0.0
    return float(min(max((float(patched_prob) - float(corrupt_prob)) / gap, 0.0), 1.0))

def robust_restore_score(
    values: pd.Series, *, quantile: float, causal_threshold: float
) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().clip(0.0, 1.0)
    if values.empty:
        return 0.0
    center = 0.5 * float(values.mean()) + 0.5 * float(values.quantile(quantile))
    coverage = float((values >= causal_threshold).mean())
    return float(center * (0.5 + 0.5 * coverage))

def restore_coverage(values: pd.Series, *, causal_threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().clip(0.0, 1.0)
    if values.empty:
        return 0.0
    return float((values >= causal_threshold).mean())

def harmonic_pair(left: float, right: float, epsilon: float = 1.0e-12) -> float:
    left = max(float(left), 0.0)
    right = max(float(right), 0.0)
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return float((2.0 * left * right) / (left + right + float(epsilon)))

def report_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(
        f"[BCTI-main] cuda memory {label}: "
        f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB",
        flush=True,
    )

def select_causal_layers(
    layer_summary: pd.DataFrame, cfg: dict
) -> tuple[list[int], pd.DataFrame]:
    """Select the highest-scoring causal layers."""

    required = {
        "layer",
        "target_causal_score",
        "forward_target_causal_score",
        "reverse_target_causal_score",
        "forward_target_causal_coverage",
        "reverse_target_causal_coverage",
        "retain_risk",
    }
    missing = required - set(layer_summary.columns)
    if missing:
        raise ValueError(f"layer summary is missing causal selection fields: {sorted(missing)}")
    ranked = layer_summary.copy()
    for column in required:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.dropna(subset=list(required)).copy()
    ranked["layer"] = ranked["layer"].astype(int)
    ranked["global_target_rank"] = ranked["target_causal_score"].rank(
        method="first", ascending=False
    ).astype(int)
    candidate_pool_size = int(cfg["candidate_pool_size"])
    direction_pool_size = int(cfg["candidate_layers_per_direction"])
    candidate_layers = set(
        ranked.nsmallest(candidate_pool_size, "global_target_rank")["layer"].tolist()
    )
    for direction in ("forward", "reverse"):
        column = f"{direction}_target_causal_score"
        ranked[f"{direction}_target_rank"] = ranked[column].rank(
            method="first", ascending=False
        ).astype(int)
        candidate_layers.update(
            ranked.nsmallest(direction_pool_size, f"{direction}_target_rank")[
                "layer"
            ].tolist()
        )
    ranked["target_causal_candidate"] = ranked["layer"].isin(candidate_layers)
    best_target_score = float(ranked["target_causal_score"].max())
    relative_floor = float(cfg.get("candidate_relative_floor", 0.0))
    if not 0.0 <= relative_floor <= 1.0:
        raise ValueError("localization.candidate_relative_floor must be in [0, 1]")
    ranked["target_causal_floor"] = best_target_score * relative_floor
    ranked["target_causal_floor_pass"] = (
        ranked["target_causal_score"] >= ranked["target_causal_floor"]
    )
    eligible = ranked[
        ranked["target_causal_candidate"] & ranked["target_causal_floor_pass"]
    ].copy()
    ranked["target_causal_floor_relaxed"] = False
    if len(eligible) < int(cfg["layer_count"]):
        eligible = ranked[ranked["target_causal_candidate"]].copy()
        ranked.loc[ranked["target_causal_candidate"], "target_causal_floor_relaxed"] = True
    if eligible.empty:
        raise RuntimeError("target-only causal candidate pool is empty")

    layer_count = int(cfg["layer_count"])
    per_direction = int(cfg.get("min_layers_per_direction", 0))
    selected: list[int] = []
    selection_reason: dict[int, str] = {}
    for direction in ("forward", "reverse"):
        column = f"{direction}_target_causal_score"
        direction_rank = f"{direction}_target_rank"
        coverage_column = f"{direction}_target_causal_coverage"
        minimum_coverage = float(cfg.get("min_directional_causal_coverage", 0.0))
        pool = eligible[
            (eligible[direction_rank] <= direction_pool_size)
            & (eligible[coverage_column] >= minimum_coverage)
        ].sort_values(
            [coverage_column, column, "target_causal_score", "retain_risk", "layer"],
            ascending=[False, False, False, True, True],
        )
        if len(pool) < per_direction:
            pool = ranked[
                ranked["target_causal_candidate"]
                & (ranked[direction_rank] <= direction_pool_size)
                & (ranked[coverage_column] >= minimum_coverage)
            ].copy()
            ranked.loc[pool.index, "target_causal_floor_relaxed"] = True
            pool = pool.sort_values(
                [coverage_column, column, "target_causal_score", "retain_risk", "layer"],
                ascending=[False, False, False, True, True],
            )
        added = 0
        for value in pool["layer"].tolist():
            layer = int(value)
            if layer not in selected:
                selected.append(layer)
                selection_reason[layer] = f"{direction}_quota"
                added += 1
            if added >= per_direction:
                break
        if added < per_direction:
            raise RuntimeError(
                f"target-causal {direction} pool cannot fill its distinct layer quota"
            )
    causal_ranked = eligible.sort_values(
        ["target_causal_score", "retain_risk", "layer"],
        ascending=[False, True, True],
    )
    for value in causal_ranked["layer"].tolist():
        layer = int(value)
        if layer not in selected:
            selected.append(layer)
            selection_reason[layer] = "causal_pool_fill"
        if len(selected) >= layer_count:
            break
    selected = selected[:layer_count]
    if not selected:
        raise RuntimeError("causal layer selection is empty")
    ranked["selected"] = ranked["layer"].isin(selected)
    ranked["selection_order"] = ranked["layer"].map(
        {layer: index + 1 for index, layer in enumerate(selected)}
    )
    ranked["selection_reason"] = ranked["layer"].map(selection_reason).fillna("")
    ranked["directional_coverage_gate"] = float(
        cfg.get("min_directional_causal_coverage", 0.0)
    )
    ranked["forward_directional_coverage_pass"] = (
        ranked["forward_target_causal_coverage"]
        >= ranked["directional_coverage_gate"]
    )
    ranked["reverse_directional_coverage_pass"] = (
        ranked["reverse_target_causal_coverage"]
        >= ranked["directional_coverage_gate"]
    )
    ranked = ranked.sort_values(
        ["target_causal_score", "retain_risk"],
        ascending=[False, True],
        ignore_index=True,
    )
    return selected, ranked

def discover_causal_layers(
    base_cfg: dict,
    splits: dict[str, list[Fact]],
    cfg: dict,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank layers by causal restoration evidence."""

    print("[BCTI-main] loading TransformerLens model for fit-only tracing", flush=True)
    model = load_from_config(base_cfg)
    trace_facts = splits["trace_forget"] + splits["trace_retain"]
    trace = run_tracing_multi(
        model,
        trace_facts,
        sites=[str(value) for value in cfg["sites"]],
        corrupt_strategies=[str(value) for value in cfg["corrupt_strategies"]],
    )
    if trace.empty:
        raise RuntimeError("fit-only causal tracing returned no rows")
    for column in ("clean_prob", "corrupt_prob", "restore_effect"):
        trace[column] = pd.to_numeric(trace[column], errors="coerce")
    trace = trace.dropna(
        subset=["layer", "site", "type", "clean_prob", "corrupt_prob", "restore_effect"]
    ).copy()
    epsilon = float(cfg["gap_epsilon"])
    trace["corruption_gap"] = trace["clean_prob"] - trace["corrupt_prob"]
    trace["patched_prob"] = trace["corrupt_prob"] + trace["restore_effect"]
    trace["normalized_restore"] = [
        normalized_restore_score(clean, corrupt, patched, epsilon)
        for clean, corrupt, patched in zip(
            trace["clean_prob"], trace["corrupt_prob"], trace["patched_prob"]
        )
    ]

    quantile = float(cfg["robust_quantile"])
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("localization.robust_quantile must be in [0, 1]")
    causal_threshold = float(cfg.get("causal_effect_threshold", 0.20))
    if not 0.0 <= causal_threshold <= 1.0:
        raise ValueError("localization.causal_effect_threshold must be in [0, 1]")
    trace["direction"] = trace["id"].astype(str).map(
        lambda value: "reverse" if ":reverse:" in value else "forward"
    )

    site_rows: list[dict] = []
    for (layer, site), group in trace.groupby(["layer", "site"], sort=False):
        forget = group.loc[group["type"] == "forget", "normalized_restore"]
        retain = group.loc[group["type"] == "retain", "normalized_restore"]
        if forget.empty or retain.empty:
            raise RuntimeError("localization requires both forget and matched-retain trace rows")
        forget_score = robust_restore_score(
            forget, quantile=quantile, causal_threshold=causal_threshold
        )
        forget_peak = float(pd.to_numeric(forget, errors="coerce").dropna().max())
        forget_coverage = restore_coverage(
            forget, causal_threshold=causal_threshold
        )
        retain_score = max(
            robust_restore_score(
                retain, quantile=quantile, causal_threshold=causal_threshold
            ),
            float(pd.to_numeric(retain, errors="coerce").dropna().quantile(quantile)),
        )
        direction_scores: dict[str, float] = {}
        direction_coverage: dict[str, float] = {}
        for direction in ("forward", "reverse"):
            directional = group[group["direction"] == direction]
            d_forget = directional.loc[
                directional["type"] == "forget", "normalized_restore"
            ]
            if d_forget.empty:
                direction_scores[direction] = 0.0
                direction_coverage[direction] = 0.0
                continue
            direction_scores[direction] = robust_restore_score(
                d_forget, quantile=quantile, causal_threshold=causal_threshold
            )
            direction_coverage[direction] = restore_coverage(
                d_forget, causal_threshold=causal_threshold
            )
        bidirectional_score = harmonic_pair(
            direction_scores["forward"], direction_scores["reverse"]
        )
        site_rows.append(
            {
                "layer": int(layer),
                "site": str(site),
                "forget_restore": forget_score,
                "forget_restore_peak": forget_peak,
                "target_causal_coverage": forget_coverage,
                "forward_target_causal_coverage": direction_coverage["forward"],
                "reverse_target_causal_coverage": direction_coverage["reverse"],
                "retain_restore": retain_score,
                "target_causal_score": bidirectional_score,
                "target_causal_bidirectional_score": bidirectional_score,
                "retain_risk": retain_score,
                
                "causal_score": bidirectional_score,
                "forward_causal_score": direction_scores["forward"],
                "reverse_causal_score": direction_scores["reverse"],
                "forward_target_causal_score": direction_scores["forward"],
                "reverse_target_causal_score": direction_scores["reverse"],
            }
        )
    site_summary = pd.DataFrame(site_rows).sort_values(
        ["causal_score", "forget_restore"], ascending=[False, False], ignore_index=True
    )
    layer_summary = (
        site_summary.groupby("layer", as_index=False)
        .agg(
            causal_score=("causal_score", "mean"),
            mean_causal_score=("causal_score", "mean"),
            target_causal_score=("target_causal_score", "mean"),
            mean_target_causal_score=("target_causal_score", "mean"),
            target_causal_bidirectional_score=("target_causal_bidirectional_score", "mean"),
            target_causal_coverage=("target_causal_coverage", "mean"),
            forward_target_causal_coverage=("forward_target_causal_coverage", "mean"),
            reverse_target_causal_coverage=("reverse_target_causal_coverage", "mean"),
            forget_restore=("forget_restore", "mean"),
            forget_restore_peak=("forget_restore_peak", "max"),
            retain_restore=("retain_restore", "max"),
            retain_risk=("retain_risk", "max"),
            forward_causal_score=("forward_causal_score", "mean"),
            reverse_causal_score=("reverse_causal_score", "mean"),
            forward_target_causal_score=("forward_target_causal_score", "mean"),
            reverse_target_causal_score=("reverse_target_causal_score", "mean"),
        )
        .sort_values(
            ["causal_score", "forget_restore", "mean_causal_score"],
            ascending=[False, False, False],
            ignore_index=True,
        )
    )
    layer_summary["target_causal_score"] = [
        harmonic_pair(row.forward_target_causal_score, row.reverse_target_causal_score)
        for row in layer_summary.itertuples(index=False)
    ]
    layer_summary["causal_score"] = layer_summary["target_causal_score"]
    layer_summary["target_causal_bidirectional_score"] = layer_summary[
        "target_causal_score"
    ]
    layer_summary = layer_summary.sort_values(
        ["target_causal_score", "forget_restore", "mean_target_causal_score"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    layers, layer_summary = select_causal_layers(layer_summary, cfg)
    report_cuda_memory("after causal tracing")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    report_cuda_memory("after releasing TransformerLens")
    return layers, trace, site_summary, layer_summary
