from __future__ import annotations

"""Ablation utilities."""

import pandas as pd


def select_ablation_layers(
    layer_summary: pd.DataFrame, cfg: dict, policy: str
) -> tuple[list[int], pd.DataFrame]:
    """Select layers enabled by the requested ablation."""
    ranked = layer_summary.copy()
    count = int(cfg["layer_count"])
    if policy == "bidirectional_causal":
        selected = ranked.loc[ranked["selected"].astype(bool), "layer"]
        layers = [int(value) for value in selected.tolist()]
        ranked["ablation_localization_policy"] = policy
        return layers, ranked

    score_column = {
        "forward_causal": "forward_target_causal_score",
        "reverse_causal": "reverse_target_causal_score",
    }.get(policy)
    if score_column is None:
        raise ValueError(f"unknown localization policy: {policy}")
    if score_column not in ranked:
        raise ValueError(f"layer summary is missing {score_column}")

    ordered = ranked.sort_values(
        [score_column, "retain_risk", "layer"],
        ascending=[False, True, True],
    )
    layers = [int(value) for value in ordered.head(count)["layer"].tolist()]
    rank_map = {
        int(layer): index + 1 for index, layer in enumerate(ordered["layer"].tolist())
    }
    ranked["ablation_localization_rank"] = ranked["layer"].astype(int).map(rank_map)
    ranked["selected"] = ranked["layer"].astype(int).isin(layers)
    ranked["selection_order"] = ranked["layer"].astype(int).map(
        {layer: index + 1 for index, layer in enumerate(layers)}
    )
    ranked["selection_reason"] = ranked["selected"].map(
        {True: f"{policy}_topk", False: "not_selected_by_ablation"}
    )
    ranked["ablation_localization_policy"] = policy
    return layers, ranked
