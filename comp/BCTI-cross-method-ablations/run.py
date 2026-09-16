from __future__ import annotations

"""Run utilities."""

import argparse
import gc
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
COMMON_ROOT = CODE_ROOT / "common"
for path in (COMMON_ROOT, CODE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from causal_tracing import run_tracing_multi  # noqa: E402
from config_utils import load_config  # noqa: E402
from data_utils import Fact, corrupt_prompt, load_facts, split_facts  # noqa: E402
from tl_pythia_loader import dtype_from_name, load_from_config  # noqa: E402

from palu_loss import palu_local_entropy_loss  # noqa: E402
from rome_localization import RomeTraceConfig, discover_rome_layers  # noqa: E402

from lora import (  # noqa: E402
    install_causal_lora,
    restore_adapters,
    snapshot_adapters,
    trainable_named_parameters,
)
from metrics import (  # noqa: E402
    Constraints,
    causal_answer_gate_pass,
    causal_answer_gate_streak,
    causal_gated_checkpoint_priority,
    causal_threshold_stop_reached,
    causal_threshold_streak,
    score,
    summarize,
)
from splits import (  # noqa: E402
    base_id,
    build_splits,
    causal_direction,
    family,
    manifest,
    rebalance_forget_splits_by_knowledge,
)


MODES = (
    "forget_only", "joint", "pcgrad", "retain_null", "sago",
    "retain_null_pcgrad", "retain_null_sago",
)
METHOD_VERSION = "BCTI-cross-methods-v1"


@dataclass(frozen=True)
class CausalInterventionSpec:
    """Describe a causal intervention."""

    layer: int
    site: str
    corruption: str

    @property
    def label(self) -> str:
        return f"layer={self.layer}:{self.site}:{self.corruption}"


class CausalInterventionStream:
    """Stream batches with causal interventions."""

    def __init__(self, specs: list[CausalInterventionSpec], seed: int) -> None:
        if not specs:
            raise ValueError("causal intervention stream requires at least one site")
        self.specs = list(specs)
        self.random = random.Random(int(seed))
        self.order: list[int] = []
        self.cursor = 0

    def next(self) -> CausalInterventionSpec:
        if self.cursor >= len(self.order):
            self.order = list(range(len(self.specs)))
            self.random.shuffle(self.order)
            self.cursor = 0
        value = self.specs[self.order[self.cursor]]
        self.cursor += 1
        return value


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


def causal_restore_training_score(
    clean_prob: torch.Tensor,
    corrupt_prob: torch.Tensor,
    patched_prob: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute the restoration training score."""

    clean = clean_prob.detach()
    corrupt = corrupt_prob.detach()
    gap = clean - corrupt
    raw = (patched_prob - corrupt) / gap.clamp_min(float(epsilon))
    clipped = raw.clamp(0.0, 1.0)
    
    
    
    upper_straight_through = raw + (clipped - raw).detach()
    active = torch.where(raw >= 0.0, upper_straight_through, torch.zeros_like(raw))
    return torch.where(gap > float(epsilon), active, torch.zeros_like(active))


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
        f"[cross-methods] cuda memory {label}: "
        f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB",
        flush=True,
    )


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def validate_config(cfg: dict, mode: str) -> None:
    for section in (
        "pipeline_config",
        "data_split",
        "localization",
        "causal_training",
        "lora",
        "evaluation",
        "training",
    ):
        if section not in cfg:
            raise ValueError(f"config is missing {section!r}")
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    localization = cfg["localization"]
    if int(localization["layer_count"]) <= 0:
        raise ValueError("localization.layer_count must be positive")
    if not list(localization["sites"]):
        raise ValueError("localization.sites cannot be empty")
    if not list(localization["corrupt_strategies"]):
        raise ValueError("localization.corrupt_strategies cannot be empty")
    if float(localization["gap_epsilon"]) <= 0.0:
        raise ValueError("localization.gap_epsilon must be positive")
    if int(localization["candidate_pool_size"]) < int(localization["layer_count"]):
        raise ValueError("candidate_pool_size must be at least layer_count")
    if int(localization["candidate_layers_per_direction"]) < 2 * int(
        localization.get("min_layers_per_direction", 0)
    ):
        raise ValueError(
            "candidate_layers_per_direction must be at least twice "
            "min_layers_per_direction so forward/reverse quotas can be distinct"
        )
    per_direction = int(localization.get("min_layers_per_direction", 0))
    if per_direction < 0 or 2 * per_direction > int(localization["layer_count"]):
        raise ValueError(
            "localization.min_layers_per_direction must be non-negative and "
            "at most half of layer_count"
        )
    lora_cfg = cfg["lora"]
    if int(lora_cfg["rank"]) <= 0 or float(lora_cfg["alpha"]) <= 0.0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not list(lora_cfg["target_modules"]):
        raise ValueError("lora.target_modules cannot be empty")
    training = cfg["training"]
    if int(training["max_steps"]) <= 0:
        raise ValueError("training.max_steps must be positive")
    if int(training["eval_every"]) <= 0:
        raise ValueError("training.eval_every must be positive")
    if int(training.get("early_dense_eval_steps", 0)) < 0:
        raise ValueError("training.early_dense_eval_steps cannot be negative")
    if int(training["early_stop_min_steps"]) <= 0:
        raise ValueError("early_stop_min_steps must be positive")
    if int(training["no_improvement_patience_evals"]) <= 0:
        raise ValueError("no_improvement_patience_evals must be positive")
    causal_stop_threshold = training.get("causal_early_stop_threshold")
    if causal_stop_threshold is not None and float(causal_stop_threshold) <= 0.0:
        raise ValueError("causal_early_stop_threshold must be positive or null")
    if int(training.get("causal_early_stop_patience_evals", 1)) <= 0:
        raise ValueError("causal_early_stop_patience_evals must be positive")
    answer_gate_min_drop = float(
        training["causal_gate_min_direction_relative_drop"]
    )
    if not 0.0 <= answer_gate_min_drop <= 1.0:
        raise ValueError(
            "causal_gate_min_direction_relative_drop must be in [0, 1]"
        )
    answer_gate_worst_score = float(
        training["causal_gate_max_worst_answer_score"]
    )
    if not 0.0 <= answer_gate_worst_score <= 1.0:
        raise ValueError(
            "causal_gate_max_worst_answer_score must be in [0, 1]"
        )
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if float(training["retain_weight"]) < 0.0:
        raise ValueError("training.retain_weight cannot be negative")
    if float(training["max_grad_norm"]) <= 0.0:
        raise ValueError("training.max_grad_norm must be positive")
    if int(training["forget_batch_size"]) <= 0 or int(training["retain_batch_size"]) <= 0:
        raise ValueError("training batch sizes must be positive")
    if str(training["gradient_normalization"]) not in {"none", "match_forget"}:
        raise ValueError("gradient_normalization must be 'none' or 'match_forget'")
    evaluation = cfg["evaluation"]
    if not 0.0 <= float(evaluation["min_forget_base_answer_score"]) <= 1.0:
        raise ValueError("min_forget_base_answer_score must be in [0, 1]")
    if int(evaluation["constraints"]["min_eligible_forget_directions"]) <= 0:
        raise ValueError("min_eligible_forget_directions must be positive")
    if int(
        evaluation["constraints"]["min_eligible_forget_prompts_per_direction"]
    ) <= 0:
        raise ValueError("min_eligible_forget_prompts_per_direction must be positive")
    for key in ("min_direction_relative_drop", "min_strong_direction_relative_drop"):
        if not 0.0 <= float(evaluation["constraints"][key]) <= 1.0:
            raise ValueError(f"evaluation.constraints.{key} must be in [0, 1]")
    data_split = cfg["data_split"]
    if int(data_split["min_known_fit_per_direction"]) <= 0:
        raise ValueError("min_known_fit_per_direction must be positive")
    if int(data_split["min_known_eval_per_direction"]) <= 0:
        raise ValueError("min_known_eval_per_direction must be positive")
    causal_training = cfg["causal_training"]
    if not 0.0 < float(causal_training["layer_weight_floor"]) <= 1.0:
        raise ValueError("causal layer_weight_floor must be in (0, 1]")
    if float(causal_training["layer_weight_power"]) <= 0.0:
        raise ValueError("causal layer_weight_power must be positive")
    ablation = cfg.get("ablation", {})
    localization_policy = str(ablation.get("localization_policy", "bcti"))
    forget_objective = str(ablation.get("forget_objective", "bcti"))
    if localization_policy not in {"bcti", "rome"}:
        raise ValueError("ablation.localization_policy must be bcti or rome")
    if forget_objective not in {"bcti", "palu"}:
        raise ValueError("ablation.forget_objective must be bcti or palu")
    if localization_policy == "rome":
        rome = cfg.get("rome", {})
        if float(rome.get("noise_multiplier", 0.0)) <= 0.0:
            raise ValueError("rome.noise_multiplier must be positive")
        if int(rome.get("noise_samples", 0)) <= 0:
            raise ValueError("rome.noise_samples must be positive")
        if int(rome.get("layer_count", 0)) != int(localization["layer_count"]):
            raise ValueError("rome.layer_count must match localization.layer_count")
    if forget_objective == "palu":
        palu = cfg.get("palu", {})
        if int(palu.get("top_k", 0)) <= 0:
            raise ValueError("palu.top_k must be positive")
        if int(palu.get("initiating_tokens", 0)) <= 0:
            raise ValueError("palu.initiating_tokens must be positive")
        if float(causal_training["causal_score_weight"]) != 0.0:
            raise ValueError("PALU ablation requires causal_score_weight == 0")
        if float(causal_training["answer_complement_weight"]) != 0.0:
            raise ValueError("PALU ablation requires answer_complement_weight == 0")
    else:
        if float(causal_training["causal_score_weight"]) <= 0.0:
            raise ValueError("causal_training.causal_score_weight must be positive")
        auxiliary_kind = str(causal_training.get("auxiliary_forget_loss", ""))
        if auxiliary_kind != "bounded_ga_probability":
            raise ValueError("BCTI objective requires bounded_ga_probability")
        auxiliary_weight = float(causal_training["answer_complement_weight"])
        if not 0.0 < auxiliary_weight <= 1.0:
            raise ValueError("answer_complement_weight must be in (0, 1]")
        auxiliary_ratio = float(causal_training["max_answer_auxiliary_gradient_ratio"])
        if not 0.0 < auxiliary_ratio < 1.0:
            raise ValueError("max_answer_auxiliary_gradient_ratio must be in (0, 1)")
    safe_layer_fraction = float(causal_training["auxiliary_safe_layer_fraction"])
    if not 0.0 < safe_layer_fraction <= 1.0:
        raise ValueError("auxiliary_safe_layer_fraction must be in (0, 1]")
    global_layer_floor_ratio = float(
        causal_training["auxiliary_global_layer_floor_ratio"]
    )
    auxiliary_ratio = float(causal_training["max_answer_auxiliary_gradient_ratio"])
    if forget_objective == "bcti" and not 0.0 < global_layer_floor_ratio <= auxiliary_ratio:
        raise ValueError(
            "auxiliary_global_layer_floor_ratio must be in "
            "(0, max_answer_auxiliary_gradient_ratio]"
        )
    answer_cap_floor = float(causal_training["answer_direction_min_cap_fraction"])
    if not 0.0 < answer_cap_floor <= 1.0:
        raise ValueError("answer_direction_min_cap_fraction must be in (0, 1]")
    answer_ema_decay = float(causal_training["answer_direction_ema_decay"])
    if not 0.0 <= answer_ema_decay < 1.0:
        raise ValueError("answer_direction_ema_decay must be in [0, 1)")
    answer_hardness_weight = float(causal_training["answer_hardness_weight"])
    if not 0.0 <= answer_hardness_weight <= 1.0:
        raise ValueError("answer_hardness_weight must be in [0, 1]")
    retain_answer_margin = float(causal_training["retain_answer_margin"])
    if not 0.0 <= retain_answer_margin <= 1.0:
        raise ValueError("retain_answer_margin must be in [0, 1]")
    if float(causal_training["retain_answer_margin_weight"]) < 0.0:
        raise ValueError("retain_answer_margin_weight must be non-negative")
    if float(training["checkpoint_bottleneck_slack"]) < 0.0:
        raise ValueError("training.checkpoint_bottleneck_slack must be non-negative")
    if "causal_gate_require_retain_acceptable" in training and not isinstance(
        training["causal_gate_require_retain_acceptable"], bool
    ):
        raise ValueError("training.causal_gate_require_retain_acceptable must be boolean")
    for key in ("direction_weight_min", "direction_weight_max"):
        if float(causal_training[key]) <= 0.0:
            raise ValueError(f"causal_training.{key} must be positive")
    configured_modes = [str(value) for value in cfg.get("run_modes", MODES)]
    unknown_modes = sorted(set(configured_modes) - set(MODES))
    if unknown_modes:
        raise ValueError(f"run_modes contains unsupported modes: {unknown_modes}")
    if not configured_modes:
        raise ValueError("run_modes cannot be empty")
    if float(causal_training["direction_weight_min"]) > float(
        causal_training["direction_weight_max"]
    ):
        raise ValueError("direction_weight_min cannot exceed direction_weight_max")
    if str(causal_training.get("direction_gradient_policy", "weighted_mean")) not in {"weighted_mean", "symmetric_pcgrad"}:
        raise ValueError("direction_gradient_policy must be weighted_mean or symmetric_pcgrad")
    retain_gradient_min_ratio = float(
        causal_training.get("retain_gradient_min_ratio", 0.0)
    )
    if not 0.0 <= retain_gradient_min_ratio <= 1.0:
        raise ValueError("retain_gradient_min_ratio must be in [0, 1]")
    if int(causal_training["hard_example_every"]) <= 0:
        raise ValueError("hard_example_every must be positive")
    if not 0.0 < float(causal_training["direction_calibration_fraction"]) < 0.5:
        raise ValueError("direction_calibration_fraction must be in (0, 0.5)")
    if int(causal_training["direction_calibration_every"]) <= 0:
        raise ValueError("direction_calibration_every must be positive")
    if int(causal_training["direction_calibration_max_per_direction"]) <= 0:
        raise ValueError("direction_calibration_max_per_direction must be positive")
    if int(causal_training["preferred_optimization_prompts_per_direction"]) <= 0:
        raise ValueError("preferred_optimization_prompts_per_direction must be positive")
    if not 0.0 <= float(causal_training["causal_neighbor_retain_fraction"]) <= 1.0:
        raise ValueError("causal_neighbor_retain_fraction must be in [0, 1]")
    if not 0.0 < float(causal_training["causal_neighbor_pool_fraction"]) <= 1.0:
        raise ValueError("causal_neighbor_pool_fraction must be in (0, 1]")
    if not 0.0 <= float(causal_training.get("hybrid_retain_risk_quantile", 0.5)) <= 1.0:
        raise ValueError("hybrid_retain_risk_quantile must be in [0, 1]")
    if not 0.0 <= float(localization.get("min_directional_causal_coverage", 0.0)) <= 1.0:
        raise ValueError("min_directional_causal_coverage must be in [0, 1]")

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

    print("[cross-methods] loading TransformerLens model for fit-only tracing", flush=True)
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


def load_causal_layers(
    source: Path,
    cfg: dict,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load selected causal layers."""

    paths = {
        "trace": source / "02_fit_causal_trace.csv",
        "site": source / "03_site_scores.csv",
        "layer": source / "04_layer_scores.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"shared localization is incomplete: {missing}")
    trace = pd.read_csv(paths["trace"])
    site_summary = pd.read_csv(paths["site"])
    layer_summary = pd.read_csv(paths["layer"])
    required = {
        "layer", "causal_score", "forget_restore", "mean_causal_score",
        "target_causal_score", "forward_target_causal_score",
        "reverse_target_causal_score", "retain_risk",
    }
    absent = required - set(layer_summary.columns)
    if absent:
        raise ValueError(f"shared layer scores are missing columns: {sorted(absent)}")
    layer_summary["layer"] = pd.to_numeric(layer_summary["layer"], errors="coerce")
    layer_summary["causal_score"] = pd.to_numeric(
        layer_summary["causal_score"], errors="coerce"
    )
    layer_summary["forget_restore"] = pd.to_numeric(
        layer_summary["forget_restore"], errors="coerce"
    )
    layer_summary["mean_causal_score"] = pd.to_numeric(
        layer_summary["mean_causal_score"], errors="coerce"
    )
    for column in (
        "forward_causal_score", "reverse_causal_score",
        "target_causal_score", "forward_target_causal_score",
        "reverse_target_causal_score", "retain_risk",
    ):
        if column not in layer_summary:
            raise ValueError(f"shared layer scores are missing column: {column}")
        layer_summary[column] = pd.to_numeric(layer_summary[column], errors="coerce")
    layer_summary = layer_summary.dropna(
        subset=[
            "layer",
            "causal_score",
            "forget_restore",
            "mean_causal_score",
            "forward_causal_score",
            "reverse_causal_score",
        ]
    ).sort_values(
        ["causal_score", "forget_restore", "mean_causal_score"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    layers, layer_summary = select_causal_layers(layer_summary, cfg)
    print(f"[cross-methods] reusing fit-only localization from {source}", flush=True)
    return layers, trace, site_summary, layer_summary


def load_hf(base_cfg: dict):
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
    device = str(model_cfg.get("device", "cuda"))
    model.to(device)
    model.eval()
    model.config.use_cache = False
    model_devices = {str(parameter.device) for parameter in model.parameters()}
    if model_devices != {str(torch.device(device))}:
        raise RuntimeError(f"HF model spans unexpected devices: {sorted(model_devices)}")
    report_cuda_memory("after loading HF model")
    return model, tokenizer, device


def token_ids(tokenizer, text: str, device: str) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if encoded.numel() == 0:
        raise ValueError(f"tokenization is empty: {text!r}")
    return encoded.to(device)


def answer_distributions(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute answer distributions."""

    prompt_ids = token_ids(tokenizer, prompt, device)
    answer_ids = token_ids(tokenizer, answer, device)
    combined = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = model(input_ids=combined).logits.float()
    start = prompt_ids.shape[1] - 1
    stop = start + answer_ids.shape[1]
    distributions = F.softmax(logits[:, start:stop, :], dim=-1)[0]
    return distributions, answer_ids[0]


def _hf_site_module(model, spec: CausalInterventionSpec):
    try:
        block = model.gpt_neox.layers[int(spec.layer)]
    except (AttributeError, IndexError) as error:
        raise ValueError(f"invalid GPT-NeoX causal layer: {spec.layer}") from error
    if spec.site == "resid_pre":
        return block, True
    if spec.site == "attn_out":
        return block.attention, False
    if spec.site == "mlp_out":
        return block.mlp, False
    raise ValueError(f"unsupported causal site: {spec.site!r}")


def _first_output_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"causal hook expected a tensor or tensor-first sequence, got {type(output)!r}")


def _replace_first_output_tensor(output, replacement: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return replacement
    values = list(output)
    values[0] = replacement
    return tuple(values) if isinstance(output, tuple) else values


def _replace_last_token(activation: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    if activation.ndim != 3:
        raise ValueError(f"causal activation must have shape [batch, tokens, hidden], got {tuple(activation.shape)}")
    value = replacement.to(device=activation.device, dtype=activation.dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape != activation[:, -1, :].shape:
        raise ValueError(
            f"causal replacement shape {tuple(value.shape)} does not match "
            f"last-token activation {tuple(activation[:, -1, :].shape)}"
        )
    patched = activation.clone()
    patched[:, -1, :] = value
    return patched


def _prompt_first_token_probability(
    model, tokenizer, prompt: str, answer: str, device: str
) -> torch.Tensor:
    prompt_ids = token_ids(tokenizer, prompt, device)
    target_id = token_ids(tokenizer, answer, device)[0, 0]
    logits = model(input_ids=prompt_ids).logits.float()
    return F.softmax(logits[:, -1, :], dim=-1)[0, target_id]


def causal_restore_components(
    model,
    tokenizer,
    fact: Fact,
    spec: CausalInterventionSpec,
    device: str,
    epsilon: float,
    *,
    fixed_clean_prob: float | None = None,
    fixed_corrupt_prob: float | None = None,
) -> dict[str, torch.Tensor | str]:
    """Compute restoration components."""

    module, is_pre_hook = _hf_site_module(model, spec)
    captured: dict[str, torch.Tensor] = {}

    if is_pre_hook:
        def capture_hook(_module, inputs):
            captured["activation"] = _first_output_tensor(inputs).detach()[:, -1, :].clone()
    else:
        def capture_hook(_module, _inputs, output):
            captured["activation"] = _first_output_tensor(output).detach()[:, -1, :].clone()

    handle = (
        module.register_forward_pre_hook(capture_hook)
        if is_pre_hook
        else module.register_forward_hook(capture_hook)
    )
    try:
        
        
        
        
        with torch.no_grad():
            clean_probability = _prompt_first_token_probability(
                model, tokenizer, fact.prompt, fact.answer, device
            )
    finally:
        handle.remove()
    if "activation" not in captured:
        raise RuntimeError(f"failed to capture causal activation at {spec.label}")

    corrupt = corrupt_prompt(fact, strategy=spec.corruption)
    with torch.no_grad():
        corrupt_probability = _prompt_first_token_probability(
            model, tokenizer, corrupt, fact.answer, device
        )
    replacement = captured["activation"]

    if is_pre_hook:
        def patch_hook(_module, inputs):
            values = list(inputs)
            values[0] = _replace_last_token(_first_output_tensor(inputs), replacement)
            return tuple(values)
    else:
        def patch_hook(_module, _inputs, output):
            activation = _first_output_tensor(output)
            return _replace_first_output_tensor(
                output, _replace_last_token(activation, replacement)
            )

    handle = (
        module.register_forward_pre_hook(patch_hook)
        if is_pre_hook
        else module.register_forward_hook(patch_hook)
    )
    try:
        patched_probability = _prompt_first_token_probability(
            model, tokenizer, corrupt, fact.answer, device
        )
    finally:
        handle.remove()

    
    
    
    
    score_clean = (
        clean_probability
        if fixed_clean_prob is None
        else torch.as_tensor(fixed_clean_prob, device=patched_probability.device, dtype=patched_probability.dtype)
    )
    score_corrupt = (
        corrupt_probability
        if fixed_corrupt_prob is None
        else torch.as_tensor(fixed_corrupt_prob, device=patched_probability.device, dtype=patched_probability.dtype)
    )
    score = (
        causal_restore_training_score(
            score_clean, score_corrupt, patched_probability, epsilon
        )
        if torch.is_grad_enabled()
        else normalized_restore_score(
            score_clean, score_corrupt, patched_probability, epsilon
        )
    )
    return {
        "score": score,
        "clean_prob": clean_probability,
        "corrupt_prob": corrupt_probability,
        "fixed_clean_prob": score_clean,
        "fixed_corrupt_prob": score_corrupt,
        "patched_prob": patched_probability,
        "corruption_gap": score_clean - score_corrupt,
        "corrupt_prompt": corrupt,
    }


def build_causal_intervention_specs(
    site_summary: pd.DataFrame,
    selected_layers: list[int],
    corrupt_strategies: list[str],
) -> list[CausalInterventionSpec]:
    selected = site_summary[
        pd.to_numeric(site_summary["layer"], errors="coerce").isin(selected_layers)
    ][["layer", "site"]].drop_duplicates()
    specs = [
        CausalInterventionSpec(int(row.layer), str(row.site), str(corruption))
        for row in selected.itertuples(index=False)
        for corruption in corrupt_strategies
    ]
    return sorted(specs, key=lambda value: (value.layer, value.site, value.corruption))


@torch.no_grad()
def causal_baseline_lookup(frame: pd.DataFrame) -> dict[tuple, tuple[float, float]]:
    if frame.empty:
        return {}
    return {
        (row.prompt, row.answer, int(row.layer), str(row.site), str(row.corruption)): (
            float(row.fixed_clean_prob if hasattr(row, "fixed_clean_prob") else row.clean_prob),
            float(row.fixed_corrupt_prob if hasattr(row, "fixed_corrupt_prob") else row.corrupt_prob),
        )
        for row in frame.itertuples(index=False)
    }


@torch.no_grad()
def evaluate_causal_objective(
    model,
    tokenizer,
    facts: list[Fact],
    specs: list[CausalInterventionSpec],
    device: str,
    epsilon: float,
    *,
    fixed_baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    baseline = causal_baseline_lookup(fixed_baseline) if fixed_baseline is not None else {}
    for fact in facts:
        for spec in specs:
            fixed = baseline.get((fact.prompt, fact.answer, spec.layer, spec.site, spec.corruption))
            values = causal_restore_components(
                model, tokenizer, fact, spec, device, epsilon,
                fixed_clean_prob=None if fixed is None else fixed[0],
                fixed_corrupt_prob=None if fixed is None else fixed[1],
            )
            rows.append(
                {
                    "id": fact.id,
                    "prompt": fact.prompt,
                    "answer": fact.answer,
                    "direction": causal_direction(fact),
                    "layer": spec.layer,
                    "site": spec.site,
                    "corruption": spec.corruption,
                    "clean_prob": float(values["clean_prob"].item()),
                    "corrupt_prob": float(values["corrupt_prob"].item()),
                    "fixed_clean_prob": float(values["fixed_clean_prob"].item()),
                    "fixed_corrupt_prob": float(values["fixed_corrupt_prob"].item()),
                    "patched_prob": float(values["patched_prob"].item()),
                    "corruption_gap": float(values["corruption_gap"].item()),
                    "normalized_restore": float(values["score"].item()),
                }
            )
    return pd.DataFrame(rows)


def summarize_causal_objective(
    frame: pd.DataFrame,
    *,
    quantile: float,
    causal_threshold: float,
    epsilon: float,
) -> dict[str, float]:
    direction_scores = {}
    direction_coverage = {}
    for direction in ("forward", "reverse"):
        values = frame.loc[
            frame["direction"] == direction, "normalized_restore"
        ] if not frame.empty else pd.Series(dtype=float)
        direction_scores[direction] = robust_restore_score(
            values, quantile=quantile, causal_threshold=causal_threshold
        )
        direction_coverage[direction] = restore_coverage(
            values, causal_threshold=causal_threshold
        )
    score = harmonic_pair(
        direction_scores["forward"], direction_scores["reverse"], epsilon
    )
    return {
        "target_causal_score": score,
        "forward_causal_score": direction_scores["forward"],
        "reverse_causal_score": direction_scores["reverse"],
        "forward_causal_coverage": direction_coverage["forward"],
        "reverse_causal_coverage": direction_coverage["reverse"],
        "mean_normalized_restore": (
            float(pd.to_numeric(frame["normalized_restore"], errors="coerce").mean())
            if not frame.empty else 0.0
        ),
    }


def select_method_independent_audit_layers(
    layer_summary: pd.DataFrame, cfg: dict
) -> list[int]:
    """Select layers for method-independent auditing."""
    score_column = "target_causal_score"
    if score_column not in layer_summary:
        raise ValueError(f"causal audit requires {score_column}")
    count = int(cfg.get("layer_count", cfg.get("candidate_pool_size", 12)))
    ranked = layer_summary.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="raise")
    ranked["layer"] = pd.to_numeric(ranked["layer"], errors="raise").astype(int)
    ranked = ranked.sort_values([score_column, "layer"], ascending=[False, True])
    layers = [int(value) for value in ranked["layer"].drop_duplicates().head(count)]
    if not layers:
        raise RuntimeError("method-independent causal audit selected no layers")
    return layers


def causal_audit_reductions(
    baseline_summary: dict, edited_summary: dict, direction_counts: dict, epsilon: float
) -> dict[str, float | None]:
    """Summarize causal audit reductions."""
    result: dict[str, float | None] = {}
    available = []
    for direction in ("forward", "reverse"):
        if int(direction_counts.get(direction, 0)) <= 0:
            result[f"{direction}_relative_drop"] = None
            continue
        before = float(baseline_summary[f"{direction}_causal_score"])
        after = float(edited_summary[f"{direction}_causal_score"])
        drop = float((before - after) / max(before, epsilon))
        result[f"{direction}_relative_drop"] = drop
        available.append(drop)
    result["direction_macro_relative_drop"] = (
        float(sum(available) / len(available)) if available else None
    )
    result["min_direction_relative_drop"] = min(available) if available else None
    result["remaining_max_direction_causal_score"] = max(
        [float(edited_summary[f"{d}_causal_score"]) for d in ("forward", "reverse")
         if int(direction_counts.get(d, 0)) > 0],
        default=None,
    )
    return result


def causal_direction_progress(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    quantile: float,
    causal_threshold: float,
    epsilon: float,
) -> dict[str, float]:
    progress = {}
    for direction in ("forward", "reverse"):
        base_values = baseline.loc[
            baseline["direction"] == direction, "normalized_restore"
        ]
        current_values = current.loc[
            current["direction"] == direction, "normalized_restore"
        ]
        base_score = robust_restore_score(
            base_values, quantile=quantile, causal_threshold=causal_threshold
        )
        current_score = robust_restore_score(
            current_values, quantile=quantile, causal_threshold=causal_threshold
        )
        progress[direction] = float(
            (base_score - current_score) / max(base_score, float(epsilon))
        )
    return progress


def answer_metrics(distributions: torch.Tensor, targets: torch.Tensor) -> dict[str, float | bool]:
    indices = torch.arange(targets.numel(), device=targets.device)
    probabilities = distributions[indices, targets].clamp_min(1e-30)
    mean_nll = float((-probabilities.log().mean()).item())
    token_accuracy = float((distributions.argmax(dim=-1) == targets).float().mean().item())
    return {
        "answer_score": float(torch.exp(-probabilities.log().neg().mean()).item()),
        "answer_nll": mean_nll,
        "answer_token_accuracy": token_accuracy,
        "teacher_forced_exact": bool(token_accuracy == 1.0),
    }


def greedy_answer_exact(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str,
) -> bool:
    """Check exact greedy answer recall."""

    generated = token_ids(tokenizer, prompt, device)
    targets = token_ids(tokenizer, answer, device)[0]
    predictions: list[int] = []
    for _ in range(targets.numel()):
        next_token = int(model(input_ids=generated).logits[:, -1, :].argmax(dim=-1).item())
        predictions.append(next_token)
        generated = torch.cat(
            [generated, torch.tensor([[next_token]], device=device, dtype=generated.dtype)],
            dim=1,
        )
    return predictions == targets.tolist()


@torch.no_grad()
def build_base_cache(
    model,
    tokenizer,
    facts: list[Fact],
    device: str,
) -> dict[tuple[str, str], dict]:
    cache: dict[tuple[str, str], dict] = {}
    for fact in facts:
        key = (fact.prompt, fact.answer)
        if key not in cache:
            distributions, targets = answer_distributions(
                model, tokenizer, fact.prompt, fact.answer, device
            )
            cache[key] = {
                "distributions": distributions.cpu(),
                "targets": targets.cpu(),
                "greedy_exact": greedy_answer_exact(
                    model, tokenizer, fact.prompt, fact.answer, device
                ),
                **answer_metrics(distributions, targets),
            }
    return cache


def base_prompt_is_known(cached: dict, evaluation: dict) -> bool:
    if bool(evaluation["require_base_greedy_exact"]):
        return bool(cached["greedy_exact"])
    return bool(cached["greedy_exact"]) or float(cached["answer_score"]) >= float(
        evaluation["min_forget_base_answer_score"]
    )


def training_eligibility(
    facts: list[Fact],
    base_cache: dict[tuple[str, str], dict],
    evaluation: dict,
) -> tuple[list[Fact], pd.DataFrame]:
    rows = []
    selected = []
    for fact in facts:
        cached = base_cache[(fact.prompt, fact.answer)]
        known = base_prompt_is_known(cached, evaluation)
        if known:
            selected.append(fact)
        rows.append(
            {
                "id": fact.id,
                "family": family(fact),
                "direction": causal_direction(fact),
                "prompt": fact.prompt,
                "answer": fact.answer,
                "base_answer_score": float(cached["answer_score"]),
                "base_greedy_exact": bool(cached["greedy_exact"]),
                "selected_for_training": known,
            }
        )
    if not selected:
        raise RuntimeError("no base-known forget prompts are available for causal training")
    return selected, pd.DataFrame(rows)


def causal_parameter_weights(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    layer_summary: pd.DataFrame,
    *,
    floor: float,
    power: float,
) -> tuple[list[float], dict[int, float]]:
    """Compute causal parameter weights."""

    score_column = (
        "target_causal_score"
        if "target_causal_score" in layer_summary.columns
        else "causal_score"
    )
    scores = {
        int(getattr(row, "layer")): float(getattr(row, score_column))
        for row in layer_summary.itertuples(index=False)
    }
    selected_layers = sorted(
        {
            int(name.split("gpt_neox.layers.", 1)[1].split(".", 1)[0])
            for name, _ in named_parameters
        }
    )
    selected_scores = [scores[layer] for layer in selected_layers]
    low, high = min(selected_scores), max(selected_scores)
    layer_weights: dict[int, float] = {}
    if high - low <= 1e-12:
        layer_weights = {layer: 1.0 for layer in selected_layers}
    else:
        for layer in selected_layers:
            normalized = (scores[layer] - low) / (high - low)
            layer_weights[layer] = float(floor + (1.0 - floor) * normalized ** power)
    parameter_weights = []
    for name, _ in named_parameters:
        layer = int(name.split("gpt_neox.layers.", 1)[1].split(".", 1)[0])
        parameter_weights.append(layer_weights[layer])
    return parameter_weights, layer_weights


def apply_gradient_weights(
    gradients_: list[torch.Tensor], weights: list[float]
) -> list[torch.Tensor]:
    if len(gradients_) != len(weights):
        raise ValueError("gradient and causal-weight lengths do not match")
    return [gradient * float(weight) for gradient, weight in zip(gradients_, weights)]


def answer_complement_loss(
    model, tokenizer, fact: Fact, device: str, epsilon: float
) -> torch.Tensor:
    """Compute the answer-complement loss."""

    distributions, targets = answer_distributions(
        model, tokenizer, fact.prompt, fact.answer, device
    )
    indices = torch.arange(targets.numel(), device=device)
    token_probabilities = distributions[indices, targets].clamp_min(epsilon)
    answer_nll = -token_probabilities.log().mean()
    answer_probability = torch.exp(-answer_nll).clamp(
        min=epsilon, max=1.0 - epsilon
    )
    return -torch.log1p(-answer_probability)


def retain_loss(
    model,
    tokenizer,
    fact: Fact,
    base_cache: dict[tuple[str, str], dict],
    device: str,
    epsilon: float,
) -> torch.Tensor:
    loss, _, _, _ = retain_loss_and_drop(
        model, tokenizer, fact, base_cache, device, epsilon
    )
    return loss


def retain_loss_and_drop(
    model,
    tokenizer,
    fact: Fact,
    base_cache: dict[tuple[str, str], dict],
    device: str,
    epsilon: float,
    *,
    answer_margin: float = 0.0,
    margin_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float, float]:
    """Measure retain loss and probability drop."""

    base = base_cache[(fact.prompt, fact.answer)]["distributions"].to(
        device=device, dtype=torch.float32
    ).clamp_min(epsilon)
    current, targets = answer_distributions(
        model, tokenizer, fact.prompt, fact.answer, device
    )
    current = current.float().clamp_min(epsilon)
    kl_loss = (base * (base.log() - current.log())).sum(dim=-1).mean()
    indices = torch.arange(targets.numel(), device=targets.device)
    target_probabilities = current[indices, targets].clamp_min(epsilon)
    current_score = torch.exp(target_probabilities.log().mean())
    base_score = torch.as_tensor(
        float(base_cache[(fact.prompt, fact.answer)]["answer_score"]),
        device=current_score.device,
        dtype=current_score.dtype,
    )
    score_drop = base_score - current_score
    margin_loss = torch.relu(score_drop - float(answer_margin))
    total_loss = kl_loss + float(margin_weight) * margin_loss
    return (
        total_loss,
        max(float(score_drop.detach().item()), 0.0),
        float(kl_loss.detach().item()),
        float(margin_loss.detach().item()),
    )


def _zeros_like(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter) for parameter in parameters]


def gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [torch.zeros_like(parameter) if value is None else value.detach() for parameter, value in zip(parameters, values)]


def answer_residual_cap_fractions(
    residuals: dict[str, float], *, minimum: float, epsilon: float,
) -> dict[str, float]:
    """Allocate residual gradient caps."""

    if not residuals:
        raise ValueError("answer residuals cannot be empty")
    peak = max(max(float(value), 0.0) for value in residuals.values())
    if peak <= float(epsilon):
        return {direction: 1.0 for direction in residuals}
    return {
        direction: max(float(minimum), max(float(value), 0.0) / peak)
        for direction, value in residuals.items()
    }


def tier_auxiliary_gradient_by_retain_risk(
    values: list[torch.Tensor],
    unsafe_parameter_mask: list[bool],
    *,
    global_floor_ratio: float,
    maximum_ratio: float,
) -> list[torch.Tensor]:
    """Scale auxiliary gradients by retain risk."""

    if len(values) != len(unsafe_parameter_mask):
        raise ValueError("auxiliary gradient and retain-risk mask lengths do not match")
    if not 0.0 < float(global_floor_ratio) <= float(maximum_ratio):
        raise ValueError("global floor must be positive and no larger than maximum ratio")
    unsafe_scale = float(global_floor_ratio) / float(maximum_ratio)
    return [
        value * unsafe_scale if unsafe else value
        for value, unsafe in zip(values, unsafe_parameter_mask)
    ]


def combine_primary_and_auxiliary_gradients(
    primary_grad: list[torch.Tensor],
    auxiliary_grad: list[torch.Tensor],
    *,
    primary_weight: float,
    auxiliary_weight: float,
    max_auxiliary_ratio: float,
    epsilon: float,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Combine primary and auxiliary gradients."""

    weighted_primary = [value * float(primary_weight) for value in primary_grad]
    weighted_auxiliary = [value * float(auxiliary_weight) for value in auxiliary_grad]
    primary_norm = gradient_norm(weighted_primary)
    raw_auxiliary_norm = gradient_norm(weighted_auxiliary)
    dot_before = float(
        sum(
            (primary.float() * auxiliary.float()).sum()
            for primary, auxiliary in zip(weighted_primary, weighted_auxiliary)
        ).item()
    )
    denominator = max(primary_norm * raw_auxiliary_norm, float(epsilon))
    cosine_before = float(dot_before / denominator)
    conflict = bool(
        primary_norm > float(epsilon)
        and raw_auxiliary_norm > float(epsilon)
        and dot_before < 0.0
    )
    projected = weighted_auxiliary
    if conflict:
        primary_norm_sq = sum(
            value.float().pow(2).sum() for value in weighted_primary
        ).clamp_min(float(epsilon))
        coefficient = torch.as_tensor(
            dot_before, device=primary_norm_sq.device, dtype=primary_norm_sq.dtype
        ) / primary_norm_sq
        projected = [
            auxiliary - coefficient.to(auxiliary.dtype) * primary
            for primary, auxiliary in zip(weighted_primary, weighted_auxiliary)
        ]
    projected_auxiliary_norm = gradient_norm(projected)
    dot_after = float(
        sum(
            (primary.float() * auxiliary.float()).sum()
            for primary, auxiliary in zip(weighted_primary, projected)
        ).item()
    )
    denominator_after = max(
        primary_norm * projected_auxiliary_norm, float(epsilon)
    )
    cosine_after = float(dot_after / denominator_after)
    if projected_auxiliary_norm <= float(epsilon) or primary_norm <= float(epsilon):
        auxiliary_scale = 0.0
    else:
        auxiliary_scale = min(
            1.0,
            float(max_auxiliary_ratio) * primary_norm / projected_auxiliary_norm,
        )
    combined = [
        primary + auxiliary * auxiliary_scale
        for primary, auxiliary in zip(weighted_primary, projected)
    ]
    return combined, {
        "causal_primary_grad_norm": primary_norm,
        "answer_auxiliary_raw_grad_norm": raw_auxiliary_norm,
        "answer_auxiliary_projected_grad_norm": projected_auxiliary_norm,
        "answer_auxiliary_applied_scale": float(auxiliary_scale),
        "answer_auxiliary_applied_grad_norm": float(
            projected_auxiliary_norm * auxiliary_scale
        ),
        "answer_auxiliary_gradient_cosine_before": cosine_before,
        "answer_auxiliary_gradient_conflict": conflict,
        "answer_auxiliary_gradient_projected": conflict,
        "answer_auxiliary_gradient_cosine_after": cosine_after,
    }


def combine_gradients(
    forget_grad: list[torch.Tensor],
    retain_grad: list[torch.Tensor],
    *,
    parameter_names: list[str],
    mode: str,
    retain_weight: float,
    normalization: str,
    epsilon: float,
    minimum_retain_gradient_ratio: float = 0.0,
    retain_null_parameter_mask: list[bool] | None = None,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    if len(parameter_names) != len(forget_grad) or len(forget_grad) != len(retain_grad):
        raise ValueError("gradient and parameter-name lengths do not match")
    if not 0.0 <= float(minimum_retain_gradient_ratio) <= 1.0:
        raise ValueError("minimum_retain_gradient_ratio must be in [0, 1]")
    if retain_null_parameter_mask is None:
        retain_null_parameter_mask = [False] * len(parameter_names)
    if len(retain_null_parameter_mask) != len(parameter_names):
        raise ValueError("retain-null mask and parameter-name lengths do not match")
    dot = sum((left.float() * right.float()).sum() for left, right in zip(forget_grad, retain_grad))
    forget_norm_sq = sum(value.float().pow(2).sum() for value in forget_grad)
    retain_norm_sq = sum(value.float().pow(2).sum() for value in retain_grad)
    retain_scale = 1.0
    if normalization == "match_forget" and mode != "forget_only" and retain_norm_sq.item() > epsilon:
        retain_scale = float(
            (forget_norm_sq.sqrt() / retain_norm_sq.sqrt().clamp_min(float(epsilon))).item()
        )
        retain_grad = [value * retain_scale for value in retain_grad]

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(parameter_names):
        group = name.rsplit(".", 1)[0]
        groups.setdefault(group, []).append(index)
    projected = [value.clone() for value in forget_grad]
    raw_conflicts = 0
    active_conflicts = 0
    active_groups = 0
    skipped_low_signal_groups = 0
    projections = 0
    coefficients: list[float] = []
    active_parameter_mask = [False] * len(parameter_names)
    minimum_ratio_sq = float(minimum_retain_gradient_ratio) ** 2
    for indices in groups.values():
        group_dot = sum(
            (forget_grad[index].float() * retain_grad[index].float()).sum()
            for index in indices
        )
        group_forget_norm_sq = sum(
            forget_grad[index].float().pow(2).sum() for index in indices
        )
        group_retain_norm_sq = sum(
            retain_grad[index].float().pow(2).sum() for index in indices
        )
        raw_conflict = bool(group_dot.item() < 0.0)
        raw_conflicts += int(raw_conflict)
        
        
        
        
        
        
        retain_signal_active = bool(
            group_retain_norm_sq.item() > epsilon
            and group_retain_norm_sq.item()
            >= minimum_ratio_sq * max(group_forget_norm_sq.item(), epsilon)
        )
        active_groups += int(retain_signal_active)
        skipped_low_signal_groups += int(
            group_retain_norm_sq.item() > epsilon and not retain_signal_active
        )
        for index in indices:
            active_parameter_mask[index] = retain_signal_active
        conflict = bool(retain_signal_active and raw_conflict)
        active_conflicts += int(conflict)
        hybrid_null = mode in {"retain_null_pcgrad", "retain_null_sago"} and any(
            retain_null_parameter_mask[index] for index in indices
        )
        project = retain_signal_active and (
            mode == "retain_null"
            or hybrid_null
            or (mode in {"pcgrad", "retain_null_pcgrad"} and conflict)
        )
        if project:
            coefficient = group_dot / group_retain_norm_sq.clamp_min(float(epsilon))
            coefficients.append(float(coefficient.item()))
            projections += 1
            for index in indices:
                projected[index] = forget_grad[index] - coefficient.to(
                    forget_grad[index].dtype
                ) * retain_grad[index]

    sago_keep_fraction = 1.0
    if mode in {"sago", "retain_null_sago"}:
        kept = 0
        total = 0
        for index, (left, right) in enumerate(zip(projected, retain_grad)):
            
            
            keep = (
                torch.ones_like(left, dtype=torch.bool)
                if not active_parameter_mask[index]
                else ((left * right >= 0) | (right == 0))
            )
            kept += int(keep.sum().item())
            total += keep.numel()
        sago_keep_fraction = float(kept / max(total, 1))
    if mode == "forget_only":
        combined = projected
    elif mode in {"sago", "retain_null_sago"}:
        
        
        
        
        combined = []
        for index, (left, right) in enumerate(zip(projected, retain_grad)):
            conflict = (
                left * right < 0
                if active_parameter_mask[index]
                else torch.zeros_like(left, dtype=torch.bool)
            )
            combined.append(torch.where(conflict, float(retain_weight) * right, left))
    else:
        combined = [
            left + float(retain_weight) * right
            for left, right in zip(projected, retain_grad)
        ]
    cosine = float(
        (dot / (forget_norm_sq.sqrt() * retain_norm_sq.sqrt()).clamp_min(float(epsilon))).item()
    )
    return combined, {
        "gradient_dot": float(dot.item()),
        "gradient_cosine": cosine,
        "gradient_conflict": bool(active_conflicts > 0),
        "gradient_conflict_fraction": float(active_conflicts / max(len(groups), 1)),
        "gradient_raw_conflict_fraction": float(raw_conflicts / max(len(groups), 1)),
        "retain_surgery_active_fraction": float(active_groups / max(len(groups), 1)),
        "retain_surgery_skipped_low_signal_fraction": float(
            skipped_low_signal_groups / max(len(groups), 1)
        ),
        "gradient_projected": bool(
            projections > 0
            or (
                mode in {"sago", "retain_null_sago"}
                and active_groups > 0
            )
        ),
        "gradient_projected_fraction": float(projections / max(len(groups), 1)),
        "projection_coefficient_mean": float(sum(coefficients) / max(len(coefficients), 1)),
        "projection_coefficient_max_abs": float(max((abs(v) for v in coefficients), default=0.0)),
        "sago_keep_fraction": sago_keep_fraction,
        "retain_gradient_scale": retain_scale,
        "forget_grad_norm": float(forget_norm_sq.sqrt().item()),
        "retain_grad_norm": float(retain_norm_sq.sqrt().item()),
    }


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    facts: list[Fact],
    base_cache: dict[tuple[str, str], dict],
    device: str,
    epsilon: float,
    min_forget_base_answer_score: float,
    require_base_greedy_exact: bool,
) -> pd.DataFrame:
    rows = []
    for fact in facts:
        cached = base_cache[(fact.prompt, fact.answer)]
        base = cached["distributions"].float().clamp_min(epsilon)
        targets = cached["targets"].long()
        current, current_targets = answer_distributions(
            model, tokenizer, fact.prompt, fact.answer, device
        )
        current = current.cpu().float().clamp_min(epsilon)
        if not torch.equal(targets, current_targets.cpu()):
            raise RuntimeError("answer tokenization changed during evaluation")
        current_metrics = answer_metrics(current, targets)
        current_greedy_exact = greedy_answer_exact(
            model, tokenizer, fact.prompt, fact.answer, device
        )
        first_target = int(targets[0].item())
        base_probability = float(base[0, first_target].item())
        current_probability = float(current[0, first_target].item())
        base_score = float(cached["answer_score"])
        current_score = float(current_metrics["answer_score"])
        score_eligible = base_score >= float(min_forget_base_answer_score)
        exact_eligible = bool(cached["greedy_exact"])
        forget_eligible = bool(
            fact.type != "forget"
            or (exact_eligible if require_base_greedy_exact else (exact_eligible or score_eligible))
        )
        rows.append(
            {
                "id": fact.id,
                "base_id": base_id(fact),
                "family": family(fact),
                "direction": causal_direction(fact),
                "type": fact.type,
                "prompt": fact.prompt,
                "answer": fact.answer,
                "base_first_token_prob": base_probability,
                "cut_first_token_prob": current_probability,
                "prob_drop": base_probability - current_probability,
                "base_answer_score": base_score,
                "cut_answer_score": current_score,
                "answer_score_drop": base_score - current_score,
                "base_answer_nll": float(cached["answer_nll"]),
                "cut_answer_nll": float(current_metrics["answer_nll"]),
                "base_answer_token_accuracy": float(cached["answer_token_accuracy"]),
                "cut_answer_token_accuracy": float(current_metrics["answer_token_accuracy"]),
                "base_teacher_forced_exact": bool(cached["teacher_forced_exact"]),
                "cut_teacher_forced_exact": bool(current_metrics["teacher_forced_exact"]),
                "base_greedy_exact": bool(cached["greedy_exact"]),
                "cut_greedy_exact": current_greedy_exact,
                "forget_eligible": forget_eligible,
                "retain_kl": (
                    float((base * (base.log() - current.log())).sum(dim=-1).mean().item())
                    if fact.type == "retain"
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


class ShuffledBatchStream:
    """Stream shuffled mini-batches."""

    def __init__(self, facts: list[Fact], batch_size: int, seed: int) -> None:
        if not facts or batch_size <= 0:
            raise ValueError("batch stream requires facts and a positive batch size")
        self.facts = list(facts)
        self.batch_size = int(batch_size)
        self.random = random.Random(int(seed))
        self.order: list[int] = []
        self.cursor = 0

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.facts)))
        self.random.shuffle(self.order)
        self.cursor = 0

    def next(self) -> list[Fact]:
        batch: list[Fact] = []
        while len(batch) < self.batch_size:
            if self.cursor >= len(self.order):
                self._reshuffle()
            take = min(self.batch_size - len(batch), len(self.order) - self.cursor)
            indices = self.order[self.cursor:self.cursor + take]
            batch.extend(self.facts[index] for index in indices)
            self.cursor += take
        return batch


class StratifiedBatchStream:
    """Stream stratified mini-batches."""

    def __init__(self, facts: list[Fact], batch_size: int, seed: int, key_fn) -> None:
        groups: dict[str, list[Fact]] = {}
        for fact in facts:
            groups.setdefault(str(key_fn(fact)), []).append(fact)
        if not groups:
            raise ValueError("stratified batch stream requires non-empty facts")
        self.keys = sorted(groups)
        self.streams = {
            key: ShuffledBatchStream(group, 1, seed + 1009 * index)
            for index, (key, group) in enumerate(sorted(groups.items()))
        }
        self.batch_size = int(batch_size)
        self.cursor = 0

    def next(self) -> list[Fact]:
        batch: list[Fact] = []
        while len(batch) < self.batch_size:
            key = self.keys[self.cursor % len(self.keys)]
            batch.extend(self.streams[key].next())
            self.cursor += 1
        return batch


def dual_signal_hardness(
    causal_score: float,
    baseline_causal_score: float,
    answer_probability: float,
    baseline_answer_probability: float,
    *,
    answer_weight: float,
    epsilon: float,
) -> float:
    """Measure dual-signal difficulty."""

    if not 0.0 <= float(answer_weight) <= 1.0:
        raise ValueError("answer_weight must be in [0, 1]")
    baseline_causal = float(baseline_causal_score)
    causal_residual = (
        max(float(causal_score), 0.0) / baseline_causal
        if baseline_causal > float(epsilon)
        else 0.0
    )
    causal_residual = min(causal_residual, 1.0)
    answer_residual = max(float(answer_probability), 0.0) / max(
        float(baseline_answer_probability), float(epsilon)
    )
    answer_residual = min(answer_residual, 1.0)
    return max(causal_residual, float(answer_weight) * answer_residual)


class AdaptiveDirectionalBatchStream:
    """Sample batches using directional difficulty."""

    def __init__(
        self,
        facts: list[Fact],
        batch_size: int,
        seed: int,
        initial_hardness: dict[tuple[str, str], float],
        hard_every: int,
    ) -> None:
        groups: dict[str, list[Fact]] = {}
        for fact in facts:
            groups.setdefault(causal_direction(fact), []).append(fact)
        if not groups:
            raise ValueError("adaptive forget stream requires non-empty facts")
        self.keys = sorted(groups)
        self.groups = {key: list(value) for key, value in groups.items()}
        self.random_streams = {
            key: ShuffledBatchStream(value, 1, seed + 2017 * index)
            for index, (key, value) in enumerate(sorted(groups.items()))
        }
        self.hardness = dict(initial_hardness)
        self.last_hard_step: dict[tuple[str, str], int] = {}
        self.batch_size = int(batch_size)
        self.hard_every = int(hard_every)
        self.step = 0

    def next(self) -> list[Fact]:
        self.step += 1
        batch: list[Fact] = []
        for offset in range(self.batch_size):
            direction = self.keys[offset % len(self.keys)]
            use_hard = self.step % self.hard_every == 0
            if use_hard:
                fact = max(
                    self.groups[direction],
                    key=lambda item: (
                        float(self.hardness.get((item.prompt, item.answer), 0.0)),
                        -self.last_hard_step.get((item.prompt, item.answer), -1),
                        item.id,
                    ),
                )
                self.last_hard_step[(fact.prompt, fact.answer)] = self.step
            else:
                fact = self.random_streams[direction].next()[0]
            batch.append(fact)
        return batch

    def update(self, fact: Fact, hardness: float) -> None:
        self.hardness[(fact.prompt, fact.answer)] = float(hardness)


class CausalNeighborRetainStream:
    """Sample retain facts near the target relation."""

    def __init__(
        self,
        facts: list[Fact],
        batch_size: int,
        seed: int,
        neighbor_scores: dict[tuple[str, str], float],
        neighbor_fraction: float,
        neighbor_pool_fraction: float = 1.0,
    ) -> None:
        self.base = StratifiedBatchStream(
            facts,
            batch_size,
            seed,
            lambda fact: fact.relation or "__unknown_relation__",
        )
        neighbors = sorted(
            [
                fact
                for fact in facts
                if float(neighbor_scores.get((fact.prompt, fact.answer), 0.0)) > 0.0
            ],
            key=lambda fact: (
                -float(neighbor_scores[(fact.prompt, fact.answer)]), fact.id
            ),
        )
        if neighbors:
            neighbor_pool_count = max(
                1, math.ceil(len(neighbors) * float(neighbor_pool_fraction))
            )
            neighbors = neighbors[:neighbor_pool_count]
        self.neighbor = (
            ShuffledBatchStream(neighbors, 1, seed + 7919) if neighbors else None
        )
        self.batch_size = int(batch_size)
        self.anchor_count = min(
            self.batch_size,
            max(0, math.ceil(self.batch_size * float(neighbor_fraction))),
        )

    def next(self) -> list[Fact]:
        batch: list[Fact] = []
        seen: set[tuple[str, str]] = set()
        if self.neighbor is not None:
            attempts = 0
            while len(batch) < self.anchor_count and attempts < 4 * self.anchor_count + 4:
                fact = self.neighbor.next()[0]
                key = (fact.prompt, fact.answer)
                attempts += 1
                if key not in seen:
                    batch.append(fact)
                    seen.add(key)
        attempts = 0
        while len(batch) < self.batch_size and attempts < 8 * self.batch_size:
            for fact in self.base.next():
                key = (fact.prompt, fact.answer)
                if key not in seen:
                    batch.append(fact)
                    seen.add(key)
                    if len(batch) == self.batch_size:
                        break
            attempts += 1
        if len(batch) < self.batch_size:
            raise RuntimeError("could not construct a unique causal-neighbor retain batch")
        return batch


def partition_direction_crossfit(
    facts: list[Fact],
    base_cache: dict[tuple[str, str], dict],
    calibration_fraction: float,
    calibration_max_per_direction: int = 1,
    preferred_optimization_per_direction: int = 2,
) -> tuple[list[Fact], list[Fact], pd.DataFrame]:
    """Partition directions for cross-fitting."""

    optimization: list[Fact] = []
    calibration: list[Fact] = []
    rows: list[dict] = []
    family_priority = {
        "causal_probe_forward": 0,
        "causal_probe_reverse": 0,
        "clue": 1,
        "neighbor": 2,
        "alias": 3,
        "paraphrase": 4,
        "reverse": 4,
        "factual": 5,
    }
    for direction in ("forward", "reverse"):
        group = [fact for fact in facts if causal_direction(fact) == direction]
        if len(group) < 2:
            raise RuntimeError(
                f"cross-fit direction {direction!r} needs at least two eligible prompts"
            )
        calibration_count = min(
            len(group) - 1,
            int(calibration_max_per_direction),
            max(1, math.ceil(len(group) * float(calibration_fraction))),
        )
        optimization_count = len(group) - calibration_count
        support_sufficient = optimization_count >= int(
            preferred_optimization_per_direction
        )
        ordered = sorted(
            group,
            key=lambda fact: (
                family_priority.get(family(fact), 9),
                -float(base_cache[(fact.prompt, fact.answer)]["answer_score"]),
                fact.id,
            ),
        )
        calibration_keys = {
            (fact.prompt, fact.answer) for fact in ordered[:calibration_count]
        }
        for fact in group:
            role = (
                "calibration"
                if (fact.prompt, fact.answer) in calibration_keys
                else "optimization"
            )
            (calibration if role == "calibration" else optimization).append(fact)
            rows.append(
                {
                    "id": fact.id,
                    "family": family(fact),
                    "direction": direction,
                    "crossfit_role": role,
                    "optimization_support_sufficient": support_sufficient,
                    "optimization_direction_count": optimization_count,
                    "preferred_optimization_direction_count": int(
                        preferred_optimization_per_direction
                    ),
                    "prompt": fact.prompt,
                    "answer": fact.answer,
                    "base_answer_score": float(
                        base_cache[(fact.prompt, fact.answer)]["answer_score"]
                    ),
                }
            )
    return optimization, calibration, pd.DataFrame(rows)


def direction_weights_from_progress(
    progress: dict[str, float],
    *,
    target_drop: float,
    power: float,
    minimum: float,
    maximum: float,
    epsilon: float,
) -> dict[str, float]:
    """Derive direction weights from progress."""

    difficulty = {
        direction: max(float(target_drop) - float(value), epsilon) ** float(power)
        for direction, value in progress.items()
    }
    mean_difficulty = sum(difficulty.values()) / len(difficulty)
    return {
        direction: min(
            float(maximum),
            max(float(minimum), value / max(mean_difficulty, epsilon)),
        )
        for direction, value in difficulty.items()
    }


def trace_direction_weights(
    layer_summary: pd.DataFrame,
    selected_layers: list[int],
    *,
    power: float,
    minimum: float,
    maximum: float,
    epsilon: float,
) -> dict[str, float]:
    """Trace direction weights."""

    selected = layer_summary[
        pd.to_numeric(layer_summary["layer"], errors="coerce").isin(selected_layers)
    ].copy()
    strengths: dict[str, float] = {}
    for direction in ("forward", "reverse"):
        column = f"{direction}_target_causal_score"
        if column not in selected.columns or selected.empty:
            strengths[direction] = 0.0
        else:
            strengths[direction] = float(
                pd.to_numeric(selected[column], errors="coerce").fillna(0.0).mean()
            )
    best = max(strengths.values()) if strengths else 0.0
    if best <= epsilon:
        return {direction: 1.0 for direction in ("forward", "reverse")}
    progress = {
        direction: strengths[direction] / max(best, epsilon)
        for direction in ("forward", "reverse")
    }
    return direction_weights_from_progress(
        progress,
        target_drop=1.0,
        power=power,
        minimum=minimum,
        maximum=maximum,
        epsilon=epsilon,
    )


def blend_direction_weights(
    calibration_weights: dict[str, float],
    trace_weights: dict[str, float],
    support_confidence: dict[str, float],
    *,
    minimum: float,
    maximum: float,
    epsilon: float,
) -> dict[str, float]:
    """Blend directional weights."""

    blended: dict[str, float] = {}
    for direction in ("forward", "reverse"):
        confidence = min(1.0, max(0.0, float(support_confidence.get(direction, 0.0))))
        blended[direction] = (
            confidence * float(calibration_weights.get(direction, 1.0))
            + (1.0 - confidence) * float(trace_weights.get(direction, 1.0))
        )
    mean_weight = sum(blended.values()) / max(len(blended), 1)
    return {
        direction: min(
            float(maximum),
            max(float(minimum), value / max(mean_weight, epsilon)),
        )
        for direction, value in blended.items()
    }


def causal_neighbor_scores(
    facts: list[Fact], trace: pd.DataFrame, selected_layers: list[int]
) -> dict[tuple[str, str], float]:
    """Score causal neighbors."""

    required = {"id", "type", "layer", "normalized_restore"}
    if not required.issubset(trace.columns):
        raise ValueError(f"causal trace lacks neighbor fields: {sorted(required - set(trace.columns))}")
    retain_trace = trace[
        (trace["type"] == "retain")
        & (pd.to_numeric(trace["layer"], errors="coerce").isin(selected_layers))
    ].copy()
    retain_trace["normalized_restore"] = pd.to_numeric(
        retain_trace["normalized_restore"], errors="coerce"
    )
    retain_trace = retain_trace.dropna(subset=["normalized_restore"])
    retain_trace["base_id"] = retain_trace["id"].astype(str).map(
        lambda value: value.split(":", 1)[0]
    )
    overlap = retain_trace.groupby("base_id")["normalized_restore"].mean().to_dict()
    return {
        (fact.prompt, fact.answer): float(overlap.get(base_id(fact), 0.0))
        for fact in facts
    }


def mean_fact_loss(loss_fn, facts: list[Fact]) -> torch.Tensor:
    return torch.stack([loss_fn(fact) for fact in facts]).mean()


def mean_gradients(gradient_sets: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    if not gradient_sets:
        raise ValueError("at least one gradient set is required")
    averaged: list[torch.Tensor] = []
    for values in zip(*gradient_sets):
        value = values[0].clone()
        for other in values[1:]:
            value.add_(other)
        averaged.append(value.div_(len(values)))
    return averaged


def gradient_norm(values: list[torch.Tensor]) -> float:
    return float(sum(value.float().pow(2).sum() for value in values).sqrt().item())


def gradient_cosine(
    left: list[torch.Tensor] | None,
    right: list[torch.Tensor] | None,
    epsilon: float,
) -> float | None:
    if left is None or right is None:
        return None
    dot = sum((a.float() * b.float()).sum() for a, b in zip(left, right))
    left_sq = sum(a.float().pow(2).sum() for a in left)
    right_sq = sum(b.float().pow(2).sum() for b in right)
    return float((dot / (left_sq.sqrt() * right_sq.sqrt()).clamp_min(epsilon)).item())


def combine_direction_gradients(
    direction_grads: dict[str, list[torch.Tensor]],
    direction_weights: dict[str, float],
    *,
    policy: str,
    epsilon: float,
) -> tuple[list[torch.Tensor], dict[str, float | bool | str | None]]:
    """Combine directional gradients."""
    if policy not in {"weighted_mean", "symmetric_pcgrad"}:
        raise ValueError(f"unknown direction gradient policy: {policy}")
    active = {k: v for k, v in direction_grads.items() if k in direction_weights}
    if not active:
        raise ValueError("no directional forget gradients are available")
    weight_sum = sum(float(direction_weights[k]) for k in active)
    if weight_sum <= 0.0:
        raise ValueError("active direction weights must have a positive sum")
    forward = active.get("forward")
    reverse = active.get("reverse")
    cosine_before = gradient_cosine(forward, reverse, epsilon)
    conflict = bool(cosine_before is not None and cosine_before < 0.0)
    projected = False
    if policy == "symmetric_pcgrad" and forward is not None and reverse is not None and conflict:
        dot = sum((a.float() * b.float()).sum() for a, b in zip(forward, reverse))
        forward_sq = sum(a.float().pow(2).sum() for a in forward)
        reverse_sq = sum(b.float().pow(2).sum() for b in reverse)
        original_forward = forward
        original_reverse = reverse
        forward = [a - (dot / reverse_sq.clamp_min(epsilon)).to(a.dtype) * b for a, b in zip(original_forward, original_reverse)]
        reverse = [b - (dot / forward_sq.clamp_min(epsilon)).to(b.dtype) * a for a, b in zip(original_forward, original_reverse)]
        active["forward"], active["reverse"] = forward, reverse
        projected = True
    combined = [
        sum(active[k][i] * direction_weights[k] for k in active) / weight_sum
        for i in range(len(next(iter(active.values()))))
    ]
    return combined, {
        "direction_gradient_policy": policy,
        "forward_reverse_gradient_cosine": cosine_before,
        "forward_reverse_gradient_conflict": conflict,
        "forward_reverse_gradient_projected": projected,
        "forward_reverse_gradient_cosine_after": gradient_cosine(forward, reverse, epsilon),
    }


def sign_keep_fraction(
    forget_grad: list[torch.Tensor], retain_grad: list[torch.Tensor]
) -> float:
    kept = 0
    total = 0
    for left, right in zip(forget_grad, retain_grad):
        keep = (left * right >= 0) | (right == 0)
        kept += int(keep.sum().item())
        total += keep.numel()
    return float(kept / max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="BCTI cross-method ablations")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--forget-id", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--localization-source", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(cfg, args.mode)
    ensure_empty(args.out_dir)
    random.seed(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))

    pipeline_cfg = load_config(Path(cfg["pipeline_config"]))
    base_cfg = load_config(Path(pipeline_cfg["base_config"]))
    facts = load_facts(base_cfg["data"]["facts_path"])
    all_forget, all_retain = split_facts(facts)
    target = [fact for fact in all_forget if fact.id == args.forget_id]
    if len(target) != 1:
        raise ValueError(f"expected exactly one forget fact for {args.forget_id}")
    splits = build_splits(target, all_retain, cfg["data_split"])
    evaluation = cfg["evaluation"]

    ablation_cfg = cfg.get("ablation", {})
    localization_policy = str(ablation_cfg.get("localization_policy", "bcti"))
    forget_objective = str(ablation_cfg.get("forget_objective", "bcti"))
    rome_trace = pd.DataFrame()
    if args.localization_source is None:
        layers, trace, site_summary, layer_summary = discover_causal_layers(
            base_cfg, splits, cfg["localization"]
        )
    else:
        layers, trace, site_summary, layer_summary = load_causal_layers(
            args.localization_source, cfg["localization"]
        )
    if localization_policy == "rome":
        if args.localization_source is not None:
            raise ValueError("ROME localization must run fresh; --localization-source is unsupported")
        print("[cross-methods] loading TransformerLens model for ROME tracing", flush=True)
        rome_model = load_from_config(base_cfg)
        try:
            rome_cfg = cfg["rome"]
            layers, rome_trace, layer_summary = discover_rome_layers(
                rome_model,
                splits["trace_forget"],
                config=RomeTraceConfig(
                    noise_multiplier=float(rome_cfg["noise_multiplier"]),
                    noise_samples=int(rome_cfg["noise_samples"]),
                    layer_count=int(rome_cfg["layer_count"]),
                ),
                seed=int(cfg.get("seed", 0)),
                bcti_layer_summary=layer_summary,
            )
        finally:
            del rome_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    causal_audit_cfg = cfg["evaluation"].get("causal_audit", {})
    audit_layers = select_method_independent_audit_layers(
        layer_summary, causal_audit_cfg
    )
    audit_specs = build_causal_intervention_specs(
        site_summary,
        audit_layers,
        [str(value) for value in cfg["localization"]["corrupt_strategies"]],
    )
    if not audit_specs:
        raise RuntimeError("method-independent causal audit produced no sites")

    trace.to_csv(args.out_dir / "02_fit_causal_trace.csv", index=False, encoding="utf-8-sig")
    if not rome_trace.empty:
        rome_trace.to_csv(
            args.out_dir / "02b_rome_forward_raw_ie_trace.csv",
            index=False,
            encoding="utf-8-sig",
        )
    site_summary.to_csv(args.out_dir / "03_site_scores.csv", index=False, encoding="utf-8-sig")
    layer_summary.to_csv(args.out_dir / "04_layer_scores.csv", index=False, encoding="utf-8-sig")

    print(
        f"[cross-methods] localization={localization_policy} "
        f"objective={forget_objective} selected layers: {layers}; loading HF model",
        flush=True,
    )
    model, tokenizer, device = load_hf(base_cfg)
    all_eval_facts = (
        splits["fit_forget"]
        + splits["fit_retain"]
        + splits["validation_forget"]
        + splits["validation_retain"]
        + splits["test_forget"]
        + splits["test_retain"]
    )
    base_cache = build_base_cache(model, tokenizer, all_eval_facts, device)
    all_forget_candidates = [
        *splits["fit_forget"],
        *splits["validation_forget"],
        *splits["test_forget"],
    ]
    forget_known = {
        (fact.prompt, fact.answer): base_prompt_is_known(
            base_cache[(fact.prompt, fact.answer)], evaluation
        )
        for fact in all_forget_candidates
    }
    forget_base_scores = {
        (fact.prompt, fact.answer): float(
            base_cache[(fact.prompt, fact.answer)]["answer_score"]
        )
        for fact in all_forget_candidates
    }
    split_audit = rebalance_forget_splits_by_knowledge(
        splits,
        forget_known,
        forget_base_scores,
        min_fit_per_direction=int(
            cfg["data_split"]["min_known_fit_per_direction"]
        ),
        min_eval_per_direction=int(
            cfg["data_split"]["min_known_eval_per_direction"]
        ),
        fail_on_unmet_direction_quotas=bool(
            cfg["data_split"].get("fail_on_unmet_direction_quotas", False)
        ),
    )
    if not bool(split_audit["all_partitions_directionally_evaluable"]):
        print(
            "[cross-methods] WARNING: direction quotas cannot all be met: "
            f"{split_audit['known_direction_counts']}",
            flush=True,
        )
    manifest(splits).to_csv(
        args.out_dir / "01_split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    eligible_fit_forget, eligibility_frame = training_eligibility(
        list(splits["fit_forget"]), base_cache, evaluation
    )
    fit_direction_counts = {
        direction: sum(
            causal_direction(fact) == direction for fact in eligible_fit_forget
        )
        for direction in ("forward", "reverse")
    }
    bidirectional_trainable = all(value > 0 for value in fit_direction_counts.values())
    if not bidirectional_trainable:
        print(
            "[cross-methods] WARNING: target is not bidirectionally trainable: "
            f"{fit_direction_counts}",
            flush=True,
        )
    eligibility_frame.to_csv(
        args.out_dir / "01b_training_eligibility.csv", index=False, encoding="utf-8-sig"
    )

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
    initial_adapter_state = snapshot_adapters(named_parameters)
    parameters = [parameter for _, parameter in named_parameters]
    causal_weights, causal_layer_weights = causal_parameter_weights(
        named_parameters,
        layer_summary,
        floor=float(cfg["causal_training"]["layer_weight_floor"]),
        power=float(cfg["causal_training"]["layer_weight_power"]),
    )
    selected_risks = layer_summary[layer_summary["layer"].isin(layers)].copy()
    retain_risk_threshold = float(selected_risks["retain_risk"].quantile(
        float(cfg["causal_training"].get("hybrid_retain_risk_quantile", 0.5))
    ))
    high_retain_risk_layers = set(
        selected_risks.loc[
            selected_risks["retain_risk"] >= retain_risk_threshold, "layer"
        ].astype(int).tolist()
    )
    retain_null_parameter_mask = [
        any(f"layers.{layer}." in name for layer in high_retain_risk_layers)
        for name, _ in named_parameters
    ]
    safe_layer_count = max(
        1,
        math.ceil(
            len(layers)
            * float(cfg["causal_training"]["auxiliary_safe_layer_fraction"])
        ),
    )
    auxiliary_safe_layers = set(
        selected_risks.sort_values(
            ["retain_risk", "target_causal_score"],
            ascending=[True, False],
        ).head(safe_layer_count)["layer"].astype(int).tolist()
    )
    auxiliary_unsafe_parameter_mask = [
        not any(f"layers.{layer}." in name for layer in auxiliary_safe_layers)
        for name, _ in named_parameters
    ]
    adapter_devices = {str(parameter.device) for parameter in parameters}
    if adapter_devices != {str(torch.device(device))}:
        raise RuntimeError(f"LoRA adapters span unexpected devices: {sorted(adapter_devices)}")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("LoRA adapter parameters must remain float32")
    trainable_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg["training"]["learning_rate"]),
        betas=tuple(float(value) for value in cfg["training"]["betas"]),
        eps=float(cfg["training"]["adam_epsilon"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    selection = evaluation["constraints"]
    limits = Constraints(
        min_forget_relative_drop=float(selection["min_forget_relative_drop"]),
        min_strong_forget_relative_drop=float(
            selection["min_strong_forget_relative_drop"]
        ),
        max_forget_worst_score=float(selection["max_forget_worst_answer_score"]),
        max_forget_increase=float(selection["max_forget_answer_score_increase"]),
        min_eligible_forget_prompts=int(selection["min_eligible_forget_prompts"]),
        min_eligible_forget_directions=int(
            selection["min_eligible_forget_directions"]
        ),
        min_eligible_forget_prompts_per_direction=int(
            selection["min_eligible_forget_prompts_per_direction"]
        ),
        min_direction_relative_drop=float(
            selection["min_direction_relative_drop"]
        ),
        min_strong_direction_relative_drop=float(
            selection["min_strong_direction_relative_drop"]
        ),
        max_retain_kl_mean=float(selection["retain_kl_mean_budget"]),
        max_retain_kl_tail=float(selection["retain_kl_tail_budget"]),
        max_retain_kl_max=float(selection["retain_kl_max_budget"]),
        max_retain_drop=float(selection["retain_max_drop"]),
    )
    training = cfg["training"]
    epsilon = float(training["probability_epsilon"])
    tail_fraction = float(base_cfg["dpcu"].get("forget_tail_fraction", 0.4))
    validation_facts = splits["validation_forget"] + splits["validation_retain"]
    validation_history: list[dict] = []
    training_history: list[dict] = []

    evaluate_kwargs = {
        "min_forget_base_answer_score": float(evaluation["min_forget_base_answer_score"]),
        "require_base_greedy_exact": bool(evaluation["require_base_greedy_exact"]),
    }

    causal_training = cfg["causal_training"]
    fit_forget, calibration_forget, crossfit_frame = partition_direction_crossfit(
        eligible_fit_forget,
        base_cache,
        float(causal_training["direction_calibration_fraction"]),
        int(causal_training["direction_calibration_max_per_direction"]),
        int(causal_training["preferred_optimization_prompts_per_direction"]),
    )
    crossfit_frame.to_csv(
        args.out_dir / "01c_direction_crossfit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fit_retain = list(splits["fit_retain"])
    if not fit_forget or not calibration_forget or not fit_retain:
        raise RuntimeError("training split is empty")

    causal_specs = build_causal_intervention_specs(
        site_summary,
        layers,
        [str(value) for value in cfg["localization"]["corrupt_strategies"]],
    )
    if not causal_specs:
        raise RuntimeError("selected layers produced no causal objective sites")
    causal_gap_epsilon = float(cfg["localization"]["gap_epsilon"])
    causal_quantile = float(cfg["localization"]["robust_quantile"])
    causal_threshold = float(cfg["localization"]["causal_effect_threshold"])
    causal_eval_facts = list(eligible_fit_forget)
    baseline_causal_eval = evaluate_causal_objective(
        model,
        tokenizer,
        causal_eval_facts,
        causal_specs,
        device,
        causal_gap_epsilon,
    )
    baseline_causal_summary = summarize_causal_objective(
        baseline_causal_eval,
        quantile=causal_quantile,
        causal_threshold=causal_threshold,
        epsilon=causal_gap_epsilon,
    )
    calibration_keys = {
        (fact.prompt, fact.answer) for fact in calibration_forget
    }
    baseline_calibration_causal = baseline_causal_eval[
        baseline_causal_eval.apply(
            lambda row: (row["prompt"], row["answer"]) in calibration_keys,
            axis=1,
        )
    ].copy()
    baseline_causal_lookup = causal_baseline_lookup(baseline_causal_eval)

    
    causal_validation_facts, causal_validation_eligibility = training_eligibility(
        list(splits["validation_forget"]), base_cache, evaluation
    )
    causal_validation_eligibility.to_csv(
        args.out_dir / "01d_causal_validation_eligibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_direction_counts = {
        direction: sum(causal_direction(fact) == direction for fact in causal_validation_facts)
        for direction in ("forward", "reverse")
    }
    if any(count == 0 for count in validation_direction_counts.values()):
        raise RuntimeError(
            "BCTI causal checkpoint selection requires known held-out probes in both directions; "
            f"got {validation_direction_counts}"
        )
    baseline_validation_causal = evaluate_causal_objective(
        model, tokenizer, causal_validation_facts, causal_specs, device,
        causal_gap_epsilon,
    )
    baseline_validation_causal_summary = summarize_causal_objective(
        baseline_validation_causal,
        quantile=causal_quantile,
        causal_threshold=causal_threshold,
        epsilon=causal_gap_epsilon,
    )

    
    
    causal_test_facts, causal_test_eligibility = training_eligibility(
        list(splits["test_forget"]), base_cache, evaluation
    )
    causal_test_eligibility.to_csv(
        args.out_dir / "01e_causal_test_audit_eligibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    causal_test_direction_counts = {
        direction: sum(causal_direction(fact) == direction for fact in causal_test_facts)
        for direction in ("forward", "reverse")
    }
    def validate(step: int) -> tuple[dict, dict[str, torch.Tensor]]:
        
        
        
        frame = evaluate(
            model, tokenizer, validation_facts, base_cache, device, epsilon,
            **evaluate_kwargs
        )
        diagnostic_summary = summarize(frame, tail_fraction)
        current_causal = evaluate_causal_objective(
            model, tokenizer, causal_validation_facts, causal_specs, device,
            causal_gap_epsilon, fixed_baseline=baseline_validation_causal,
        )
        causal_summary = summarize_causal_objective(
            current_causal,
            quantile=causal_quantile,
            causal_threshold=causal_threshold,
            epsilon=causal_gap_epsilon,
        )
        row = {
            "step": int(step),
            **diagnostic_summary,
            
            
            "causal_target_score": max(
                causal_summary["forward_causal_score"],
                causal_summary["reverse_causal_score"],
            ),
            "causal_direction_macro_score": 0.5 * (
                causal_summary["forward_causal_score"]
                + causal_summary["reverse_causal_score"]
            ),
            "causal_direction_harmonic_score": causal_summary["target_causal_score"],
            "causal_forward_score": causal_summary["forward_causal_score"],
            "causal_reverse_score": causal_summary["reverse_causal_score"],
            "causal_forward_coverage": causal_summary["forward_causal_coverage"],
            "causal_reverse_coverage": causal_summary["reverse_causal_coverage"],
            "causal_relative_drop": (
                max(
                    baseline_validation_causal_summary["forward_causal_score"],
                    baseline_validation_causal_summary["reverse_causal_score"],
                )
                - max(
                    causal_summary["forward_causal_score"],
                    causal_summary["reverse_causal_score"],
                )
            ) / max(
                max(
                    baseline_validation_causal_summary["forward_causal_score"],
                    baseline_validation_causal_summary["reverse_causal_score"],
                ),
                causal_gap_epsilon,
            ),
        }
        row["causal_threshold_pass"] = bool(
            training.get("causal_early_stop_threshold") is not None
            and float(row["causal_target_score"])
            <= float(training["causal_early_stop_threshold"])
        )
        row["answer_gate_min_direction_pass"] = bool(
            float(row["forget_min_direction_relative_drop"])
            >= float(training["causal_gate_min_direction_relative_drop"])
        )
        row["answer_gate_worst_score_pass"] = bool(
            float(row["forget_worst_cut_score"])
            <= float(training["causal_gate_max_worst_answer_score"])
        )
        row["causal_answer_joint_gate_pass"] = causal_answer_gate_pass(
            causal_score=float(row["causal_target_score"]),
            causal_threshold=training.get("causal_early_stop_threshold"),
            min_direction_relative_drop=float(
                row["forget_min_direction_relative_drop"]
            ),
            required_min_direction_relative_drop=float(
                training["causal_gate_min_direction_relative_drop"]
            ),
            worst_answer_score=float(row["forget_worst_cut_score"]),
            maximum_worst_answer_score=float(
                training["causal_gate_max_worst_answer_score"]
            ),
        )
        checkpoint_status = score(diagnostic_summary, limits)
        for key, value in checkpoint_status.items():
            row[f"checkpoint_{key}"] = value
        validation_history.append(row)
        return row, snapshot_adapters(named_parameters)

    best_row, best_state = validate(0)
    best_key = causal_gated_checkpoint_priority(
        best_row,
        training.get("causal_early_stop_threshold"),
        float(training["checkpoint_bottleneck_slack"]),
    )
    causal_site_stream = CausalInterventionStream(causal_specs, int(cfg.get("seed", 0)) + 307)

    seed = int(cfg.get("seed", 0))
    baseline_prompt_causal = (
        baseline_causal_eval.groupby(["prompt", "answer"])["normalized_restore"]
        .mean()
        .to_dict()
    )
    initial_hardness = {
        (fact.prompt, fact.answer): dual_signal_hardness(
            float(baseline_prompt_causal.get((fact.prompt, fact.answer), 0.0)),
            float(baseline_prompt_causal.get((fact.prompt, fact.answer), 0.0)),
            float(base_cache[(fact.prompt, fact.answer)]["answer_score"]),
            float(base_cache[(fact.prompt, fact.answer)]["answer_score"]),
            answer_weight=float(causal_training["answer_hardness_weight"]),
            epsilon=epsilon,
        )
        for fact in fit_forget
    }
    direction_answer_residual_ema = {
        direction: sum(
            float(base_cache[(fact.prompt, fact.answer)]["answer_score"])
            for fact in fit_forget if causal_direction(fact) == direction
        ) / max(
            sum(causal_direction(fact) == direction for fact in fit_forget), 1
        )
        for direction in ("forward", "reverse")
    }
    answer_direction_cap_fractions = answer_residual_cap_fractions(
        direction_answer_residual_ema,
        minimum=float(causal_training["answer_direction_min_cap_fraction"]),
        epsilon=epsilon,
    )
    forget_stream = AdaptiveDirectionalBatchStream(
        fit_forget,
        int(training["forget_batch_size"]),
        seed + 101,
        initial_hardness,
        int(causal_training["hard_example_every"]),
    )
    neighbor_scores = causal_neighbor_scores(fit_retain, trace, layers)
    retain_stream = CausalNeighborRetainStream(
        fit_retain,
        int(training["retain_batch_size"]),
        seed + 211,
        neighbor_scores,
        float(causal_training["causal_neighbor_retain_fraction"]),
        float(causal_training["causal_neighbor_pool_fraction"]),
    )
    parameter_names = [name for name, _ in named_parameters]
    evaluations_without_improvement = 0
    stopped_early = False
    early_stop_reason: str | None = None
    causal_threshold_hits = 0
    causal_answer_gate_hits = 0
    causal_stop_threshold = training.get("causal_early_stop_threshold")
    causal_stop_patience = int(training.get("causal_early_stop_patience_evals", 1))
    final_training_step = 0
    preferred_direction_support = max(
        int(causal_training["preferred_optimization_prompts_per_direction"]), 1
    )
    optimization_direction_counts = {
        direction: sum(causal_direction(fact) == direction for fact in fit_forget)
        for direction in ("forward", "reverse")
    }
    support_confidence_floor = float(
        causal_training.get("trace_direction_fallback_min_confidence", 0.0)
    )
    direction_support_confidence = {
        direction: max(
            support_confidence_floor,
            min(1.0, optimization_direction_counts[direction] / preferred_direction_support),
        )
        for direction in ("forward", "reverse")
    }
    direction_trace_weights = trace_direction_weights(
        layer_summary,
        layers,
        power=float(causal_training["direction_difficulty_power"]),
        minimum=float(causal_training["direction_weight_min"]),
        maximum=float(causal_training["direction_weight_max"]),
        epsilon=epsilon,
    )
    current_calibration_causal = evaluate_causal_objective(
        model,
        tokenizer,
        calibration_forget,
        causal_specs,
        device,
        causal_gap_epsilon,
        fixed_baseline=baseline_calibration_causal,
    )
    calibration_progress = causal_direction_progress(
        baseline_calibration_causal,
        current_calibration_causal,
        quantile=causal_quantile,
        causal_threshold=causal_threshold,
        epsilon=causal_gap_epsilon,
    )
    calibration_direction_weights = direction_weights_from_progress(
        calibration_progress,
        target_drop=float(selection["min_direction_relative_drop"]),
        power=float(causal_training["direction_difficulty_power"]),
        minimum=float(causal_training["direction_weight_min"]),
        maximum=float(causal_training["direction_weight_max"]),
        epsilon=epsilon,
    )
    direction_weights = blend_direction_weights(
        calibration_direction_weights,
        direction_trace_weights,
        direction_support_confidence,
        minimum=float(causal_training["direction_weight_min"]),
        maximum=float(causal_training["direction_weight_max"]),
        epsilon=epsilon,
    )

    for step in range(1, int(training["max_steps"]) + 1):
        final_training_step = step
        if step > 1 and (step - 1) % int(
            causal_training["direction_calibration_every"]
        ) == 0:
            current_calibration_causal = evaluate_causal_objective(
                model,
                tokenizer,
                calibration_forget,
                causal_specs,
                device,
                causal_gap_epsilon,
                fixed_baseline=baseline_calibration_causal,
            )
            calibration_progress = causal_direction_progress(
                baseline_calibration_causal,
                current_calibration_causal,
                quantile=causal_quantile,
                causal_threshold=causal_threshold,
                epsilon=causal_gap_epsilon,
            )
            calibration_direction_weights = direction_weights_from_progress(
                calibration_progress,
                target_drop=float(selection["min_direction_relative_drop"]),
                power=float(causal_training["direction_difficulty_power"]),
                minimum=float(causal_training["direction_weight_min"]),
                maximum=float(causal_training["direction_weight_max"]),
                epsilon=epsilon,
            )
            direction_weights = blend_direction_weights(
                calibration_direction_weights,
                direction_trace_weights,
                direction_support_confidence,
                minimum=float(causal_training["direction_weight_min"]),
                maximum=float(causal_training["direction_weight_max"]),
                epsilon=epsilon,
            )
        answer_direction_cap_fractions = answer_residual_cap_fractions(
            direction_answer_residual_ema,
            minimum=float(causal_training["answer_direction_min_cap_fraction"]),
            epsilon=epsilon,
        )
        forget_batch = forget_stream.next()
        retain_batch = retain_stream.next()
        optimizer.zero_grad(set_to_none=True)
        direction_batches: dict[str, list[Fact]] = {}
        for fact in forget_batch:
            direction_batches.setdefault(causal_direction(fact), []).append(fact)
        direction_losses: dict[str, torch.Tensor] = {}
        direction_causal_losses: dict[str, torch.Tensor] = {}
        direction_auxiliary_losses: dict[str, torch.Tensor] = {}
        direction_grads: dict[str, list[torch.Tensor]] = {}
        fact_hardness: dict[tuple[str, str], float] = {}
        step_causal_scores: list[float] = []
        step_auxiliary_losses: list[float] = []
        step_clean_probs: list[float] = []
        step_corrupt_probs: list[float] = []
        step_patched_probs: list[float] = []
        step_causal_gaps: list[float] = []
        step_causal_sites: list[str] = []
        step_auxiliary_gradient_scales: list[float] = []
        step_primary_gradient_norms: list[float] = []
        step_auxiliary_pre_mask_gradient_norms: list[float] = []
        step_auxiliary_raw_gradient_norms: list[float] = []
        step_auxiliary_projected_gradient_norms: list[float] = []
        step_auxiliary_gradient_norms: list[float] = []
        step_auxiliary_gradient_conflicts: list[bool] = []
        step_auxiliary_gradient_projections: list[bool] = []
        step_auxiliary_cosines_before: list[float] = []
        step_auxiliary_cosines_after: list[float] = []
        step_palu_topk_coverages: list[float] = []
        step_palu_initiating_tokens: list[int] = []
        step_palu_redundant_tokens: list[int] = []
        for direction, facts in sorted(direction_batches.items()):
            fact_losses: list[torch.Tensor] = []
            fact_causal_losses: list[torch.Tensor] = []
            fact_auxiliary_losses: list[torch.Tensor] = []
            fact_gradient_sets: list[list[torch.Tensor]] = []
            for fact in facts:
                spec = causal_site_stream.next()
                fixed = baseline_causal_lookup.get(
                    (fact.prompt, fact.answer, spec.layer, spec.site, spec.corruption)
                )
                if fixed is None:
                    raise KeyError(f"missing fixed causal baseline for {fact.id} at {spec.label}")
                with torch.no_grad():
                    causal_values = causal_restore_components(
                        model,
                        tokenizer,
                        fact,
                        spec,
                        device,
                        causal_gap_epsilon,
                        fixed_clean_prob=fixed[0],
                        fixed_corrupt_prob=fixed[1],
                    )
                causal_loss = causal_values["score"]
                if forget_objective == "palu":
                    palu_cfg = cfg["palu"]
                    palu_loss, palu_log = palu_local_entropy_loss(
                        model,
                        tokenizer,
                        fact,
                        base_cache,
                        device,
                        top_k=int(palu_cfg["top_k"]),
                        initiating_tokens=int(palu_cfg["initiating_tokens"]),
                        detach_global_mean=bool(
                            palu_cfg.get("detach_global_mean", True)
                        ),
                    )
                    combined_fact_grad = apply_gradient_weights(
                        gradients(palu_loss, parameters), causal_weights
                    )
                    auxiliary_loss = palu_loss
                    auxiliary_pre_mask_grad_norm = gradient_norm(
                        combined_fact_grad
                    )
                    primary_norm = gradient_norm(combined_fact_grad)
                    auxiliary_log = {
                        "answer_auxiliary_applied_scale": 0.0,
                        "causal_primary_grad_norm": primary_norm,
                        "answer_auxiliary_raw_grad_norm": 0.0,
                        "answer_auxiliary_projected_grad_norm": 0.0,
                        "answer_auxiliary_applied_grad_norm": 0.0,
                        "answer_auxiliary_gradient_conflict": False,
                        "answer_auxiliary_gradient_projected": False,
                        "answer_auxiliary_gradient_cosine_before": 0.0,
                        "answer_auxiliary_gradient_cosine_after": 0.0,
                    }
                    loss = palu_loss
                    with torch.no_grad():
                        answer_diag = answer_complement_loss(
                            model, tokenizer, fact, device, epsilon
                        )
                    answer_probability = max(
                        0.0,
                        min(1.0, 1.0 - math.exp(-float(answer_diag.item()))),
                    )
                    step_palu_topk_coverages.append(
                        float(palu_log["palu_target_topk_coverage"])
                    )
                    step_palu_initiating_tokens.append(
                        int(palu_log["palu_initiating_tokens"])
                    )
                    step_palu_redundant_tokens.append(
                        int(palu_log["palu_redundant_sensitive_tokens"])
                    )
                else:
                    
                    
                    causal_values = causal_restore_components(
                        model,
                        tokenizer,
                        fact,
                        spec,
                        device,
                        causal_gap_epsilon,
                        fixed_clean_prob=fixed[0],
                        fixed_corrupt_prob=fixed[1],
                    )
                    causal_loss = causal_values["score"]
                    causal_grad = apply_gradient_weights(
                        gradients(causal_loss, parameters), causal_weights
                    )
                    auxiliary_loss = answer_complement_loss(
                        model, tokenizer, fact, device, epsilon
                    )
                    auxiliary_grad = apply_gradient_weights(
                        gradients(auxiliary_loss, parameters), causal_weights
                    )
                    auxiliary_pre_mask_grad_norm = gradient_norm(auxiliary_grad)
                    auxiliary_grad = tier_auxiliary_gradient_by_retain_risk(
                        auxiliary_grad,
                        auxiliary_unsafe_parameter_mask,
                        global_floor_ratio=float(
                            causal_training["auxiliary_global_layer_floor_ratio"]
                        ),
                        maximum_ratio=float(
                            causal_training["max_answer_auxiliary_gradient_ratio"]
                        ),
                    )
                    combined_fact_grad, auxiliary_log = (
                        combine_primary_and_auxiliary_gradients(
                            causal_grad,
                            auxiliary_grad,
                            primary_weight=float(
                                causal_training["causal_score_weight"]
                            ),
                            auxiliary_weight=float(
                                causal_training["answer_complement_weight"]
                            ),
                            max_auxiliary_ratio=float(
                                causal_training[
                                    "max_answer_auxiliary_gradient_ratio"
                                ]
                            )
                            * float(answer_direction_cap_fractions[direction]),
                            epsilon=float(training["gradient_epsilon"]),
                        )
                    )
                    loss = (
                        float(causal_training["causal_score_weight"])
                        * causal_loss
                        + float(causal_training["answer_complement_weight"])
                        * float(
                            auxiliary_log["answer_auxiliary_applied_scale"]
                        )
                        * auxiliary_loss
                    )
                    auxiliary_value_for_answer = float(
                        auxiliary_loss.detach().item()
                    )
                    answer_probability = max(
                        0.0,
                        min(
                            1.0,
                            1.0 - math.exp(-auxiliary_value_for_answer),
                        ),
                    )
                fact_losses.append(loss)
                fact_causal_losses.append(causal_loss)
                fact_auxiliary_losses.append(auxiliary_loss)
                fact_gradient_sets.append(combined_fact_grad)
                causal_value = float(causal_loss.detach().item())
                step_causal_scores.append(causal_value)
                auxiliary_value = float(auxiliary_loss.detach().item())
                fact_hardness[(fact.prompt, fact.answer)] = dual_signal_hardness(
                    causal_value,
                    float(
                        baseline_prompt_causal.get(
                            (fact.prompt, fact.answer), 0.0
                        )
                    ),
                    answer_probability,
                    float(base_cache[(fact.prompt, fact.answer)]["answer_score"]),
                    answer_weight=float(
                        causal_training["answer_hardness_weight"]
                    ),
                    epsilon=epsilon,
                )
                ema_decay = float(causal_training["answer_direction_ema_decay"])
                direction_answer_residual_ema[direction] = (
                    ema_decay * direction_answer_residual_ema[direction]
                    + (1.0 - ema_decay) * answer_probability
                )
                step_auxiliary_losses.append(auxiliary_value)
                step_clean_probs.append(
                    float(causal_values["clean_prob"].detach().item())
                )
                step_corrupt_probs.append(
                    float(causal_values["corrupt_prob"].detach().item())
                )
                step_patched_probs.append(
                    float(causal_values["patched_prob"].detach().item())
                )
                step_causal_gaps.append(
                    float(causal_values["corruption_gap"].detach().item())
                )
                step_causal_sites.append(spec.label)
                step_auxiliary_gradient_scales.append(
                    float(auxiliary_log["answer_auxiliary_applied_scale"])
                )
                step_primary_gradient_norms.append(
                    float(auxiliary_log["causal_primary_grad_norm"])
                )
                step_auxiliary_pre_mask_gradient_norms.append(
                    float(auxiliary_pre_mask_grad_norm)
                )
                step_auxiliary_raw_gradient_norms.append(
                    float(auxiliary_log["answer_auxiliary_raw_grad_norm"])
                )
                step_auxiliary_projected_gradient_norms.append(
                    float(auxiliary_log["answer_auxiliary_projected_grad_norm"])
                )
                step_auxiliary_gradient_norms.append(
                    float(auxiliary_log["answer_auxiliary_applied_grad_norm"])
                )
                step_auxiliary_gradient_conflicts.append(
                    bool(auxiliary_log["answer_auxiliary_gradient_conflict"])
                )
                step_auxiliary_gradient_projections.append(
                    bool(auxiliary_log["answer_auxiliary_gradient_projected"])
                )
                step_auxiliary_cosines_before.append(
                    float(
                        auxiliary_log[
                            "answer_auxiliary_gradient_cosine_before"
                        ]
                    )
                )
                step_auxiliary_cosines_after.append(
                    float(
                        auxiliary_log[
                            "answer_auxiliary_gradient_cosine_after"
                        ]
                    )
                )
            direction_losses[direction] = torch.stack(fact_losses).mean()
            direction_causal_losses[direction] = torch.stack(fact_causal_losses).mean()
            direction_auxiliary_losses[direction] = torch.stack(fact_auxiliary_losses).mean()
            direction_grads[direction] = mean_gradients(fact_gradient_sets)
        for fact in forget_batch:
            forget_stream.update(
                fact, fact_hardness[(fact.prompt, fact.answer)]
            )
        
        
        weight_sum = sum(direction_weights.values())
        f_loss = sum(
            direction_losses[direction] * direction_weights[direction]
            for direction in direction_losses
        ) / weight_sum
        f_causal_loss = sum(
            direction_causal_losses[direction] * direction_weights[direction]
            for direction in direction_causal_losses
        ) / weight_sum
        f_auxiliary_loss = sum(
            direction_auxiliary_losses[direction] * direction_weights[direction]
            for direction in direction_auxiliary_losses
        ) / weight_sum
        forget_grad, direction_gradient_log = combine_direction_gradients(
            direction_grads,
            direction_weights,
            policy=str(causal_training.get("direction_gradient_policy", "weighted_mean")),
            epsilon=float(training["gradient_epsilon"]),
        )
        retain_batch_max_drop = 0.0
        retain_pressure = 1.0
        if args.mode == "forget_only":
            r_loss = torch.zeros((), device=device, dtype=torch.float32)
            retain_grad = _zeros_like(parameters)
        else:
            
            
            
            retain_losses: list[torch.Tensor] = []
            retain_gradient_sets: list[list[torch.Tensor]] = []
            retain_drops: list[float] = []
            retain_kl_values: list[float] = []
            retain_margin_values: list[float] = []
            for fact in retain_batch:
                loss, score_drop, kl_value, margin_value = retain_loss_and_drop(
                    model,
                    tokenizer,
                    fact,
                    base_cache,
                    device,
                    epsilon,
                    answer_margin=float(causal_training["retain_answer_margin"]),
                    margin_weight=float(causal_training["retain_answer_margin_weight"]),
                )
                retain_losses.append(loss)
                retain_drops.append(score_drop)
                retain_kl_values.append(kl_value)
                retain_margin_values.append(margin_value)
                retain_gradient_sets.append(gradients(loss, parameters))
            r_loss = torch.stack(retain_losses).mean()
            retain_grad = mean_gradients(retain_gradient_sets)
            retain_batch_max_drop = max(retain_drops, default=0.0)
            
            
            
            retain_pressure = 1.0
        forward_grad = direction_grads.get("forward")
        reverse_grad = direction_grads.get("reverse")
        combined, gradient_log = combine_gradients(
            forget_grad,
            retain_grad,
            parameter_names=parameter_names,
            mode=args.mode,
            retain_weight=float(training["retain_weight"]) * retain_pressure,
            normalization=str(training["gradient_normalization"]),
            epsilon=float(training["gradient_epsilon"]),
            minimum_retain_gradient_ratio=float(
                causal_training.get("retain_gradient_min_ratio", 0.0)
            ),
            retain_null_parameter_mask=retain_null_parameter_mask,
        )
        if not all(torch.isfinite(value).all() for value in combined):
            raise FloatingPointError(f"non-finite combined gradient at step {step}")
        pre_step = snapshot_adapters(named_parameters)
        for parameter, value in zip(parameters, combined):
            parameter.grad = value.to(device=parameter.device, dtype=parameter.dtype)
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
        if not torch.isfinite(total_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            restore_adapters(named_parameters, pre_step)
            raise FloatingPointError(f"non-finite adapter parameter at step {step}; update rolled back")
        training_history.append(
            {
                "step": step,
                "forget_ids": "|".join(fact.id for fact in forget_batch),
                "retain_ids": "|".join(fact.id for fact in retain_batch),
                "forget_loss": float(f_loss.detach().item()),
                "causal_forget_loss": float(f_causal_loss.detach().item()),
                "answer_complement_auxiliary_loss": float(
                    f_auxiliary_loss.detach().item()
                ),
                "causal_objective_sites": "|".join(step_causal_sites),
                "causal_score_batch_mean": sum(step_causal_scores) / len(step_causal_scores),
                "forget_objective": forget_objective,
                "palu_target_topk_coverage_mean": (
                    sum(step_palu_topk_coverages) / len(step_palu_topk_coverages)
                    if step_palu_topk_coverages else None
                ),
                "palu_initiating_tokens_mean": (
                    sum(step_palu_initiating_tokens) / len(step_palu_initiating_tokens)
                    if step_palu_initiating_tokens else None
                ),
                "palu_redundant_sensitive_tokens_mean": (
                    sum(step_palu_redundant_tokens) / len(step_palu_redundant_tokens)
                    if step_palu_redundant_tokens else None
                ),
                "dual_signal_hardness_batch_mean": sum(fact_hardness.values()) / len(fact_hardness),
                "dual_signal_hardness_batch_max": max(fact_hardness.values()),
                "causal_clean_prob_batch_mean": sum(step_clean_probs) / len(step_clean_probs),
                "causal_corrupt_prob_batch_mean": sum(step_corrupt_probs) / len(step_corrupt_probs),
                "causal_patched_prob_batch_mean": sum(step_patched_probs) / len(step_patched_probs),
                "causal_corruption_gap_batch_mean": sum(step_causal_gaps) / len(step_causal_gaps),
                "causal_primary_grad_norm_mean": sum(step_primary_gradient_norms) / len(step_primary_gradient_norms),
                "answer_auxiliary_pre_retain_tiering_grad_norm_mean": sum(step_auxiliary_pre_mask_gradient_norms) / len(step_auxiliary_pre_mask_gradient_norms),
                "answer_auxiliary_raw_grad_norm_mean": sum(step_auxiliary_raw_gradient_norms) / len(step_auxiliary_raw_gradient_norms),
                "answer_auxiliary_unsafe_parameter_fraction": sum(auxiliary_unsafe_parameter_mask) / max(len(auxiliary_unsafe_parameter_mask), 1),
                "answer_auxiliary_projected_grad_norm_mean": sum(step_auxiliary_projected_gradient_norms) / len(step_auxiliary_projected_gradient_norms),
                "answer_auxiliary_applied_grad_norm_mean": sum(step_auxiliary_gradient_norms) / len(step_auxiliary_gradient_norms),
                "answer_auxiliary_applied_scale_mean": sum(step_auxiliary_gradient_scales) / len(step_auxiliary_gradient_scales),
                "answer_auxiliary_gradient_conflict_fraction": sum(step_auxiliary_gradient_conflicts) / len(step_auxiliary_gradient_conflicts),
                "answer_auxiliary_gradient_projected_fraction": sum(step_auxiliary_gradient_projections) / len(step_auxiliary_gradient_projections),
                "answer_auxiliary_gradient_cosine_before_mean": sum(step_auxiliary_cosines_before) / len(step_auxiliary_cosines_before),
                "answer_auxiliary_gradient_cosine_after_mean": sum(step_auxiliary_cosines_after) / len(step_auxiliary_cosines_after),
                "forward_forget_loss": (
                    float(direction_losses["forward"].detach().item())
                    if "forward" in direction_losses
                    else None
                ),
                "reverse_forget_loss": (
                    float(direction_losses["reverse"].detach().item())
                    if "reverse" in direction_losses
                    else None
                ),
                "forward_causal_forget_loss": (
                    float(direction_causal_losses["forward"].detach().item())
                    if "forward" in direction_causal_losses else None
                ),
                "reverse_causal_forget_loss": (
                    float(direction_causal_losses["reverse"].detach().item())
                    if "reverse" in direction_causal_losses else None
                ),
                "forward_answer_auxiliary_loss": (
                    float(direction_auxiliary_losses["forward"].detach().item())
                    if "forward" in direction_auxiliary_losses else None
                ),
                "reverse_answer_auxiliary_loss": (
                    float(direction_auxiliary_losses["reverse"].detach().item())
                    if "reverse" in direction_auxiliary_losses else None
                ),
                "forward_direction_weight": direction_weights.get("forward"),
                "reverse_direction_weight": direction_weights.get("reverse"),
                "forward_answer_residual_ema": direction_answer_residual_ema.get("forward"),
                "reverse_answer_residual_ema": direction_answer_residual_ema.get("reverse"),
                "forward_answer_auxiliary_cap_fraction": answer_direction_cap_fractions.get("forward"),
                "reverse_answer_auxiliary_cap_fraction": answer_direction_cap_fractions.get("reverse"),
                "forward_calibration_direction_weight": calibration_direction_weights.get("forward"),
                "reverse_calibration_direction_weight": calibration_direction_weights.get("reverse"),
                "forward_trace_direction_weight": direction_trace_weights.get("forward"),
                "reverse_trace_direction_weight": direction_trace_weights.get("reverse"),
                "forward_direction_support_confidence": direction_support_confidence.get("forward"),
                "reverse_direction_support_confidence": direction_support_confidence.get("reverse"),
                "forward_calibration_drop": calibration_progress.get("forward"),
                "reverse_calibration_drop": calibration_progress.get("reverse"),
                "retain_loss": float(r_loss.detach().item()),
                "retain_batch_kl_mean": (
                    sum(retain_kl_values) / len(retain_kl_values)
                    if args.mode != "forget_only"
                    else 0.0
                ),
                "retain_batch_margin_loss_mean": (
                    sum(retain_margin_values) / len(retain_margin_values)
                    if args.mode != "forget_only"
                    else 0.0
                ),
                "retain_batch_max_answer_score_drop": retain_batch_max_drop,
                "retain_pressure": retain_pressure,
                "causal_neighbor_retain_count": sum(
                    float(neighbor_scores.get((fact.prompt, fact.answer), 0.0)) > 0.0
                    for fact in retain_batch
                ),

                "forward_forget_grad_norm": (
                    gradient_norm(forward_grad) if forward_grad is not None else None
                ),
                "reverse_forget_grad_norm": (
                    gradient_norm(reverse_grad) if reverse_grad is not None else None
                ),
                "forward_sago_keep_fraction": (
                    sign_keep_fraction(forward_grad, retain_grad)
                    if forward_grad is not None
                    else None
                ),
                "reverse_sago_keep_fraction": (
                    sign_keep_fraction(reverse_grad, retain_grad)
                    if reverse_grad is not None
                    else None
                ),
                "pre_clip_total_grad_norm": float(total_norm.item()),
                "post_clip_total_grad_norm": float(
                    min(total_norm.item(), float(training["max_grad_norm"]))
                ),
                "gradient_clipped": bool(total_norm.item() > float(training["max_grad_norm"])),
                "causal_weight_min": float(min(causal_weights)),
                "causal_weight_max": float(max(causal_weights)),
                "hybrid_retain_null_parameter_fraction": float(
                    sum(retain_null_parameter_mask) / max(len(retain_null_parameter_mask), 1)
                ),
                **direction_gradient_log,
                **gradient_log,
            }
        )
        dense_steps = int(training.get("early_dense_eval_steps", 0))
        should_evaluate = (
            step <= dense_steps
            or step % int(training["eval_every"]) == 0
            or step == int(training["max_steps"])
        )
        if should_evaluate:
            row, state = validate(step)
            key = causal_gated_checkpoint_priority(
                row,
                training.get("causal_early_stop_threshold"),
                float(training["checkpoint_bottleneck_slack"]),
            )
            print(
                f"[cross-methods] mode={args.mode} step={step} "
                f"causal={row['causal_target_score']:.6f} "
                f"forward={row['causal_forward_score']:.6f} "
                f"reverse={row['causal_reverse_score']:.6f} "
                f"forget_drop={row['forget_relative_drop']:.4f} "
                f"retain_kl={row['retain_kl_mean']:.4f}",
                flush=True,
            )
            if key < best_key:
                best_key, best_row, best_state = key, row, state
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            causal_threshold_hits = causal_threshold_streak(
                causal_threshold_hits,
                float(row["causal_target_score"]),
                None if causal_stop_threshold is None else float(causal_stop_threshold),
            )
            retain_gate_required = bool(
                training.get("causal_gate_require_retain_acceptable", False)
            )
            retain_gate_pass = (
                (not retain_gate_required)
                or bool(row.get("checkpoint_retain_acceptable", False))
            )
            causal_answer_gate_hits = causal_answer_gate_streak(
                causal_answer_gate_hits if retain_gate_pass else 0,
                causal_score=float(row["causal_target_score"]),
                causal_threshold=(
                    None if causal_stop_threshold is None
                    else float(causal_stop_threshold)
                ),
                min_direction_relative_drop=float(
                    row["forget_min_direction_relative_drop"]
                ),
                required_min_direction_relative_drop=float(
                    training["causal_gate_min_direction_relative_drop"]
                ),
                worst_answer_score=float(row["forget_worst_cut_score"]),
                maximum_worst_answer_score=float(
                    training["causal_gate_max_worst_answer_score"]
                ),
            )
            row["causal_threshold_streak"] = causal_threshold_hits
            row["causal_answer_joint_gate_streak"] = causal_answer_gate_hits
            if causal_threshold_stop_reached(
                step=step,
                minimum_step=int(training["early_stop_min_steps"]),
                streak=causal_answer_gate_hits,
                patience=causal_stop_patience,
                threshold=(
                    None if causal_stop_threshold is None
                    else float(causal_stop_threshold)
                ),
            ):
                stopped_early = True
                early_stop_reason = (
                    "causal_threshold_answer_and_retain_check"
                    if retain_gate_required else "causal_threshold_and_answer_check"
                )
                print(
                    f"[cross-methods] early stop at step={step}: causal score "
                    f"<= {float(causal_stop_threshold):.6f}, minimum direction "
                    f"drop >= {float(training['causal_gate_min_direction_relative_drop']):.4f}, "
                    f"and worst answer score <= {float(training['causal_gate_max_worst_answer_score']):.4f} "
                    f"retain_gate={'on' if retain_gate_required else 'off'} "
                    f"for {causal_answer_gate_hits} evaluations",
                    flush=True,
                )
                break
            if (
                step >= int(training["early_stop_min_steps"])
                and evaluations_without_improvement
                >= int(training["no_improvement_patience_evals"])
            ):
                stopped_early = True
                early_stop_reason = "no_validation_improvement"
                print(
                    f"[cross-methods] early stop at step={step}: no validation improvement "
                    f"for {evaluations_without_improvement} evaluations",
                    flush=True,
                )
                break

    restore_adapters(named_parameters, best_state)
    fit_result = evaluate(
        model,
        tokenizer,
        splits["fit_forget"] + splits["fit_retain"],
        base_cache,
        device,
        epsilon,
        **evaluate_kwargs,
    )
    validation_result = evaluate(
        model, tokenizer, validation_facts, base_cache, device, epsilon, **evaluate_kwargs
    )
    test_result = evaluate(
        model,
        tokenizer,
        splits["test_forget"] + splits["test_retain"],
        base_cache,
        device,
        epsilon,
        **evaluate_kwargs,
    )
    fit_summary = summarize(fit_result, tail_fraction)
    validation_summary = summarize(validation_result, tail_fraction)
    test_summary = summarize(test_result, tail_fraction)
    test_status = score(test_summary, limits)
    edited_causal_eval = evaluate_causal_objective(
        model,
        tokenizer,
        causal_eval_facts,
        causal_specs,
        device,
        causal_gap_epsilon,
        fixed_baseline=baseline_causal_eval,
    )
    edited_causal_summary = summarize_causal_objective(
        edited_causal_eval,
        quantile=causal_quantile,
        causal_threshold=causal_threshold,
        epsilon=causal_gap_epsilon,
    )
    causal_score_drop = float(
        baseline_causal_summary["target_causal_score"]
        - edited_causal_summary["target_causal_score"]
    )
    causal_score_relative_drop = float(
        causal_score_drop
        / max(baseline_causal_summary["target_causal_score"], causal_gap_epsilon)
    )
    baseline_causal_output = baseline_causal_eval.copy()
    baseline_causal_output.insert(0, "phase", "baseline")
    edited_causal_output = edited_causal_eval.copy()
    edited_causal_output.insert(0, "phase", "edited")
    causal_objective_eval = pd.concat(
        [baseline_causal_output, edited_causal_output], ignore_index=True
    )
    
    
    restore_adapters(named_parameters, initial_adapter_state)
    baseline_test_causal_audit = evaluate_causal_objective(
        model, tokenizer, causal_test_facts, audit_specs, device, causal_gap_epsilon
    )
    baseline_test_causal_audit_summary = summarize_causal_objective(
        baseline_test_causal_audit,
        quantile=causal_quantile,
        causal_threshold=causal_threshold,
        epsilon=causal_gap_epsilon,
    )
    restore_adapters(named_parameters, best_state)
    edited_test_causal_audit = evaluate_causal_objective(
        model,
        tokenizer,
        causal_test_facts,
        audit_specs,
        device,
        causal_gap_epsilon,
        fixed_baseline=baseline_test_causal_audit,
    )
    edited_test_causal_audit_summary = summarize_causal_objective(
        edited_test_causal_audit,
        quantile=causal_quantile,
        causal_threshold=causal_threshold,
        epsilon=causal_gap_epsilon,
    )
    test_causal_audit_reductions = causal_audit_reductions(
        baseline_test_causal_audit_summary,
        edited_test_causal_audit_summary,
        causal_test_direction_counts,
        causal_gap_epsilon,
    )
    baseline_test_causal_output = baseline_test_causal_audit.copy()
    baseline_test_causal_output.insert(0, "phase", "baseline")
    edited_test_causal_output = edited_test_causal_audit.copy()
    edited_test_causal_output.insert(0, "phase", "edited")
    test_causal_audit_frame = pd.concat(
        [baseline_test_causal_output, edited_test_causal_output], ignore_index=True
    )
    selection_kind = "causal_hard_gate_then_validation_safety"

    pd.DataFrame([asdict(value) for value in installed]).to_csv(
        args.out_dir / "05_installed_adapters.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_history).to_csv(
        args.out_dir / "06_training_history.csv", index=False, encoding="utf-8-sig"
    )
    causal_objective_eval.to_csv(
        args.out_dir / "06b_causal_objective_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {"phase": "baseline", **baseline_causal_summary},
            {"phase": "edited", **edited_causal_summary},
        ]
    ).to_csv(
        args.out_dir / "06c_causal_objective_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_frame = pd.DataFrame(validation_history)
    validation_frame.to_csv(
        args.out_dir / "07_validation_history.csv", index=False, encoding="utf-8-sig"
    )
    validation_frame.sort_values(
        ["causal_target_score", "causal_direction_macro_score", "step"]
    ).to_csv(
        args.out_dir / "07b_causal_validation_ranked.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fit_result.to_csv(args.out_dir / "08_fit_eval.csv", index=False, encoding="utf-8-sig")
    validation_result.to_csv(
        args.out_dir / "09_validation_eval.csv", index=False, encoding="utf-8-sig"
    )
    test_result.to_csv(args.out_dir / "10_test_eval.csv", index=False, encoding="utf-8-sig")
    test_causal_audit_frame.to_csv(
        args.out_dir / "10b_test_causal_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([
        {"phase": "baseline", **baseline_test_causal_audit_summary},
        {"phase": "edited", **edited_test_causal_audit_summary},
    ]).to_csv(
        args.out_dir / "10c_test_causal_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        {
            "version": METHOD_VERSION,
            "mode": args.mode,
            "forget_id": args.forget_id,
            "selected_layers": layers,
            "adapter_info": [asdict(value) for value in installed],
            "state_dict": best_state,
            "best_validation_step": int(best_row["step"]),
            "selection_kind": selection_kind,
            "causal_layer_weights": causal_layer_weights,
            "causal_objective_specs": [asdict(value) for value in causal_specs],
            "baseline_causal_objective": baseline_causal_summary,
            "edited_causal_objective": edited_causal_summary,
            "causal_score_relative_drop": causal_score_relative_drop,
            "test_causal_audit_layers": audit_layers,
            "test_causal_audit_specs": [asdict(value) for value in audit_specs],
            "baseline_test_causal_audit": baseline_test_causal_audit_summary,
            "edited_test_causal_audit": edited_test_causal_audit_summary,
            "test_causal_audit_reductions": test_causal_audit_reductions,

            "hybrid_retain_null_high_risk_layers": sorted(high_retain_risk_layers),
            "hybrid_retain_risk_threshold": retain_risk_threshold,
            "fit_optimization_forget_ids": [fact.id for fact in fit_forget],
            "fit_calibration_forget_ids": [
                fact.id for fact in calibration_forget
            ],
            "direction_trace_weights": direction_trace_weights,
            "direction_support_confidence": direction_support_confidence,
        },
        args.out_dir / "adapter_checkpoint.pt",
    )
    summary = {
        "version": METHOD_VERSION,
        "mode": args.mode,
        "forget_id": args.forget_id,
        "selected_layers": layers,
        "trainable_parameter_count": trainable_count,
        "base_model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_step": int(best_row["step"]),
        "selection_kind": selection_kind,
        "selection_uses_general_forget_metrics": True,
        "selection_uses_retain_constraints": True,
        "selection_metrics_apply_only_after_causal_threshold": True,
        "checkpoint_bottleneck_slack": float(training["checkpoint_bottleneck_slack"]),
        "selection_policy": (
            "causal hard gate on held-out max(forward, reverse); before any pass, "
            "pure causal ranking; inside the causal-feasible set, prefer formal "
            "feasibility, then strong feasibility, then quantized worst-constraint "
            "tiers, weak-direction forgetting margin, exact violation, and causal score"
        ),
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
        "causal_early_stop_threshold": (
            None if causal_stop_threshold is None else float(causal_stop_threshold)
        ),
        "causal_early_stop_patience_evals": causal_stop_patience,
        "early_stop_requires_causal_threshold": True,
        "early_stop_answer_check_is_secondary_only": True,
        "early_stop_retain_check_is_secondary_only": bool(
            training.get("causal_gate_require_retain_acceptable", False)
        ),
        "causal_gate_min_direction_relative_drop": float(
            training["causal_gate_min_direction_relative_drop"]
        ),
        "causal_gate_max_worst_answer_score": float(
            training["causal_gate_max_worst_answer_score"]
        ),
        "final_causal_threshold_streak": int(causal_threshold_hits),
        "final_causal_answer_joint_gate_streak": int(causal_answer_gate_hits),
        "final_training_step": final_training_step,
        "data_splits_are_prompt_disjoint": True,
        "forget_holdout_is_family_stratified": False,
        "forget_holdout_origin_is_family_stratified": True,
        "forget_split_is_globally_direction_repaired": True,
        "split_direction_audit": split_audit,
        "causal_localization_uses_fit_only": True,
        "causal_trace_prompts_are_pinned_to_fit": True,
        "causal_retain_tracing_covers_all_fit_relations": bool(
            cfg["data_split"].get("trace_all_retain_relations", False)
        ),
        "ablation_name": str(ablation_cfg.get("name", "bcti_bcti")),
        "localization_policy": localization_policy,
        "forget_objective_policy": forget_objective,
        "causal_localization_is_direction_balanced": localization_policy == "bcti",
        "causal_layer_candidate_authority": (
            "rome_forward_raw_indirect_effect"
            if localization_policy == "rome"
            else "target_forget_restoration_only"
        ),
        "causal_retain_role_in_layer_selection": "secondary_safety_rank_within_target_candidates",
        "causal_layer_score_policy": (
            "rome_forward_subject_last_mlp_raw_indirect_effect"
            if localization_policy == "rome"
            else "bidirectional_harmonic_target_restoration_with_coverage"
        ),
        "causal_layer_candidate_pool_size": int(
            cfg["localization"]["candidate_pool_size"]
        ),
        "causal_layer_direction_candidate_pool_size": int(
            cfg["localization"]["candidate_layers_per_direction"]
        ),
        "causal_layer_min_layers_per_direction": int(
            cfg["localization"]["min_layers_per_direction"]
        ),
        "causal_layer_min_directional_coverage": float(
            cfg["localization"].get("min_directional_causal_coverage", 0.0)
        ),
        "causal_layer_candidate_relative_floor": float(
            cfg["localization"]["candidate_relative_floor"]
        ),
        "causal_effect_threshold": float(
            cfg["localization"]["causal_effect_threshold"]
        ),
        "causal_relation_training_is_bidirectional": bidirectional_trainable,
        "forget_objective": (
            "palu_reference_topk_prefix_local_entropy"
            if forget_objective == "palu"
            else (
                "fixed_baseline_normalized_causal_restore_primary_plus_"
                "capped_bounded_ga_probability_auxiliary"
            )
        ),
        "palu_config": (cfg.get("palu") if forget_objective == "palu" else None),
        "rome_config": (cfg.get("rome") if localization_policy == "rome" else None),
        "auxiliary_forget_loss": str(
            causal_training["auxiliary_forget_loss"]
        ),
        "causal_score_is_primary_forget_objective": forget_objective == "bcti",
        "causal_primary_gradient_norm_guaranteed_above_auxiliary": forget_objective == "bcti",
        "causal_primary_descent_protected_from_auxiliary_conflict": forget_objective == "bcti",
        "auxiliary_conflict_policy": "pcgrad_project_auxiliary_against_causal_primary",
        "causal_score_formula": "clip((patched_prob-corrupt_prob)/(clean_prob-corrupt_prob),0,1); zero_if_gap_nonpositive",
        "causal_score_forward_is_identical_before_during_after": True,
        "causal_score_backward_policy": (
            "fixed_original_clean_corrupt_denominator, stop_gradient baselines, "
            "upper_clip_straight_through"
        ),
        "causal_score_weight": float(causal_training["causal_score_weight"]),
        "answer_complement_auxiliary_weight": float(
            causal_training["answer_complement_weight"]
        ),
        "max_answer_auxiliary_gradient_ratio": float(
            causal_training["max_answer_auxiliary_gradient_ratio"]
        ),
        "answer_auxiliary_direction_control": "fit_only_answer_residual_ema",
        "answer_direction_min_cap_fraction": float(
            causal_training["answer_direction_min_cap_fraction"]
        ),
        "answer_direction_ema_decay": float(
            causal_training["answer_direction_ema_decay"]
        ),
        "final_answer_direction_residual_ema": direction_answer_residual_ema,
        "final_answer_direction_cap_fractions": answer_direction_cap_fractions,
        "answer_auxiliary_retain_risk_tiered": True,
        "answer_auxiliary_safe_layer_fraction": float(
            causal_training["auxiliary_safe_layer_fraction"]
        ),
        "answer_auxiliary_global_layer_floor_ratio": float(
            causal_training["auxiliary_global_layer_floor_ratio"]
        ),
        "answer_auxiliary_safe_layers": sorted(auxiliary_safe_layers),
        "answer_auxiliary_unsafe_parameter_fraction": float(
            sum(auxiliary_unsafe_parameter_mask)
            / max(len(auxiliary_unsafe_parameter_mask), 1)
        ),
        "causal_objective_site_count": len(causal_specs),
        "baseline_causal_objective": baseline_causal_summary,
        "edited_causal_objective": edited_causal_summary,
        "causal_score_drop": causal_score_drop,
        "causal_score_relative_drop": causal_score_relative_drop,
        "causal_layer_weighted_training": True,
        "fit_only_direction_difficulty_weighting": True,
        "direction_gradient_policy": str(causal_training.get("direction_gradient_policy", "weighted_mean")),
        "retain_gradient_min_ratio": float(
            causal_training.get("retain_gradient_min_ratio", 0.0)
        ),
        "fit_direction_control_is_cross_fitted": True,
        "fit_direction_control_uses_trace_fallback": True,
        "direction_trace_weights": direction_trace_weights,
        "direction_support_confidence": direction_support_confidence,
        "calibration_direction_weights": calibration_direction_weights,
        "final_direction_weights": direction_weights,
        "fit_optimization_forget_count": len(fit_forget),
        "fit_calibration_forget_count": len(calibration_forget),
        "fit_optimization_direction_counts": {
            direction: sum(
                causal_direction(fact) == direction for fact in fit_forget
            )
            for direction in ("forward", "reverse")
        },
        "fit_calibration_direction_counts": {
            direction: sum(
                causal_direction(fact) == direction for fact in calibration_forget
            )
            for direction in ("forward", "reverse")
        },
        "crossfit_low_support_directions": sorted(
            crossfit_frame.loc[
                ~crossfit_frame["optimization_support_sufficient"].astype(bool),
                "direction",
            ].unique().tolist()
        ),
        "crossfit_support_is_sufficient": bool(
            crossfit_frame["optimization_support_sufficient"].all()
        ),
        "causal_hard_example_training": True,
        "hard_example_signal": "max_fit_normalized_causal_and_answer_residuals",
        "causal_hardness_is_baseline_normalized": True,
        "answer_hardness_weight": float(causal_training["answer_hardness_weight"]),
        "retain_objective": "reference_kl_plus_fit_answer_margin",
        "retain_answer_margin": float(causal_training["retain_answer_margin"]),
        "retain_answer_margin_weight": float(
            causal_training["retain_answer_margin_weight"]
        ),
        "retain_weight_is_fixed": True,
        "retain_checkpoint_gating": True,
        "retain_checkpoint_gating_only_after_causal_threshold": True,
        "baseline_causal_denominator_is_fixed": True,
        "causal_validation_direction_counts": validation_direction_counts,
        "baseline_validation_causal_objective": baseline_validation_causal_summary,
        "test_causal_audit_is_method_independent": True,
        "test_causal_audit_uses_checkpoint_selection": False,
        "test_causal_audit_layer_policy": "original_bidirectional_top_candidate_layers",
        "test_causal_audit_layers": audit_layers,
        "test_causal_audit_direction_counts": causal_test_direction_counts,
        "baseline_test_causal_audit": baseline_test_causal_audit_summary,
        "edited_test_causal_audit": edited_test_causal_audit_summary,
        "test_causal_audit_reductions": test_causal_audit_reductions,

        "causal_layer_weights": causal_layer_weights,
        "hybrid_retain_null_enabled": args.mode in {
            "retain_null_pcgrad", "retain_null_sago"
        },
        "hybrid_retain_null_high_risk_layers": sorted(high_retain_risk_layers),
        "hybrid_retain_risk_threshold": retain_risk_threshold,
        "hybrid_gradient_policy": (
            "retain_null_then_pcgrad"
            if args.mode == "retain_null_pcgrad"
            else "retain_null_then_sago"
            if args.mode == "retain_null_sago"
            else args.mode
        ),
        "eligible_fit_forget_count": len(eligible_fit_forget),
        "total_fit_forget_count": len(splits["fit_forget"]),
        "eligible_fit_forget_direction_counts": fit_direction_counts,
        "bidirectional_trainable": bidirectional_trainable,
        "early_dense_validation_steps": int(
            training.get("early_dense_eval_steps", 0)
        ),
        "evaluation": evaluation,
        "constraints": asdict(limits),
        "fit_summary": fit_summary,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "validation_status": score(validation_summary, limits),
        "test_status": test_status,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[cross-methods] complete mode={args.mode} best_step={best_row['step']} "
        f"test_forget_drop={test_summary['forget_relative_drop']:.4f} "
        f"test_retain_kl={test_summary['retain_kl_mean']:.4f} "
        f"test_feasible={test_status['feasible']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

