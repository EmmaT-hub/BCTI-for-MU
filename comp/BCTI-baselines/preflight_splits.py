from __future__ import annotations

"""Preflight splits utilities."""

import argparse
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
COMMON_ROOT = CODE_ROOT / "common"
for path in (COMMON_ROOT, CODE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_utils import load_config  # noqa: E402
from data_utils import load_facts, split_facts  # noqa: E402
from splits import build_splits, family  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    supported_modes = {
        "forget_only", "joint", "pcgrad", "retain_null", "sago",
        "retain_null_pcgrad", "retain_null_sago",
    }
    configured_modes = [str(value) for value in cfg.get("run_modes", [])]
    unknown = sorted(set(configured_modes) - supported_modes)
    if unknown or not configured_modes:
        raise RuntimeError(
            f"invalid run_modes: configured={configured_modes} unknown={unknown}"
        )
    pipeline_cfg = load_config(Path(cfg["pipeline_config"]))
    base_cfg = load_config(Path(pipeline_cfg["base_config"]))
    forget, retain = split_facts(load_facts(base_cfg["data"]["facts_path"]))
    for target in forget:
        splits = build_splits([target], retain, cfg["data_split"])
        trace_families = {family(fact) for fact in splits["trace_forget"]}
        if "factual" not in trace_families or "reverse" not in trace_families:
            raise RuntimeError(
                f"{target.id}: causal trace must contain factual and reverse directions"
            )
        if bool(cfg["data_split"].get("trace_all_retain_relations", False)):
            fit_relations = {
                fact.relation for fact in splits["fit_retain"] if fact.relation
            }
            trace_retain_relations = {
                fact.relation for fact in splits["trace_retain"] if fact.relation
            }
            missing_relations = fit_relations - trace_retain_relations
            if missing_relations:
                raise RuntimeError(
                    f"{target.id}: retain trace misses fit relations: "
                    f"{sorted(missing_relations)}"
                )
        counts = ", ".join(
            f"{name}={len(rows)}" for name, rows in splits.items()
        )
        print(f"[BCTI-ablation-internal] split preflight {target.id}: {counts}")


if __name__ == "__main__":
    main()

