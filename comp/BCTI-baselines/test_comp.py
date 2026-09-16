from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ablation import select_ablation_layers


class AblationConfigTests(unittest.TestCase):
    def test_variants_are_derived_from_main(self) -> None:
        base = yaml.safe_load((HERE / "config_base_main.yaml").read_text(encoding="utf-8"))
        expected = {
            "config_forward_only_localization.yaml": ("forward_causal", "primary_with_answer_auxiliary", 1.0, 0.90),
            "config_reverse_only_localization.yaml": ("reverse_causal", "primary_with_answer_auxiliary", 1.0, 0.90),
            "config_causal_score_only.yaml": ("bidirectional_causal", "primary_only", 0.0, 0.0),
            "config_answer_objective_only.yaml": ("bidirectional_causal", "localization_and_diagnostic_only", 1.0, 0.0),
        }
        for filename, expected_values in expected.items():
            cfg = yaml.safe_load((HERE / filename).read_text(encoding="utf-8"))
            actual = (
                cfg["ablation"]["localization_policy"],
                cfg["ablation"]["causal_score_role"],
                cfg["causal_training"]["answer_complement_weight"],
                cfg["causal_training"]["max_answer_auxiliary_gradient_ratio"],
            )
            self.assertEqual(actual, expected_values)
            for key in base:
                if key not in {"experiment_name", "causal_training", "ablation"}:
                    self.assertEqual(cfg[key], base[key])
            for key, value in base["causal_training"].items():
                if key not in {"answer_complement_weight", "max_answer_auxiliary_gradient_ratio"}:
                    self.assertEqual(cfg["causal_training"][key], value)


class DirectionalLocalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "layer": [0, 1, 2, 3],
            "selected": [False, True, False, True],
            "forward_target_causal_score": [0.1, 0.8, 0.4, 0.7],
            "reverse_target_causal_score": [0.9, 0.2, 0.8, 0.1],
            "retain_risk": [0.1, 0.3, 0.2, 0.1],
        })
        self.cfg = {"layer_count": 2}

    def test_full_preserves_main_selection(self) -> None:
        layers, _ = select_ablation_layers(self.frame, self.cfg, "bidirectional_causal")
        self.assertEqual(layers, [1, 3])

    def test_forward_and_reverse_use_only_the_declared_direction(self) -> None:
        forward, _ = select_ablation_layers(self.frame, self.cfg, "forward_causal")
        reverse, _ = select_ablation_layers(self.frame, self.cfg, "reverse_causal")
        self.assertEqual(forward, [1, 3])
        self.assertEqual(reverse, [0, 2])




if __name__ == "__main__":
    unittest.main()
