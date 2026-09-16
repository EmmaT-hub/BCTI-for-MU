from __future__ import annotations

import ast
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BCTI_CONFIG = HERE.parents[1] / "code" / "BCTI-main" / "config_early_stop.yaml"


def resolved_config() -> dict:
    common = yaml.safe_load((HERE / "config_common.yaml").read_text(encoding="utf-8-sig"))
    specific = yaml.safe_load((HERE / "config_npo_lora.yaml").read_text(encoding="utf-8-sig"))
    specific.pop("common_config")
    for key, value in specific.items():
        if isinstance(value, dict) and isinstance(common.get(key), dict):
            common[key] = {**common[key], **value}
        else:
            common[key] = value
    return common


class NPOLoRATests(unittest.TestCase):
    def test_bcti_alignment(self):
        bcti = yaml.safe_load(BCTI_CONFIG.read_text(encoding="utf-8-sig"))
        cfg = resolved_config()
        for section in ("data_split", "localization", "lora"):
            self.assertEqual(cfg[section], bcti[section])
        self.assertEqual(cfg["pipeline_config"], bcti["pipeline_config"])
        self.assertEqual(cfg["seed"], bcti["seed"])
        self.assertEqual(cfg["evaluation"]["constraints"], bcti["evaluation"]["constraints"])
        for key in (
            "max_steps", "eval_every", "learning_rate", "forget_batch_size",
            "retain_batch_size", "max_grad_norm", "weight_decay", "betas", "adam_epsilon",
        ):
            self.assertEqual(cfg["training"][key], bcti["training"][key])
        self.assertEqual(cfg["training"]["schedule"], "constant")
        self.assertEqual(cfg["training"]["warmup_steps"], 0)

    def test_rwku_objective_scale_and_gradient(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is unavailable")
        from methods import npo_from_log_probabilities
        from methods_npo_lora import rwku_npo_from_log_probabilities

        current = torch.tensor([-2.0], requires_grad=True)
        reference = torch.tensor([-1.0])
        beta = 0.1
        rwku = rwku_npo_from_log_probabilities(current, reference, beta)
        canonical = npo_from_log_probabilities(current, reference, beta)
        self.assertAlmostEqual(float(rwku), float(canonical) * beta / 2.0, places=6)
        rwku.backward()
        self.assertGreater(float(current.grad), 0.0)

    def test_aggressive_checkpoint_tradeoff(self):
        cfg = resolved_config()
        self.assertEqual(cfg["baseline"]["retain_tradeoff_weight"], 2.0)
        source = (HERE / "run_npo_lora.py").read_text(encoding="utf-8")
        self.assertIn("aggressive_checkpoint_priority", source)

    def test_runtime_does_not_open_paper(self):
        source = (HERE / "run_npo_lora.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertNotIn(".pdf", source.lower())
        launcher = (HERE / "submit_npo_lora.sh").read_text(encoding="utf-8")
        self.assertNotIn(".pdf", launcher.lower())
        self.assertNotRegex(launcher, r"(?m)^\s*(?:bash\s+)?[^#\n]*run_all_comparisons")

    def test_localization_and_lora_are_exact_code_copies(self):
        main_source = (HERE.parents[1] / "code" / "BCTI-main" / "run.py").read_text(encoding="utf-8")
        local_source = (HERE / "localization.py").read_text(encoding="utf-8")
        main_tree = ast.parse(main_source)
        local_tree = ast.parse(local_source)
        names = {
            "normalized_restore_score", "robust_restore_score", "restore_coverage",
            "harmonic_pair", "select_causal_layers", "discover_causal_layers",
        }
        main_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in main_tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        }
        local_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in local_tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        }
        self.assertEqual(local_functions, main_functions)
        self.assertEqual(
            (HERE / "lora.py").read_bytes(),
            (HERE.parents[1] / "code" / "BCTI-main" / "lora.py").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
