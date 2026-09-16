from __future__ import annotations

import ast
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BCTI_CONFIG = HERE.parents[1] / "code" / "BCTI-main" / "config_early_stop.yaml"
METHODS = ("ga", "grad_diff", "ga_kl", "npo", "simnpo")


def load_resolved(method: str) -> dict:
    common = yaml.safe_load((HERE / "config_common.yaml").read_text(encoding="utf-8-sig"))
    specific = yaml.safe_load((HERE / f"config_{method}.yaml").read_text(encoding="utf-8-sig"))
    specific.pop("common_config")
    return {**common, **specific}


class PaperBaselineTests(unittest.TestCase):
    def test_shared_data_and_evaluation_match_bcti(self):
        bcti = yaml.safe_load(BCTI_CONFIG.read_text(encoding="utf-8-sig"))
        common = yaml.safe_load((HERE / "config_common.yaml").read_text(encoding="utf-8-sig"))
        self.assertEqual(common["pipeline_config"], bcti["pipeline_config"])
        self.assertEqual(common["seed"], bcti["seed"])
        self.assertEqual(common["data_split"], bcti["data_split"])
        self.assertEqual(
            common["evaluation"]["constraints"], bcti["evaluation"]["constraints"]
        )
        self.assertEqual(
            common["evaluation"]["min_forget_base_answer_score"],
            bcti["evaluation"]["min_forget_base_answer_score"],
        )
        self.assertEqual(
            common["evaluation"]["require_base_greedy_exact"],
            bcti["evaluation"]["require_base_greedy_exact"],
        )

    def test_five_configs_have_distinct_citations(self):
        citations = []
        for method in METHODS:
            cfg = load_resolved(method)
            self.assertEqual(cfg["baseline"]["method"], method)
            citations.append(cfg["baseline"]["citation_key"])
        self.assertEqual(len(citations), len(set(citations)))

    def test_reference_registry_is_one_paper_per_method(self):
        references = yaml.safe_load((HERE / "references.yaml").read_text(encoding="utf-8"))
        self.assertEqual(len(references), 5)
        papers = []
        for method in METHODS:
            citation_key = load_resolved(method)["baseline"]["citation_key"]
            entry = references[citation_key]
            self.assertTrue(entry["title"])
            self.assertTrue(entry["venue"])
            papers.append(entry["paper"])
        self.assertEqual(len(papers), len(set(papers)))

    def test_objective_formulas_and_gradients(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is unavailable in the local verification environment")
        from methods import (
            ga_from_nll,
            ga_kl_from_values,
            grad_diff_from_nll,
            npo_from_log_probabilities,
            simnpo_from_log_probabilities,
        )

        forget = torch.tensor([2.0], requires_grad=True)
        retain = torch.tensor([3.0], requires_grad=True)
        self.assertEqual(float(ga_from_nll(forget)), -2.0)
        self.assertEqual(float(grad_diff_from_nll(forget, retain, 0.5)), -0.5)
        self.assertEqual(float(ga_kl_from_values(forget, retain, 0.5)), -0.5)

        current = torch.tensor([-2.0], requires_grad=True)
        reference = torch.tensor([-1.0])
        npo = npo_from_log_probabilities(current, reference, 0.1)
        npo.backward(retain_graph=True)
        self.assertGreater(float(current.grad), 0.0)  
        current.grad.zero_()
        simnpo = simnpo_from_log_probabilities(current, torch.tensor([2]), 3.0, 0.0)
        simnpo.backward()
        self.assertGreater(float(current.grad), 0.0)  

    def test_runner_is_independent_and_parseable(self):
        source = (HERE / "run.py").read_text(encoding="utf-8-sig")
        ast.parse(source)
        for forbidden in (
            "install_causal_lora",
            "CausalIntervention",
            "run_tracing_multi",
            "PCGrad",
            "localization_source",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("rebalance_forget_splits_by_knowledge", source)
        self.assertIn("validation_priority", source)

    def test_run_all_is_sequential_and_well_formed(self):
        source = (HERE / "run_all_comparisons.sh").read_text(encoding="utf-8-sig")
        positions = [source.index(f"run_one {method}") for method in METHODS]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Select exactly TWO", source)
        self.assertIn("RUN_ID", source)
        self.assertNotRegex(source, r"run_one\s+\w+\s*&")
        self.assertNotIn("\r\n", source)


if __name__ == "__main__":
    unittest.main()
