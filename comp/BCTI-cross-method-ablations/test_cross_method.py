from __future__ import annotations

import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

try:
    import torch
    from palu_loss import palu_local_entropy_loss
    from rome_localization import _subject_token_positions
except ModuleNotFoundError:
    torch = None
    palu_local_entropy_loss = None
    _subject_token_positions = None


class ConfigContractTests(unittest.TestCase):
    def load(self, name: str) -> dict:
        return yaml.safe_load((HERE / name).read_text(encoding="utf-8"))

    def test_three_cells_change_only_declared_module(self) -> None:
        reference = self.load("config_bcti_bcti.yaml")
        rome = self.load("config_rome_bcti.yaml")
        palu = self.load("config_bcti_palu.yaml")
        self.assertEqual(reference["ablation"]["localization_policy"], "bcti")
        self.assertEqual(reference["ablation"]["forget_objective"], "bcti")
        self.assertEqual(rome["ablation"]["localization_policy"], "rome")
        self.assertEqual(rome["ablation"]["forget_objective"], "bcti")
        self.assertEqual(palu["ablation"]["localization_policy"], "bcti")
        self.assertEqual(palu["ablation"]["forget_objective"], "palu")
        for section in ("data_split", "localization", "lora", "evaluation"):
            self.assertEqual(reference[section], rome[section])
            self.assertEqual(reference[section], palu[section])
        self.assertEqual(reference["training"], rome["training"])
        self.assertEqual(reference["training"], palu["training"])

    def test_paper_fixed_settings_and_zero_bcti_forget_weights(self) -> None:
        rome = self.load("config_rome_bcti.yaml")
        palu = self.load("config_bcti_palu.yaml")
        self.assertEqual(rome["rome"]["noise_multiplier"], 3.0)
        self.assertEqual(rome["rome"]["layer_count"], rome["localization"]["layer_count"])
        self.assertEqual(palu["palu"]["top_k"], 5000)
        self.assertEqual(palu["palu"]["initiating_tokens"], 3)
        self.assertEqual(palu["causal_training"]["causal_score_weight"], 0.0)
        self.assertEqual(palu["causal_training"]["answer_complement_weight"], 0.0)

    def test_launchers_write_under_requested_output_root(self) -> None:
        for name in (
            "submit_bcti_bcti.sh",
            "submit_rome_bcti.sh",
            "submit_bcti_palu.sh",
            "run_all.sh",
        ):
            text = (HERE / name).read_text(encoding="utf-8")
            self.assertIn("outputs/50-targets/ablation/cross-methods", text)
            self.assertNotIn("\r\n", text)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable in this client environment")
class ObjectiveUnitTests(unittest.TestCase):
    def test_palu_loss_is_finite_and_differentiable(self) -> None:
        class Tokenizer:
            def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
                ids = [1, 2] if "prompt" in text else [3, 4, 5, 6]
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 8))

            def forward(self, input_ids):
                batch, tokens = input_ids.shape
                return Output(self.bias.view(1, 1, -1).expand(batch, tokens, -1))

        class Fact:
            id = "f:test"
            prompt = "prompt"
            answer = "answer"

        distributions = torch.softmax(torch.arange(8, dtype=torch.float32), dim=0)
        base_cache = {
            (Fact.prompt, Fact.answer): {
                "distributions": distributions.repeat(4, 1)
            }
        }
        model = Model()
        loss, log = palu_local_entropy_loss(
            model,
            Tokenizer(),
            Fact(),
            base_cache,
            "cpu",
            top_k=5,
            initiating_tokens=3,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.bias.grad)
        self.assertEqual(log["palu_top_k"], 5)
        self.assertEqual(log["palu_initiating_tokens"], 3)
        self.assertEqual(log["palu_redundant_sensitive_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
