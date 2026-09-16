from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from lora import LoRALinear, install_causal_lora, trainable_named_parameters


class FakeOLMoSequentialBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.att_proj = nn.Linear(8, 24, bias=False)
        self.attn_out = nn.Linear(8, 8, bias=False)
        self.ff_proj = nn.Linear(8, 32, bias=False)
        self.ff_out = nn.Linear(32, 8, bias=False)


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeOLMoSequentialBlock() for _ in range(3)])


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = FakeTransformer()


class FakeOLMoForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()


class OLMoLoRAArchitectureTests(unittest.TestCase):
    def test_only_selected_olmo_block_receives_all_four_adapters(self) -> None:
        model = FakeOLMoForCausalLM()
        targets = ["att_proj", "attn_out", "ff_proj", "ff_out"]
        installed = install_causal_lora(
            model, [1], targets, rank=2, alpha=4.0, dropout=0.0
        )
        self.assertEqual(len(installed), 4)
        self.assertEqual({item.layer for item in installed}, {1})
        self.assertEqual({item.module for item in installed}, set(targets))
        self.assertTrue(all(item.name.startswith("model.transformer.blocks.1.") for item in installed))
        for name in targets:
            self.assertIsInstance(getattr(model.model.transformer.blocks[1], name), LoRALinear)
            self.assertIsInstance(getattr(model.model.transformer.blocks[0], name), nn.Linear)
            self.assertIsInstance(getattr(model.model.transformer.blocks[2], name), nn.Linear)
        trainable = trainable_named_parameters(model)
        self.assertEqual(len(trainable), 8)
        self.assertTrue(all("model.transformer.blocks.1." in name for name, _ in trainable))
        self.assertTrue(all(name.endswith(("lora_a", "lora_b")) for name, _ in trainable))

    def test_wrapped_olmo_projections_preserve_shapes(self) -> None:
        model = FakeOLMoForCausalLM()
        install_causal_lora(
            model,
            [0],
            ["att_proj", "attn_out", "ff_proj", "ff_out"],
            rank=2,
            alpha=4.0,
            dropout=0.0,
        )
        block = model.model.transformer.blocks[0]
        for module in (block.att_proj, block.attn_out, block.ff_proj, block.ff_out):
            inputs = torch.randn(2, 3, module.base.in_features)
            outputs = module(inputs)
            self.assertEqual(outputs.shape, (2, 3, module.base.out_features))

    def test_invalid_olmo_layer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid OLMo layers"):
            install_causal_lora(
                FakeOLMoForCausalLM(),
                [3],
                ["att_proj"],
                rank=2,
                alpha=4.0,
                dropout=0.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
