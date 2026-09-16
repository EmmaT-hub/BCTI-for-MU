from __future__ import annotations

"""Lora utilities."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Apply a low-rank linear update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_input = self.dropout(inputs).float()
        adapter_output = F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
        return base_output + (self.scaling * adapter_output).to(base_output.dtype)


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    layer: int
    module: str
    in_features: int
    out_features: int
    rank: int


def _resolve_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def install_causal_lora(
    model: nn.Module,
    layers: list[int],
    module_suffixes: list[str],
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[AdapterInfo]:
    """Attach LoRA adapters to selected causal layers."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    n_layers = len(model.model.transformer.blocks)
    invalid = sorted({int(layer) for layer in layers if not 0 <= int(layer) < n_layers})
    if invalid:
        raise ValueError(f"invalid OLMo layers: {invalid}; n_layers={n_layers}")
    if not module_suffixes:
        raise ValueError("target_modules cannot be empty")

    installed: list[AdapterInfo] = []
    for layer in sorted(set(int(value) for value in layers)):
        for suffix in module_suffixes:
            full_name = f"model.transformer.blocks.{layer}.{suffix}"
            parent, attribute = _resolve_parent(model, full_name)
            base = getattr(parent, attribute)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"LoRA target is not nn.Linear: {full_name} ({type(base)!r})")
            wrapped = LoRALinear(
                base,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
            )
            
            
            
            wrapped.to(device=base.weight.device)
            wrapped.train(model.training)
            setattr(parent, attribute, wrapped)
            installed.append(
                AdapterInfo(
                    name=full_name,
                    layer=layer,
                    module=str(suffix),
                    in_features=base.in_features,
                    out_features=base.out_features,
                    rank=int(rank),
                )
            )
    if not installed:
        raise RuntimeError("no LoRA adapters were installed")
    return installed


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters after LoRA installation")
    unexpected = [name for name, _ in parameters if not name.endswith(("lora_a", "lora_b"))]
    if unexpected:
        raise RuntimeError(f"non-LoRA parameters are trainable: {unexpected[:5]}")
    return parameters


@torch.no_grad()
def snapshot_adapters(named_parameters: list[tuple[str, nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in named_parameters}


@torch.no_grad()
def restore_adapters(
    named_parameters: list[tuple[str, nn.Parameter]],
    state: dict[str, torch.Tensor],
) -> None:
    expected = {name for name, _ in named_parameters}
    if set(state) != expected:
        raise ValueError("adapter checkpoint keys do not match the installed adapters")
    for name, parameter in named_parameters:
        parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
