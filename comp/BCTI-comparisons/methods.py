from __future__ import annotations

"""Methods utilities."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


METHODS = ("ga", "grad_diff", "ga_kl", "npo", "simnpo")


@dataclass(frozen=True)
class MethodSpec:
    citation_key: str
    objective: str
    needs_retain: bool
    needs_reference: bool


METHOD_SPECS = {
    "ga": MethodSpec(
        "jang2023",
        "L = -CE_forget (gradient ascent on forget NLL)",
        False,
        False,
    ),
    "grad_diff": MethodSpec(
        "liu2022",
        "L = -CE_forget + lambda_retain * CE_retain",
        True,
        False,
    ),
    "ga_kl": MethodSpec(
        "yao2024",
        "L = -CE_forget + lambda_retain * KL(p_ref || p_theta)_retain",
        True,
        True,
    ),
    "npo": MethodSpec(
        "zhang2024",
        "L = -(2/beta) log sigmoid(-beta * log(p_theta/p_ref))",
        False,
        True,
    ),
    "simnpo": MethodSpec(
        "fan2025",
        "L = -(2/beta) log sigmoid(-(beta/|y|)log p_theta - gamma) + lambda_retain * CE_retain",
        True,
        False,
    ),
}


def ga_from_nll(forget_nll: torch.Tensor) -> torch.Tensor:
    """Compute gradient-ascent loss from negative log likelihood."""
    return -forget_nll.mean()


def grad_diff_from_nll(
    forget_nll: torch.Tensor,
    retain_nll: torch.Tensor,
    retain_weight: float,
) -> torch.Tensor:
    """Compute the forget-versus-retain gradient-difference objective."""
    return -forget_nll.mean() + float(retain_weight) * retain_nll.mean()


def ga_kl_from_values(
    forget_nll: torch.Tensor,
    retain_kl: torch.Tensor,
    retain_weight: float,
) -> torch.Tensor:
    """Combine gradient ascent with a retain KL penalty."""
    return -forget_nll.mean() + float(retain_weight) * retain_kl.mean()


def npo_from_log_probabilities(
    current_sequence_logp: torch.Tensor,
    reference_sequence_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute the negative-preference-optimization objective."""
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("NPO beta must be positive")
    log_ratio = current_sequence_logp - reference_sequence_logp
    return (-(2.0 / beta) * F.logsigmoid(-beta * log_ratio)).mean()


def simnpo_from_log_probabilities(
    current_sequence_logp: torch.Tensor,
    response_lengths: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Compute the simplified NPO objective."""
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("SimNPO beta must be positive")
    lengths = response_lengths.to(
        device=current_sequence_logp.device,
        dtype=current_sequence_logp.dtype,
    )
    if bool((lengths <= 0).any()):
        raise ValueError("SimNPO response lengths must be positive")
    margin = -(beta / lengths) * current_sequence_logp - float(gamma)
    return (-(2.0 / beta) * F.logsigmoid(margin)).mean()
