from __future__ import annotations

"""Methods npo lora utilities."""

import torch
import torch.nn.functional as F

def rwku_npo_from_log_probabilities(
    current_sequence_logp: torch.Tensor,
    reference_sequence_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute the RWKU variant of NPO."""
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("RWKU NPO beta must be positive")
    log_ratio = current_sequence_logp - reference_sequence_logp
    return (-F.logsigmoid(-beta * log_ratio)).mean()
