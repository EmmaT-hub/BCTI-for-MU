from __future__ import annotations

"""Palu loss utilities."""

import torch


def palu_local_entropy_loss(
    model,
    tokenizer,
    fact,
    base_cache: dict,
    device: str,
    *,
    top_k: int = 5000,
    initiating_tokens: int = 3,
    detach_global_mean: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute PALU?s local entropy objective."""

    if top_k <= 0:
        raise ValueError("PALU top_k must be positive")
    if initiating_tokens <= 0:
        raise ValueError("PALU initiating_tokens must be positive")
    prompt_ids = tokenizer(
        fact.prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    answer_ids = tokenizer(
        fact.answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    if prompt_ids.numel() == 0 or answer_ids.numel() == 0:
        raise ValueError(f"empty PALU tokenization for fact {fact.id}")
    combined = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = model(input_ids=combined).logits.float()
    start = prompt_ids.shape[1] - 1
    token_count = min(int(initiating_tokens), int(answer_ids.shape[1]))
    current = logits[:, start : start + token_count, :][0]

    reference = base_cache[(fact.prompt, fact.answer)]["distributions"]
    reference = reference[:token_count].to(device=device, dtype=torch.float32)
    k = min(int(top_k), int(reference.shape[-1]))
    top_indices = torch.topk(reference, k=k, dim=-1).indices
    selected_logits = current.gather(dim=-1, index=top_indices)
    target = current.mean(dim=-1, keepdim=True)
    if detach_global_mean:
        target = target.detach()
    loss = (selected_logits - target).pow(2).mean()
    target_ids = answer_ids[0, :token_count]
    target_coverage = float(
        (top_indices == target_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()
    )
    return loss, {
        "palu_top_k": int(k),
        "palu_initiating_tokens": int(token_count),
        "palu_sensitive_tokens": int(answer_ids.shape[1]),
        "palu_redundant_sensitive_tokens": int(answer_ids.shape[1] - token_count),
        "palu_target_topk_coverage": target_coverage,
        "palu_detach_global_mean": int(bool(detach_global_mean)),
    }
