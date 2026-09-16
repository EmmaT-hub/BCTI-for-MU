from __future__ import annotations

"""Rome localization utilities."""

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class RomeTraceConfig:
    noise_multiplier: float = 3.0
    noise_samples: int = 10
    layer_count: int = 4


def _first_token_id(model, answer: str) -> int:
    ids = model.to_tokens(answer, prepend_bos=False)
    if ids.numel() == 0:
        raise ValueError(f"ROME answer tokenization is empty: {answer!r}")
    return int(ids[0, 0].item())


def _first_token_probability(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    return torch.softmax(logits[:, -1, :].float(), dim=-1)[:, int(target_id)]


def _subject_token_positions(model, prompt: str, subject: str) -> list[int]:
    if not subject:
        raise ValueError("ROME localization requires fact.subject")
    start = prompt.find(subject)
    if start < 0:
        raise ValueError(f"subject {subject!r} is absent from prompt {prompt!r}")
    end = start + len(subject)
    encoded = model.tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("ROME localization requires a fast tokenizer with offsets")
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if int(right) > start and int(left) < end
    ]
    if not positions:
        raise RuntimeError(
            f"token offsets do not overlap subject {subject!r} in {prompt!r}"
        )
    token_count = int(model.to_tokens(prompt).shape[1])
    offset_shift = token_count - len(offsets)
    if offset_shift not in {0, 1}:
        raise RuntimeError(
            "unexpected TransformerLens/tokenizer length difference: "
            f"tokens={token_count}, offsets={len(offsets)}"
        )
    return [position + offset_shift for position in positions]


def _embedding_std(model) -> float:
    weight = getattr(model, "W_E", None)
    if weight is None:
        raise AttributeError("TransformerLens model does not expose W_E")
    value = float(weight.detach().float().std().item())
    if not value > 0.0:
        raise RuntimeError(f"invalid empirical embedding std: {value}")
    return value


@torch.no_grad()
def trace_rome_fact(
    model,
    fact,
    *,
    noise_multiplier: float,
    noise_samples: int,
    generator: torch.Generator,
) -> list[dict]:
    """Measure ROME-style restoration for one fact."""

    if noise_samples <= 0:
        raise ValueError("ROME noise_samples must be positive")
    tokens = model.to_tokens(fact.prompt)
    subject_positions = _subject_token_positions(
        model, fact.prompt, fact.subject
    )
    subject_last = subject_positions[-1]
    target_id = _first_token_id(model, fact.answer)
    clean_logits, clean_cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name.endswith("hook_mlp_out"),
    )
    clean_prob = float(_first_token_probability(clean_logits, target_id)[0].item())
    repeated = tokens.repeat(int(noise_samples), 1)
    noise_scale = float(noise_multiplier) * _embedding_std(model)
    hidden_size = int(model.cfg.d_model)
    noise = torch.randn(
        (int(noise_samples), len(subject_positions), hidden_size),
        device=tokens.device,
        dtype=torch.float32,
        generator=generator,
    ) * noise_scale

    def corrupt_embeddings(activation, hook=None):
        corrupted = activation.clone()
        corrupted[:, subject_positions, :] += noise.to(
            device=corrupted.device, dtype=corrupted.dtype
        )
        return corrupted

    corrupted_logits = model.run_with_hooks(
        repeated,
        fwd_hooks=[("hook_embed", corrupt_embeddings)],
    )
    corrupt_probs = _first_token_probability(corrupted_logits, target_id)
    rows: list[dict] = []
    for layer in range(int(model.cfg.n_layers)):
        hook_name = f"blocks.{layer}.hook_mlp_out"
        clean_value = clean_cache[hook_name][:, subject_last, :].detach().clone()

        def restore_subject_last(activation, hook=None, value=clean_value):
            patched = activation.clone()
            replacement = value.to(device=patched.device, dtype=patched.dtype)
            patched[:, subject_last, :] = replacement.expand(
                patched.shape[0], -1
            )
            return patched

        patched_logits = model.run_with_hooks(
            repeated,
            fwd_hooks=[
                ("hook_embed", corrupt_embeddings),
                (hook_name, restore_subject_last),
            ],
        )
        patched_probs = _first_token_probability(patched_logits, target_id)
        for sample in range(int(noise_samples)):
            corrupt_prob = float(corrupt_probs[sample].item())
            patched_prob = float(patched_probs[sample].item())
            rows.append(
                {
                    "id": fact.id,
                    "type": fact.type,
                    "prompt": fact.prompt,
                    "answer": fact.answer,
                    "subject": fact.subject,
                    "direction": "forward",
                    "layer": int(layer),
                    "site": "mlp_out",
                    "subject_last_token_pos": int(subject_last),
                    "noise_sample": int(sample),
                    "noise_multiplier": float(noise_multiplier),
                    "noise_scale": float(noise_scale),
                    "clean_prob": clean_prob,
                    "corrupt_prob": corrupt_prob,
                    "patched_prob": patched_prob,
                    "raw_indirect_effect": patched_prob - corrupt_prob,
                }
            )
    return rows


def discover_rome_layers(
    model,
    facts: list,
    *,
    config: RomeTraceConfig,
    seed: int,
    bcti_layer_summary: pd.DataFrame,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """Select candidate layers with ROME-style tracing."""

    forward_facts = [
        fact
        for fact in facts
        if fact.type == "forget"
        and ":reverse:" not in str(fact.id)
        and ":causal_probe_reverse:" not in str(fact.id)
        and not str(getattr(fact, "relation", "")).startswith("reverse_")
    ]
    if not forward_facts:
        raise RuntimeError("ROME localization found no forward forget facts")
    device = model.W_E.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    rows: list[dict] = []
    for fact in forward_facts:
        rows.extend(
            trace_rome_fact(
                model,
                fact,
                noise_multiplier=float(config.noise_multiplier),
                noise_samples=int(config.noise_samples),
                generator=generator,
            )
        )
    trace = pd.DataFrame(rows)
    if trace.empty:
        raise RuntimeError("ROME causal tracing returned no rows")
    layer_scores = (
        trace.groupby("layer", as_index=False)
        .agg(
            rome_raw_ie=("raw_indirect_effect", "mean"),
            rome_raw_ie_std=("raw_indirect_effect", "std"),
            rome_clean_prob=("clean_prob", "mean"),
            rome_corrupt_prob=("corrupt_prob", "mean"),
            rome_patched_prob=("patched_prob", "mean"),
            rome_trace_rows=("raw_indirect_effect", "size"),
        )
        .sort_values(["rome_raw_ie", "layer"], ascending=[False, True])
        .reset_index(drop=True)
    )
    count = int(config.layer_count)
    if count <= 0:
        raise ValueError("ROME layer_count must be positive")
    selected = layer_scores.head(count)["layer"].astype(int).tolist()
    if len(selected) != count:
        raise RuntimeError(
            f"ROME requested {count} layers but only found {len(selected)}"
        )
    ranked = bcti_layer_summary.copy().merge(layer_scores, on="layer", how="left")
    ranked["rome_raw_ie"] = pd.to_numeric(
        ranked["rome_raw_ie"], errors="coerce"
    ).fillna(float("-inf"))
    rank_map = {
        int(layer): index + 1
        for index, layer in enumerate(layer_scores["layer"].astype(int).tolist())
    }
    ranked["rome_rank"] = ranked["layer"].astype(int).map(rank_map)
    ranked["selected"] = ranked["layer"].astype(int).isin(selected)
    ranked["selection_order"] = ranked["layer"].astype(int).map(
        {layer: index + 1 for index, layer in enumerate(selected)}
    )
    ranked["selection_reason"] = ranked["selected"].map(
        {True: "rome_raw_ie_topk", False: "not_selected_by_rome"}
    )
    ranked["localization_policy"] = "rome_forward_raw_ie"
    return selected, trace, ranked
