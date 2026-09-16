
from __future__ import annotations
"""Locate knowledge-bearing paths through activation corruption and restoration."""


import argparse

from pathlib import Path


import pandas as pd

import torch


from activation_patching import PatchSpec, clean_value_from_cache, run_with_patch

from config_utils import ensure_dir, load_config

from data_utils import corrupt_prompt, load_facts

from eval_target_prob import answer_token_ids

from tl_pythia_loader import load_from_config

from visualization import plot_causal_tracing



def cache_names_filter(name: str) -> bool:
    """Return whether a hook belongs in the activation cache."""
    
    return (
        
        name.endswith("hook_resid_pre")
        
        or name.endswith("hook_attn_out")
        
        or name.endswith("hook_mlp_out")
    
    )



@torch.no_grad()

def first_token_prob_from_logits(model, logits: torch.Tensor, answer: str) -> float:
    """Read a target probability from final-position logits."""
    
    target_id = int(answer_token_ids(model, answer)[0].item())
    
    probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
    
    return float(probs[0, target_id].item())



@torch.no_grad()




def trace_one_fact(model, fact, sites: list[str], corrupt_strategy: str) -> list[dict]:
    """Measure layerwise restoration effects for one fact."""
    
    clean_tokens = model.to_tokens(fact.prompt)
    
    corrupt = corrupt_prompt(fact, strategy=corrupt_strategy)
    
    corrupt_tokens = model.to_tokens(corrupt)

    
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=cache_names_filter)
    
    corrupt_logits = model(corrupt_tokens)

    
    clean_prob = first_token_prob_from_logits(model, clean_logits, fact.answer)
    
    corrupt_prob = first_token_prob_from_logits(model, corrupt_logits, fact.answer)
    
    rows = []
    
    for layer in range(model.cfg.n_layers):
        
        for site in sites:
            
            spec = PatchSpec(layer=layer, site=site, token_pos=-1)
            
            replacement = clean_value_from_cache(clean_cache, layer=layer, site=site, token_pos=-1)
            
            patched_logits = run_with_patch(model, corrupt_tokens, spec, replacement)
            
            patched_prob = first_token_prob_from_logits(model, patched_logits, fact.answer)
            
            rows.append(
                
                {
                    
                    "id": fact.id,
                    
                    "type": fact.type,
                    
                    "prompt": fact.prompt,
                    
                    "corrupt_prompt": corrupt,
                    
                    "answer": fact.answer,
                    
                    "layer": layer,
                    
                    "site": site,
                    
                    "corruption": corrupt_strategy,
                    
                    "clean_prob": clean_prob,
                    
                    "corrupt_prob": corrupt_prob,
                    
                    "patched_prob": patched_prob,
                    
                    "restore_effect": patched_prob - corrupt_prob,
                    
                    "remaining_gap": clean_prob - patched_prob,
                
                }
            
            )
    
    return rows



def run_tracing(model, facts, sites: list[str], corrupt_strategy: str, max_examples: int | None = None) -> pd.DataFrame:
    
    """Trace facts with single-site activation restoration."""
    
    rows = []
    
    selected = facts[:max_examples] if max_examples else facts
    
    for fact in selected:
        
        rows.extend(trace_one_fact(model, fact, sites=sites, corrupt_strategy=corrupt_strategy))
    
    return pd.DataFrame(rows)





def run_tracing_multi(
    
    model,
    
    facts,
    
    sites: list[str],
    
    corrupt_strategies: list[str],
    
    max_examples: int | None = None,

) -> pd.DataFrame:
    
    """Trace facts across multiple activation sites."""
    
    frames = [
        
        run_tracing(
            
            model,
            
            facts,
            
            sites=sites,
            
            corrupt_strategy=strategy,
            
            max_examples=max_examples,
        
        )
        
        for strategy in corrupt_strategies
    
    ]
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def main() -> None:
    """Parse arguments and run the configured workflow."""
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default="code/knowledge_group/base_config.yaml")
    
    parser.add_argument("--out", default="outputs/50-targets/causal_tracing.csv")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    facts = load_facts(cfg["data"]["facts_path"])
    
    model = load_from_config(cfg)
    
    df = run_tracing(
        
        model,
        
        facts,
        
        sites=list(cfg["tracing"]["sites"]),
        
        corrupt_strategy=cfg["tracing"].get("corrupt_strategy", "subject_unknown"),
        
        max_examples=cfg["tracing"].get("max_examples"),
    
    )
    
    out = Path(args.out)
    
    ensure_dir(out.parent)
    
    df.to_csv(out, index=False, encoding="utf-8-sig")
    
    plot_causal_tracing(df, out.with_suffix(".png"))
    
    print(df.sort_values("restore_effect", ascending=False).head(20).to_string(index=False))
    
    print(f"saved: {out}")
    
    print(f"saved: {out.with_suffix('.png')}")



if __name__ == "__main__":
    
    main()
