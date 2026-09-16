
from __future__ import annotations
"""Measure target-answer recall with TransformerLens or Hugging Face models."""


import argparse

import json

from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


import pandas as pd

import torch

import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


from config_utils import ensure_dir, load_config

from data_utils import Fact, load_facts

from tl_pythia_loader import dtype_from_name, load_from_config

from visualization import plot_target_prob



def answer_token_ids(model, answer: str) -> torch.Tensor:
    """Tokenize an answer without prompt-side special tokens."""
    
    ids = model.to_tokens(answer, prepend_bos=False).squeeze(0)
    
    if ids.numel() == 0:
        
        raise ValueError(f"Empty answer tokenization: {answer!r}")
    
    return ids.to(model.cfg.device)



@torch.no_grad()




def target_stats(model, prompt: str, answer: str, top_k: int = 10) -> dict:
    """Compute answer probability, first-token rank, and top-k statistics."""
    
    prompt_tokens = model.to_tokens(prompt)
    
    answer_ids = answer_token_ids(model, answer)
    
    full_tokens = torch.cat([prompt_tokens, answer_ids.unsqueeze(0)], dim=1)
    
    logits = model(full_tokens)
    
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    
    labels = full_tokens[:, 1:]
    
    start = prompt_tokens.shape[1] - 1
    
    end = start + answer_ids.numel()
    
    answer_log_probs = log_probs[:, start:end, :].gather(-1, labels[:, start:end].unsqueeze(-1)).squeeze(-1)
    
    seq_logprob = answer_log_probs.sum().item()
    
    seq_prob = float(torch.exp(answer_log_probs.sum()).item())

    
    next_logits = model(prompt_tokens)[:, -1, :].float()
    
    first_id = int(answer_ids[0].item())
    
    first_log_probs = F.log_softmax(next_logits, dim=-1)
    
    first_prob = float(first_log_probs.exp()[0, first_id].item())
    
    first_rank = int((next_logits[0] > next_logits[0, first_id]).sum().item() + 1)
    
    values, indices = torch.topk(first_log_probs.exp()[0], k=top_k)
    
    top_tokens = [
        
        {
            
            "token": model.to_string(int(tok.item())),
            
            "token_id": int(tok.item()),
            
            "prob": float(prob.item()),
        
        }
        
        for prob, tok in zip(values, indices)
    
    ]
    
    return {
        
        "answer_num_tokens": int(answer_ids.numel()),
        
        "answer_logprob": float(seq_logprob),
        
        "answer_prob": seq_prob,
        
        "first_token_prob": first_prob,
        
        "first_token_rank": first_rank,
        
        "top_tokens": top_tokens,
    
    }



def eval_facts(model, facts: list[Fact], top_k: int) -> pd.DataFrame:
    
    """Evaluate target statistics with a TransformerLens model."""
    
    rows = []
    
    for fact in facts:
        
        stats = target_stats(model, fact.prompt, fact.answer, top_k=top_k)
        
        row = {
            
            "id": fact.id,
            
            "type": fact.type,
            
            "prompt": fact.prompt,
            
            "answer": fact.answer,
            
            "subject": fact.subject,
            
            "relation": fact.relation,
        
        }
        
        row.update({k: v for k, v in stats.items() if k != "top_tokens"})
        
        row["top_tokens_json"] = json.dumps(stats["top_tokens"], ensure_ascii=False)
        
        rows.append(row)
    
    return pd.DataFrame(rows)



def load_hf_from_config(config: dict):
    
    """Load a Hugging Face causal language model and tokenizer."""
    
    mcfg = config["model"]
    
    dtype = dtype_from_name(mcfg.get("dtype", "float16"))
    
    device = mcfg.get("device", "cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(
        
        mcfg["local_path"],
        
        local_files_only=bool(mcfg.get("local_files_only", True)),
    
    )
    
    if tokenizer.pad_token is None:
        
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        
        mcfg["local_path"],
        
        torch_dtype=dtype,
        
        local_files_only=bool(mcfg.get("local_files_only", True)),
        
        low_cpu_mem_usage=True,
    
    ).to(device)
    
    model.eval()
    
    return model, tokenizer, device



@torch.no_grad()

def hf_target_stats(model, tokenizer, device: str, prompt: str, answer: str, top_k: int = 10) -> dict:
    
    """Compute target statistics from Hugging Face logits."""
    
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
    
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    
    full_ids = torch.cat([prompt_enc["input_ids"], answer_ids], dim=1)
    
    full_attn = torch.ones_like(full_ids, device=device)
    
    logits = model(input_ids=full_ids, attention_mask=full_attn).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    
    labels = full_ids[:, 1:]
    
    start = prompt_enc["input_ids"].shape[1] - 1
    
    end = start + answer_ids.shape[1]
    
    answer_log_probs = log_probs[:, start:end, :].gather(-1, labels[:, start:end].unsqueeze(-1)).squeeze(-1)

    
    next_logits = model(**prompt_enc).logits[:, -1, :].float()
    
    first_id = int(answer_ids[0, 0].item())
    
    next_probs = F.softmax(next_logits, dim=-1)
    
    values, indices = torch.topk(next_probs[0], k=top_k)
    
    return {
        
        "answer_num_tokens": int(answer_ids.shape[1]),
        
        "answer_logprob": float(answer_log_probs.sum().item()),
        
        "answer_prob": float(torch.exp(answer_log_probs.sum()).item()),
        
        "first_token_prob": float(next_probs[0, first_id].item()),
        
        "first_token_rank": int((next_logits[0] > next_logits[0, first_id]).sum().item() + 1),
        
        "top_tokens": [
            
            {
                
                "token": tokenizer.decode([int(tok.item())]),
                
                "token_id": int(tok.item()),
                
                "prob": float(prob.item()),
            
            }
            
            for prob, tok in zip(values, indices)
        
        ],
    
    }



def eval_facts_hf(model, tokenizer, device: str, facts: list[Fact], top_k: int) -> pd.DataFrame:
    
    """Evaluate target statistics with a Hugging Face model."""
    
    rows = []
    
    for fact in facts:
        
        stats = hf_target_stats(model, tokenizer, device, fact.prompt, fact.answer, top_k=top_k)
        
        row = {
            
            "id": fact.id,
            
            "type": fact.type,
            
            "prompt": fact.prompt,
            
            "answer": fact.answer,
            
            "subject": fact.subject,
            
            "relation": fact.relation,
        
        }
        
        row.update({k: v for k, v in stats.items() if k != "top_tokens"})
        
        row["top_tokens_json"] = json.dumps(stats["top_tokens"], ensure_ascii=False)
        
        rows.append(row)
    
    return pd.DataFrame(rows)



def main() -> None:
    """Parse arguments and run the configured workflow."""
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default="code/knowledge_group/base_config.yaml")
    
    parser.add_argument("--out", default="outputs/50-targets/target_prob.csv")
    
    parser.add_argument("--backend", choices=["hf", "tl"], default="hf")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    facts = load_facts(cfg["data"]["facts_path"])
    
    if args.backend == "hf":
        
        model, tokenizer, device = load_hf_from_config(cfg)
        
        df = eval_facts_hf(model, tokenizer, device, facts, top_k=int(cfg["eval"].get("top_k", 10)))
    
    else:
        
        model = load_from_config(cfg)
        
        df = eval_facts(model, facts, top_k=int(cfg["eval"].get("top_k", 10)))
    
    out = Path(args.out)
    
    ensure_dir(out.parent)
    
    df.to_csv(out, index=False, encoding="utf-8-sig")
    
    plot_target_prob(df, out.with_suffix(".png"))
    
    print(df[["id", "type", "answer_logprob", "first_token_prob", "first_token_rank"]].to_string(index=False))
    
    print(f"saved: {out}")
    
    print(f"saved: {out.with_suffix('.png')}")



if __name__ == "__main__":
    
    main()
