
from __future__ import annotations
"""Build causal-unit variants and train a detector that gates activation cuts."""


import argparse

import json

import re

from dataclasses import asdict, dataclass

from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F


from config_utils import ensure_dir, load_config

from data_utils import Fact, load_facts, split_facts



@dataclass(frozen=True)

class CausalUnit:
    """Store one detector example derived from a factual or adversarial variant."""
    
    id: str
    
    fact_id: str
    
    split: str
    
    variant: str
    
    prompt: str
    
    answer: str
    
    subject: str
    
    relation: str
    
    target_label: int
    
    attack_label: int = 0



RELATION_PATTERNS = {
    
    "capital": ["capital of {subject}", "what is the capital of {subject}", "{subject} capital"],
    
    "located_in": ["{subject} is located in", "where is {subject}", "which city is {subject} in"],
    
    "currency": ["currency of {subject}", "what currency does {subject} use", "{subject} currency"],

}



def normalize_text(text: str) -> str:
    """Normalize case and whitespace for lexical matching."""
    
    return re.sub(r"\s+", " ", text.lower()).strip()



def relation_phrase(relation: str) -> str:
    
    """Convert a relation identifier into a readable phrase."""
    
    return relation.replace("_", " ")





def build_causal_units(forget_facts: list[Fact], retain_facts: list[Fact]) -> list[CausalUnit]:
    """Expand forget and retain facts into detector training variants."""
    
    units: list[CausalUnit] = []
    
    facts = forget_facts + retain_facts
    
    retain_by_relation: dict[str, list[Fact]] = {}
    
    for fact in retain_facts:
        
        retain_by_relation.setdefault(fact.relation, []).append(fact)

    
    for fact in facts:
        
        is_target = int(fact.type == "forget")
        
        units.append(
            
            CausalUnit(
                
                id=f"{fact.id}:factual",
                
                fact_id=fact.id,
                
                split=fact.type,
                
                variant="factual",
                
                prompt=fact.prompt,
                
                answer=fact.answer,
                
                subject=fact.subject,
                
                relation=fact.relation,
                
                target_label=is_target,
            
            )
        
        )
        
        units.append(
            
            CausalUnit(
                
                id=f"{fact.id}:unknown_state",
                
                fact_id=fact.id,
                
                split=fact.type,
                
                variant="unknown_state",
                
                prompt=fact.unknown_prompt or f"{fact.prompt} unknown",
                
                answer=" unknown",
                
                subject=fact.subject,
                
                relation=fact.relation,
                
                target_label=is_target,
            
            )
        
        )
        
        for idx, template in enumerate(RELATION_PATTERNS.get(fact.relation, [])):
            
            prompt = template.format(subject=fact.subject)
            
            units.append(
                
                CausalUnit(
                    
                    id=f"{fact.id}:paraphrase:{idx}",
                    
                    fact_id=fact.id,
                    
                    split=fact.type,
                    
                    variant="paraphrase",
                    
                    prompt=prompt,
                    
                    answer=fact.answer,
                    
                    subject=fact.subject,
                    
                    relation=fact.relation,
                    
                    target_label=is_target,
                    
                    attack_label=is_target,
                
                )
            
            )
        
        if fact.subject:
            
            alias = fact.subject.replace("The ", "").replace("the ", "")
            
            if alias != fact.subject:
                
                units.append(
                    
                    CausalUnit(
                        
                        id=f"{fact.id}:alias",
                        
                        fact_id=fact.id,
                        
                        split=fact.type,
                        
                        variant="alias",
                        
                        prompt=fact.prompt.replace(fact.subject, alias),
                        
                        answer=fact.answer,
                        
                        subject=fact.subject,
                        
                        relation=fact.relation,
                        
                        target_label=is_target,
                        
                        attack_label=is_target,
                    
                    )
                
                )
        
        if fact.type == "forget":
            
            for neighbor in retain_by_relation.get(fact.relation, [])[:2]:
                
                units.append(
                    
                    CausalUnit(
                        
                        id=f"{fact.id}:neighbor:{neighbor.id}",
                        
                        fact_id=fact.id,
                        
                        split="neighbor",
                        
                        variant="neighbor",
                        
                        prompt=neighbor.prompt,
                        
                        answer=neighbor.answer,
                        
                        subject=neighbor.subject,
                        
                        relation=neighbor.relation,
                        
                        target_label=0,
                    
                    )
                
                )
        
        units.append(
            
            CausalUnit(
                
                id=f"{fact.id}:placebo",
                
                fact_id=fact.id,
                
                split="placebo",
                
                variant="placebo",
                
                prompt=f"The unrelated biography of {fact.subject} mentions",
                
                answer=" unknown",
                
                subject=fact.subject,
                
                relation=fact.relation,
                
                target_label=0,
            
            )
        
        )
    
    return units



def encode_confounders(unit: CausalUnit) -> dict:
    """Encode confounder indicators used by the detector."""
    
    prompt = normalize_text(unit.prompt)
    
    subject = normalize_text(unit.subject)
    
    return {
        
        "template": unit.variant,
        
        "relation": unit.relation,
        
        "subject_token_count": len(subject.split()),
        
        "prompt_token_count": len(prompt.split()),
        
        "has_subject": int(bool(subject and subject in prompt)),
        
        "is_attack_variant": unit.attack_label,
    
    }



def featurize_unit(unit: CausalUnit, forget_facts: list[Fact], retain_facts: list[Fact]) -> list[float]:
    """Encode one causal unit as detector features."""
    
    prompt = normalize_text(unit.prompt)
    
    subject = normalize_text(unit.subject)
    
    relation = relation_phrase(unit.relation)
    
    same_relation_retain = [x for x in retain_facts if x.relation == unit.relation]
    
    retain_subject_hit = any(normalize_text(x.subject) in prompt for x in same_relation_retain if x.subject)
    
    forget_subject_hit = any(normalize_text(x.subject) in prompt for x in forget_facts if x.subject)
    
    return [
        
        float(bool(subject and subject in prompt)),
        
        float(relation in prompt or unit.relation.replace("_", " ") in prompt),
        
        float(any(word in prompt for word in ["where", "which city", "what", "capital", "currency", "located"])),
        
        float(forget_subject_hit),
        
        float(retain_subject_hit),
        
        float(unit.variant == "factual"),
        
        float(unit.variant == "paraphrase"),
        
        float(unit.variant == "alias"),
        
        float(unit.variant == "neighbor"),
        
        float(unit.variant == "placebo"),
        
        float(unit.variant == "unknown_state"),
        
        len(prompt.split()) / 32.0,
    
    ]



class TargetDetector(nn.Module):
    """Predict whether an input should activate the target-knowledge gate."""
    
    def __init__(self, num_features: int):
        
        
        super().__init__()
        
        self.linear = nn.Linear(num_features, 1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        return self.linear(x).squeeze(-1)



def fp_retain_loss(logits: torch.Tensor, labels: torch.Tensor, retain_mask: torch.Tensor) -> torch.Tensor:
    """Penalize retained examples that incorrectly activate the gate."""
    
    if not bool(retain_mask.any()):
        
        return logits.new_tensor(0.0)
    
    probs = torch.sigmoid(logits[retain_mask])
    
    return probs.mean()



def fn_attack_loss(logits: torch.Tensor, labels: torch.Tensor, attack_mask: torch.Tensor) -> torch.Tensor:
    """Penalize target attacks that fail to activate the gate."""
    
    if not bool(attack_mask.any()):
        
        return logits.new_tensor(0.0)
    
    probs = torch.sigmoid(logits[attack_mask])
    
    return (1.0 - probs).mean()





def train_target_detector(
    
    units: list[CausalUnit],
    
    forget_facts: list[Fact],
    
    retain_facts: list[Fact],
    
    lambda_fp: float = 2.0,
    
    lambda_fn: float = 2.0,
    
    lr: float = 0.05,
    
    epochs: int = 300,

) -> tuple[TargetDetector, dict]:
    """Fit the target detector and return training diagnostics."""
    
    features = torch.tensor([featurize_unit(unit, forget_facts, retain_facts) for unit in units], dtype=torch.float32)
    
    labels = torch.tensor([unit.target_label for unit in units], dtype=torch.float32)
    
    retain_mask = torch.tensor([unit.split in {"retain", "neighbor", "placebo"} for unit in units], dtype=torch.bool)
    
    attack_mask = torch.tensor([bool(unit.attack_label) for unit in units], dtype=torch.bool)

    
    model = TargetDetector(features.shape[1])
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    log = []
    
    for epoch in range(epochs):
        
        logits = model(features)
        
        ce = F.binary_cross_entropy_with_logits(logits, labels)
        
        fp = fp_retain_loss(logits, labels, retain_mask)
        
        fn = fn_attack_loss(logits, labels, attack_mask)
        
        loss = ce + lambda_fp * fp + lambda_fn * fn
        
        opt.zero_grad(set_to_none=True)
        
        loss.backward()
        
        opt.step()
        
        if epoch % 25 == 0 or epoch == epochs - 1:
            
            log.append({"epoch": epoch, "loss": float(loss.item()), "ce": float(ce.item()), "fp_retain": float(fp.item()), "fn_attack": float(fn.item())})

    
    with torch.no_grad():
        
        probs = torch.sigmoid(model(features))
        
        preds = (probs > 0.5).float()
        
        report = {
            
            "accuracy": float((preds == labels).float().mean().item()),
            
            "fp_retain": float(((preds == 1) & retain_mask).float().sum().item()),
            
            "fn_attack": float(((preds == 0) & attack_mask).float().sum().item()),
            
            "train_log": log,
        
        }
    
    return model, report



def detector_probability(detector: TargetDetector, unit: CausalUnit, forget_facts: list[Fact], retain_facts: list[Fact]) -> float:
    
    """Return the detector probability for one causal unit."""
    
    with torch.no_grad():
        
        x = torch.tensor([featurize_unit(unit, forget_facts, retain_facts)], dtype=torch.float32)
        
        return float(torch.sigmoid(detector(x))[0].item())



def save_detector(detector: TargetDetector, path: Path, report: dict, units: list[CausalUnit]) -> None:
    """Save detector weights and feature metadata."""
    
    ensure_dir(path.parent)
    
    torch.save({"state_dict": detector.state_dict(), "num_features": detector.linear.in_features}, path)
    
    (path.with_suffix(".report.json")).write_text(
        
        json.dumps({"report": report, "units": [asdict(unit) | {"confounders": encode_confounders(unit)} for unit in units]}, indent=2, ensure_ascii=False),
        
        encoding="utf-8",
    
    )



def main() -> None:
    """Parse arguments and run the configured workflow."""
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default="code/knowledge_group/base_config.yaml")
    
    parser.add_argument("--out-dir", default="outputs/50-targets/circuit_breaker")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    facts = load_facts(cfg["data"]["facts_path"])
    
    forget_facts, retain_facts = split_facts(facts)
    
    bcfg = cfg.get("breaker", {})
    
    units = build_causal_units(forget_facts, retain_facts)
    
    detector, report = train_target_detector(
        
        units,
        
        forget_facts,
        
        retain_facts,
        
        lambda_fp=float(bcfg.get("lambda_fp", 2.0)),
        
        lambda_fn=float(bcfg.get("lambda_fn", 2.0)),
        
        lr=float(bcfg.get("learning_rate", 0.05)),
        
        epochs=int(bcfg.get("epochs", 300)),
    
    )
    
    out_dir = ensure_dir(args.out_dir)
    
    save_detector(detector, out_dir / "target_detector.pt", report, units)
    
    rows = []
    
    for unit in units:
        
        rows.append(asdict(unit) | encode_confounders(unit) | {"detector_prob": detector_probability(detector, unit, forget_facts, retain_facts)})
    
    pd.DataFrame(rows).to_csv(out_dir / "causal_units_detector.csv", index=False, encoding="utf-8-sig")
    
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    print(f"saved: {out_dir}")



if __name__ == "__main__":
    
    main()
