
from __future__ import annotations
"""Load experiment configuration and prepare output directories."""


from pathlib import Path

from typing import Any


import yaml



def load_config(path: str | Path = "code/knowledge_group/base_config.yaml") -> dict[str, Any]:
    
    """Load a YAML configuration mapping."""
    
    with Path(path).open("r", encoding="utf-8") as f:
        
        return yaml.safe_load(f)



def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its path."""
    
    p = Path(path)
    
    p.mkdir(parents=True, exist_ok=True)
    
    return p
