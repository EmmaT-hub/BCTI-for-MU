
from __future__ import annotations
"""Load local Hugging Face Pythia checkpoints into TransformerLens."""


from contextlib import contextmanager

from typing import Iterator


import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer



def dtype_from_name(name: str) -> torch.dtype:
    
    """Resolve a configured floating-point dtype."""
    
    normalized = name.lower()
    
    if normalized in {"float16", "fp16", "half"}:
        
        return torch.float16
    
    if normalized in {"bfloat16", "bf16"}:
        
        return torch.bfloat16
    
    if normalized in {"float32", "fp32"}:
        
        return torch.float32
    
    raise ValueError(f"Unsupported dtype: {name}")



@contextmanager

def patch_tl_autoconfig(model_name: str, local_path: str, local_files_only: bool = True) -> Iterator[None]:
    """Register local configuration support with TransformerLens."""
    
    import transformer_lens.loading_from_pretrained as loading

    
    original_loading = loading.AutoConfig.from_pretrained
    
    original_hf = AutoConfig.from_pretrained

    
    def patched_auto_config_from_pretrained(name_or_path, *args, **kwargs):
        """Load model configuration with the local compatibility patch."""
        
        if str(name_or_path) == model_name:
            
            kwargs["local_files_only"] = local_files_only
            
            return original_hf(local_path, *args, **kwargs)
        
        return original_hf(name_or_path, *args, **kwargs)

    
    loading.AutoConfig.from_pretrained = patched_auto_config_from_pretrained
    
    try:
        
        yield
    
    finally:
        
        loading.AutoConfig.from_pretrained = original_loading






def load_pythia_tl(
    
    model_name: str = "EleutherAI/pythia-1.4b-deduped",
    
    local_path: str = "models/pythia-1.4b-deduped",
    
    device: str = "cuda",
    
    dtype: str = "float16",
    
    local_files_only: bool = True,

) -> HookedTransformer:
    """Load a local Pythia checkpoint into TransformerLens."""
    
    torch_dtype = dtype_from_name(dtype)
    
    
    
    
    
    
    
    
    
    
    
    source_device = "cpu" if device.startswith("cuda") else device
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        
        local_path,
        
        torch_dtype=torch_dtype,
        
        local_files_only=local_files_only,
        
        low_cpu_mem_usage=True,
        
        device_map={"": source_device},
    
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        
        local_path,
        
        local_files_only=local_files_only,
    
    )
    
    if tokenizer.pad_token is None:
        
        tokenizer.pad_token = tokenizer.eos_token

    
    with patch_tl_autoconfig(model_name, local_path, local_files_only=local_files_only):
        
        loader = getattr(HookedTransformer, "from_pretrained_no_processing", None)
        
        if loader is None:
            
            loader = HookedTransformer.from_pretrained
        
        model = loader(
            
            model_name,
            
            hf_model=hf_model,
            
            tokenizer=tokenizer,
            
            device=device,
            
            dtype=torch_dtype,
            
            local_files_only=local_files_only,
        
        )
    
    del hf_model
    
    if torch.cuda.is_available():
        
        torch.cuda.empty_cache()
    
    model.eval()
    
    return model





def load_from_config(config: dict) -> HookedTransformer:
    
    """Load the configured TransformerLens model."""
    
    m = config["model"]
    
    return load_pythia_tl(
        
        model_name=m["model_name"],
        
        local_path=m["local_path"],
        
        device=m.get("device", "cuda"),
        
        dtype=m.get("dtype", "float16"),
        
        local_files_only=bool(m.get("local_files_only", True)),
    
    )
