from __future__ import annotations  

"""Patch selected activations by layer, hook site, and token position."""

from dataclasses import dataclass 

import torch  


SITE_TO_HOOK = {  
    "resid_pre": "blocks.{layer}.hook_resid_pre",  
    "attn_out": "blocks.{layer}.hook_attn_out",  
    "mlp_out": "blocks.{layer}.hook_mlp_out",  
}  


@dataclass(frozen=True)  
class PatchSpec:  
    """Identify one activation patch by layer, hook site, and token position."""

    layer: int  
    site: str  
    token_pos: int = -1  

    @property  
    def hook_name(self) -> str:  
        """Return the TransformerLens hook selected by this patch."""

        if self.site not in SITE_TO_HOOK:  
            raise ValueError(  
                f"Unknown site {self.site!r}. Valid: {sorted(SITE_TO_HOOK)}"  
            )  
        return SITE_TO_HOOK[self.site].format(layer=self.layer)  


def get_hook_name(layer: int, site: str) -> str:  
    """Resolve the TransformerLens hook for a layer and site."""

    return PatchSpec(layer=layer, site=site).hook_name  


def clean_value_from_cache(  
    cache,  
    layer: int,  
    site: str,  
    token_pos: int = -1,  
) -> torch.Tensor:  
    """Read the clean activation selected by a patch."""

    value = cache[get_hook_name(layer, site)]  
    return value[:, token_pos, :].detach().clone()  





def make_token_replacement_hook(  
    replacement: torch.Tensor,  
    token_pos: int = -1,  
    strength: float = 1.0,  
):  
    """Create a hook that replaces one token activation."""

    def hook_fn(activation, hook=None):  
        """Apply the captured activation edit."""

        patched = activation.clone()  
        repl = replacement.to(device=patched.device, dtype=patched.dtype)  
        if repl.ndim == 3:  
            repl = repl[:, token_pos, :]  
        if repl.ndim == 1:  
            repl = repl.unsqueeze(0)  
        strength_clamped = max(0.0, float(strength))  
        patched[:, token_pos, :] = (  
            (1.0 - strength_clamped) * patched[:, token_pos, :]  
            + strength_clamped * repl  
        )  
        return patched  

    return hook_fn  


def make_zero_ablation_hook(token_pos: int = -1):  
    """Create a hook that zeros one token activation."""

    def hook_fn(activation, hook=None):  
        """Apply the captured activation edit."""

        patched = activation.clone()  
        patched[:, token_pos, :] = 0  
        return patched  

    return hook_fn  


def run_with_patch(  
    model,  
    tokens,  
    spec: PatchSpec,  
    replacement: torch.Tensor,  
    strength: float = 1.0,  
):  
    """Run a forward pass with one activation patch."""

    replacement_hook = make_token_replacement_hook(  
        replacement,  
        spec.token_pos,  
        strength,  
    )  
    return model.run_with_hooks(  
        tokens,  
        fwd_hooks=[(spec.hook_name, replacement_hook)],  
    )  




def run_with_multi_patch(  
    model,  
    tokens,  
    replacements: dict[PatchSpec, torch.Tensor],  
    strength: float = 1.0,  
):  
    """Run a forward pass with multiple activation patches."""

    hooks = [  
        (  
            spec.hook_name,  
            make_token_replacement_hook(value, spec.token_pos, strength),  
        )  
        for spec, value in replacements.items()  
    ]  
    return model.run_with_hooks(tokens, fwd_hooks=hooks)  
