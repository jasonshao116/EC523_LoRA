# src/lora/inject.py
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
try:
    # Most transformers versions
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
except Exception:
    # Fallback if path changes
    from transformers.modeling_utils import Conv1D  # type: ignore

from .layers import LoRALinear

# GPT-2 targets
TARGETS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

def _get_parent_by_name(root: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj

def _wrap_linear_like(module: nn.Module, r: int, alpha: int, dropout: float) -> LoRALinear:
    """
    Create a LoRALinear that mimics either nn.Linear or Conv1D.
    For Conv1D, its weight has shape (in, out); Linear expects (out, in) -> we transpose.
    """
    if isinstance(module, nn.Linear):
        out_f, in_f = module.out_features, module.in_features
        lora = LoRALinear(in_f, out_f, r=r, alpha=alpha, dropout=dropout, bias=(module.bias is not None))
        with torch.no_grad():
            lora.weight.copy_(module.weight.data)
            if module.bias is not None:
                lora.bias.copy_(module.bias.data)
        return lora

    if isinstance(module, Conv1D):
        in_f, out_f = module.weight.shape  # Conv1D stores (in, out)
        lora = LoRALinear(in_f, out_f, r=r, alpha=alpha, dropout=dropout, bias=(module.bias is not None))
        with torch.no_grad():
            # Linear wants (out, in)
            lora.weight.copy_(module.weight.data.t())
            if module.bias is not None:
                lora.bias.copy_(module.bias.data)
        return lora

    raise TypeError("Unsupported module type for LoRA wrapping")

def inject_lora_gpt2(
    model: PreTrainedModel,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    targets: Optional[List[str]] = None,
    skip_if_contains: Optional[List[str]] = None,
    verbose: bool = True,
) -> PreTrainedModel:
    targets = targets or TARGETS
    skip_if_contains = skip_if_contains or ["score", "lm_head", "classifier"]

    wrapped = []
    for name, module in list(model.named_modules()):
        is_linear_like = isinstance(module, nn.Linear) or isinstance(module, Conv1D)
        if not is_linear_like:
            continue
        if any(s in name for s in skip_if_contains):
            continue
        if not any(t in name for t in targets):
            continue

        parent = _get_parent_by_name(model, name)
        attr = name.split(".")[-1]
        lora_lin = _wrap_linear_like(module, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_lin)
        wrapped.append(name)

    if verbose:
        print(f"[LoRA] Wrapped {len(wrapped)} linear-like layers:")
        for w in wrapped:
            print("  -", w)
    return model
