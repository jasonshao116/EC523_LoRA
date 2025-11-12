import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Minimal LoRA: W <- W + (B @ A) * (alpha/r)
    Only A and B are trained; base weight W stays frozen.
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.r = int(r) if r else 0
        self.alpha = alpha
        self.scaling = (alpha / r) if self.r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)

        if self.r > 0:
            self.A = nn.Parameter(torch.zeros(self.r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, self.r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora = self.dropout(x) @ self.A.t() @ self.B.t()
            return base + self.scaling * lora
        return base

    @torch.no_grad()
    def merge_weights_(self):
        if self.r > 0:
            self.weight += self.scaling * (self.B @ self.A)
