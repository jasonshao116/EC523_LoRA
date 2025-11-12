import torch
from src.lora.layers import LoRALinear

def test_lora_linear_shapes():
    layer = LoRALinear(64, 32, r=8, alpha=16)
    x = torch.randn(4, 64)
    y = layer(x)
    assert y.shape == (4, 32)

def test_merge_no_crash():
    layer = LoRALinear(32, 16, r=4, alpha=8)
    layer.merge_weights_()  # should not raise
