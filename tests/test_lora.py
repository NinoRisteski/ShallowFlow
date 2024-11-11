import pytest
import torch
from src.shallowflow.optimizations.lora import LoRALayer

def test_lora_initialization():
    lora = LoRALayer(10, 5, rank=4)
    assert lora.lora_A.shape == (10, 4)
    assert lora.lora_B.shape == (4, 5)

def test_lora_forward():
    lora = LoRALayer(10, 5, rank=4)
    x = torch.randn(2, 10)
    output = lora(x)
    assert output.shape == (2, 5)

def test_lora_backward():
    lora = LoRALayer(10, 5, rank=4)
    x = torch.randn(2, 10, requires_grad=True)
    output = lora(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None

def test_lora_scaling():
    lora = LoRALayer(10, 5, rank=4, alpha=8)
    assert lora.scaling == 2.0  # alpha/rank = 8/4