import pytest
import torch
from src.shallowflow.optimizations.quantization import Quantizer

def test_quantizer_initialization():
    quantizer = Quantizer(bits=8)
    assert quantizer.bits == 8
    assert quantizer.max_val == 127

def test_quantization_dequantization():
    quantizer = Quantizer(bits=8)
    x = torch.randn(100, 100)
    x_q, scale, zero_point = quantizer.quantize(x)
    x_dq = quantizer.dequantize(x_q, scale, zero_point)
    
    # Check if dequantized values are close to original
    assert torch.allclose(x, x_dq, rtol=0.1)

def test_quantization_range():
    quantizer = Quantizer(bits=8)
    x = torch.randn(100, 100)
    x_q, _, _ = quantizer.quantize(x)
    assert x_q.max() <= 127
    assert x_q.min() >= -128

@pytest.mark.parametrize("bits", [4, 8, 16])
def test_different_bit_widths(bits):
    quantizer = Quantizer(bits=bits)
    x = torch.randn(10, 10)
    x_q, scale, zero_point = quantizer.quantize(x)
    
    # Ensure proper dtype based on bit width
    if bits <= 8:
        expected_dtype = torch.int8
        max_value = 2**(bits) - 1  # Use full range instead of signed range
    else:
        expected_dtype = torch.int16
        max_value = 2**(bits) - 1

    assert x_q.dtype == expected_dtype
    assert x_q.max() <= max_value
    assert x_q.min() >= 0  # Assuming unsigned quantization