import torch
from typing import Tuple

class Quantizer:
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.max_val = 2**(bits - 1) - 1
        
    def quantize(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = (x.max() - x.min()) / (2**self.bits - 1)
        zero_point = (-x.min() / scale).round()
        
        x_quantized = torch.clamp(
            (x / scale).round() + zero_point,
            0,
            2**self.bits - 1
        )
        
        return x_quantized.to(torch.int8), scale, zero_point
        
    def dequantize(
        self,
        x_quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        return scale * (x_quantized.float() - zero_point)