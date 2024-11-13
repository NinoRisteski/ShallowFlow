import torch
from typing import Tuple

class Quantizer:
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.min_val = -(2**(bits-1))
        self.max_val = 2**(bits-1) - 1
        
    def quantize(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = (x.max() - x.min()) / (2**self.bits - 1)
        zero_point = (x.min() / scale).round()
        
        x_q = (x / scale + zero_point).round()
        
        x_q = torch.clamp(x_q, self.min_val, self.max_val)
        
        if self.bits <= 8:
            x_q = x_q.to(torch.int8)
        else:
            x_q = x_q.to(torch.int16)
        
        return x_q, scale, zero_point
        
    def dequantize(
        self,
        x_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        return scale * (x_q.float() - zero_point)