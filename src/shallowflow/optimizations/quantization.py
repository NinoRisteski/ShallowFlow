import torch
from typing import Tuple
import warnings

class Quantizer:
    def __init__(self, bits: int = 8):
        if not isinstance(bits, int) or bits <= 0 or bits > 32:
            raise ValueError("bits must be a positive integer <= 32")
        self.bits = bits
        self.min_val = -(2**(bits-1))
        self.max_val = 2**(bits-1) - 1
        
    def quantize(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            raise RuntimeError("Cannot quantize empty tensor")
            
        if torch.isinf(x).any() or torch.isnan(x).any():
            warnings.warn("Input tensor contains inf or nan values")
            x = torch.nan_to_num(x, nan=0.0, posinf=x[~torch.isinf(x)].max().item(), 
                               neginf=x[~torch.isinf(x)].min().item())
        
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