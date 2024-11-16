import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import math

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: int = 8
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combine original transformation with LoRA update
        original = self.linear(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original + lora