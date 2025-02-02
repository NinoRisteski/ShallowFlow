import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import math

class LoRALayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,  # Pass the base layer instead of dimensions
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Store dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Store the base layer and freeze it
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA dropout
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # LoRA matrices (only these will be trained)
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
            self.lora_B = nn.Parameter(torch.empty(rank, self.out_features))
            self.reset_lora_parameters()
        
        # For tracking purposes
        self.merged = False
    
    def reset_lora_parameters(self):
        """Initialize LoRA parameters"""
        if self.rank > 0:
            # Initialize A with kaiming
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialize B as zero
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass through frozen base layer
        base_output = self.base_layer(x)
        
        if self.rank > 0:
            # LoRA forward pass
            lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            return base_output + lora_output * self.scaling
        return base_output
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into base weights for inference"""
        if self.merged or self.rank == 0:
            return
        
        # Merge weights
        self.base_layer.weight.data += (self.lora_B @ self.lora_A).T * self.scaling
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from base weights"""
        if not self.merged or self.rank == 0:
            return
            
        # Unmerge weights
        self.base_layer.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
        self.merged = False