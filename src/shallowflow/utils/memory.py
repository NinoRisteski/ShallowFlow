import torch
from typing import Dict

class MemoryOptimizer:
    def __init__(self, gpu_memory: int = 16):  # T4 has 16GB
        self.gpu_memory = gpu_memory * 1024 * 1024 * 1024  # Convert to bytes
        
    def get_optimal_batch_size(self, model: torch.nn.Module) -> int:
        total_params = sum(p.numel() for p in model.parameters())
        param_size = total_params * 4  # Assuming float32
        
        available_memory = self.gpu_memory - (param_size * 3)
        
        return max(1, int(available_memory / (param_size * 4)))