import torch
from typing import Dict

class MemoryTracker:
    def __init__(self, gpu_memory: int = 16):
        # Handle None case
        if gpu_memory is None:
            self.gpu_memory = 16 * 1024 * 1024 * 1024  # Default to 16GB in bytes
        else:
            self.gpu_memory = gpu_memory * 1024 * 1024 * 1024  # Convert to bytes
        
    def get_memory_stats(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**2,
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**2
            }
        return {}

    def check_memory_available(self, required_memory: int) -> bool:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.memory_allocated() + required_memory < self.gpu_memory