import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

@dataclass
class GTX1660Config:
    memory_limit: int = 6  # 6GB VRAM
    batch_size: int = 8
    mixed_precision: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    gradient_checkpointing: bool = True
    pin_memory: bool = True

class GTX1660Optimizer:
    def __init__(self, config: Optional[GTX1660Config] = None):
        self.config = config or GTX1660Config()
        self.scaler = GradScaler()
        
    def setup_mixed_precision(
        self,
        model: torch.nn.Module
    ) -> Tuple[torch.nn.Module, GradScaler]:
        model = model.cuda()
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model, self.scaler

    def create_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )

    def get_memory_stats(self) -> dict:
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2
        }