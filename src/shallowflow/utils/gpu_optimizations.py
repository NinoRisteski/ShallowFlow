import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel, 
    size_based_auto_wrap_policy, 
    enable_wrap, 
    wrap,
    BackwardPrefetch
)

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

    def _check_memory(self):
        if torch.cuda.memory_allocated() > 0.9 * self.config.memory_limit * 1e9:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU memory nearly exhausted")

    def prepare_model(self, model: torch.nn.Module) -> FullyShardedDataParallel:
        """Wrap model in FSDP"""
        # Auto wrapping policy
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=self.config.min_num_params
        )
        
        # FSDP configuration
        fsdp_config = {
            "auto_wrap_policy": auto_wrap_policy,
            "mixed_precision": self._get_mixed_precision_policy(),
            "cpu_offload": self._get_cpu_offload()
        }
        
        if self.config.backward_prefetch:
            fsdp_config["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            
        # Wrap model with FSDP
        with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp_config):
            wrapped_model = wrap(model)
            
        return wrapped_model