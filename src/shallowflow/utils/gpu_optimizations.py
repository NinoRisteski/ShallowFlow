import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
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
    min_num_params: int = 1e6  # Minimum number of parameters for FSDP wrapping
    backward_prefetch: bool = True
    cpu_offload: bool = False

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

    def _get_mixed_precision_policy(self) -> Optional[MixedPrecision]:
        """Get mixed precision policy for FSDP."""
        if not self.config.mixed_precision:
            return None
            
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        
    def _get_cpu_offload(self) -> Optional[CPUOffload]:
        """Get CPU offload configuration for FSDP."""
        if not self.config.cpu_offload:
            return None
            
        return CPUOffload(offload_params=True)

    def prepare_model(self, model: torch.nn.Module) -> FSDP:
        """Wrap model in FSDP with appropriate optimizations."""
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
            
        # Check memory before wrapping
        self._check_memory()
            
        # Wrap model with FSDP
        wrapped_model = FSDP(model, **fsdp_config)
        
        return wrapped_model