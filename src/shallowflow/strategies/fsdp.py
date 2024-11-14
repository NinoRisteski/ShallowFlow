import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from dataclasses import dataclass
from typing import Optional, Dict
from functools import partial
from unittest.mock import patch

@dataclass
class FSDPConfig:
    min_num_params: int = 1e6  # Minimum number of parameters for FSDP wrapping
    cpu_offload: bool = False
    mixed_precision: bool = True
    backward_prefetch: bool = True
    activation_checkpointing: bool = False

class StrategyInstance:
    def __init__(self, config: FSDPConfig):
        self.config = config
        
    def get_mixed_precision_policy(self):
        if self.config.mixed_precision:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        return None
        
    # Additional strategy-related methods can be added here

class FSDPStrategy:
    def __init__(self, config: Optional[FSDPConfig] = None):
        self.config = config or FSDPConfig()
        self.strategy_instance = StrategyInstance(self.config)
        
    def _get_mixed_precision_policy(self):
        """Get mixed precision policy based on config"""
        if self.config.mixed_precision:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        return None
        
    def _get_cpu_offload(self):
        """Configure CPU offload"""
        if self.config.cpu_offload:
            return CPUOffload(offload_params=True)
        return None
        
    def _get_backward_prefetch(self):
        if self.config.backward_prefetch:
            return BackwardPrefetch.BACKWARD_PRE
        return None
        
    def _create_mixed_precision_policy(self):
        """Create and return a mixed precision policy for FSDP."""
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        
    def prepare_model(self, model: torch.nn.Module) -> FullyShardedDataParallel:
        """Wrap model in FSDP"""
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=self.config.min_num_params
        )
        
        mixed_precision_policy = self._get_mixed_precision_policy()
    
        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            mixed_precision=mixed_precision_policy,
            cpu_offload=self._get_cpu_offload(),
            backward_prefetch=self._get_backward_prefetch(),
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        ):
            wrapped_model = wrap(model)
            
        return wrapped_model
    
        
    def prepare_optimizer(
        self,
        model: FullyShardedDataParallel,
        optimizer_class,
        **optimizer_kwargs
    ):
        """Create optimizer for FSDP model"""
        return optimizer_class(
            model.parameters(),
            **optimizer_kwargs
        )