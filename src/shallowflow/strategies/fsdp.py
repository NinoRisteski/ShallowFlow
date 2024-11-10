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

@dataclass
class FSDPConfig:
    min_num_params: int = 1e6  # Minimum number of parameters for FSDP wrapping
    cpu_offload: bool = False
    mixed_precision: bool = True
    backward_prefetch: bool = True
    activation_checkpointing: bool = False

class FSDPStrategy:
    def __init__(self, config: Optional[FSDPConfig] = None):
        self.config = config or FSDPConfig()
        
    def _get_mixed_precision_policy(self):
        """Configure mixed precision policy"""
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