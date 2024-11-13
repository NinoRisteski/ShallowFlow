import pytest
import torch
from src.shallowflow.strategies.fsdp import FSDPStrategy, FSDPConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fsdp_initialization():
    config = FSDPConfig(min_num_params=1e6)
    strategy = FSDPStrategy(config)
    assert strategy.config.mixed_precision == True

def test_fsdp_model_preparation(small_model):
    config = FSDPConfig(min_num_params=1e6)
    strategy = FSDPStrategy(config)
    
    # Move model to CPU for testing
    small_model = small_model.cpu()
    
    # Create auto wrap policy
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=config.min_num_params,
        force_leaf_modules=set(),
        module=small_model,
        recurse=True,
        nonwrapped_numel=0
    )
    
    wrapped_model = strategy.prepare_model(
        small_model,
        auto_wrap_policy=auto_wrap_policy
    )
    
    assert isinstance(wrapped_model, torch.distributed.fsdp.FullyShardedDataParallel)

@pytest.mark.parametrize("bit_width", [4, 8, 16])
def test_different_bit_widths(bit_width):
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    )
    
    fsdp_config = FSDP.config(
        mixed_precision=mixed_precision_policy,
        # ... rest of your config ...
    )