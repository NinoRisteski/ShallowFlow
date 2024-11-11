import pytest
import torch
from src.shallowflow.strategies.fsdp import FSDPStrategy, FSDPConfig

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fsdp_initialization():
    config = FSDPConfig(min_num_params=1e6)
    strategy = FSDPStrategy(config)
    assert strategy.config.mixed_precision == True

def test_fsdp_model_preparation(small_model):
    config = FSDPConfig(min_num_params=1e6)
    strategy = FSDPStrategy(config)
    wrapped_model = strategy.prepare_model(small_model)
    assert isinstance(wrapped_model, torch.distributed.fsdp.FullyShardedDataParallel)