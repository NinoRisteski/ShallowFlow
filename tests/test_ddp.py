import os
import pytest
import torch
import torch.distributed as dist
from src.shallowflow.strategies.ddp import DDPStrategy, DDPConfig

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ddp_initialization():
    config = DDPConfig(world_size=1, rank=0)
    strategy = DDPStrategy(config)
    assert dist.is_initialized()

def test_ddp_model_preparation(small_model):
    # Set required environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    
    config = DDPConfig(world_size=1, rank=0)
    strategy = DDPStrategy(config)
    wrapped_model = strategy.prepare_model(small_model)
    assert isinstance(wrapped_model, torch.nn.parallel.DistributedDataParallel)
    
    # Clean up after test
    if dist.is_initialized():
        dist.destroy_process_group()