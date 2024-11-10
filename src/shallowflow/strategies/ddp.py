import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class DDPConfig:
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    find_unused_parameters: bool = False

class DDPStrategy:
    def __init__(self, config: Optional[DDPConfig] = None):
        self.config = config or DDPConfig()
        self._setup_distributed()
        
    def _setup_distributed(self):
        """Initialize distributed process group"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
    def prepare_model(self, model: torch.nn.Module) -> DistributedDataParallel:
        """Wrap model in DDP"""
        device = torch.device(f"cuda:{self.config.rank}")
        model = model.to(device)
        return DistributedDataParallel(
            model,
            device_ids=[self.config.rank],
            find_unused_parameters=self.config.find_unused_parameters
        )
        
    def prepare_dataloader(self, dataloader: torch.utils.data.DataLoader):
        """Prepare dataloader for distributed training"""
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataloader.dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank
        )
        return torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=True
        )
        
    def cleanup(self):
        """Cleanup distributed process group"""
        if dist.is_initialized():
            dist.destroy_process_group()