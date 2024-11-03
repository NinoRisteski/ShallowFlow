# src/shallowflow/trainer/base.py
from dataclasses import dataclass
import torch
from typing import Optional, Dict, Any

@dataclass
class AWSTrainingConfig:
    instance_type: str = "g4dn.xlarge"
    gpu_memory: int = 16  # GB
    max_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    mixed_precision: bool = True
    quantization: Optional[str] = "8bit"

class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: AWSTrainingConfig = AWSTrainingConfig()
    ):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validate_aws_setup()
        
    def _validate_aws_setup(self):
        from shallowflow.utils.aws_utils import AWSManager
        self.aws_manager = AWSManager()
        self.aws_manager.validate_setup()
        
    def _setup_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )