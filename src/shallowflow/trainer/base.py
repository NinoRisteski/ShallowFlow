from dataclasses import dataclass
import torch

@dataclass
class AWSTrainingConfig:
    instance_type: str = "g4dn.xlarge"
    gpu_memory: int = 16  # GB
    max_batch_size: int = 32
    gradient_accumulation_steps: int = 4

class BaseTrainer:
    def __init__(
        self,
        model,
        config: AWSTrainingConfig = AWSTrainingConfig()
    ):
        self.model = model
        self.config = config
        self._validate_aws_setup()