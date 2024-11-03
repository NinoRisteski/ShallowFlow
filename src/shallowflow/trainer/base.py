from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self._setup_device()
        
    def _setup_device(self):
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)