from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    model_name: str
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    use_lora: bool = False
    use_quantization: bool = False
    aws_instance: str = "g4dn.xlarge"
    gpu_memory: int = 16  # GB

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 8
    dropout: float = 0.1

@dataclass
class QuantizationConfig:
    bits: int = 8
    symmetric: bool = True