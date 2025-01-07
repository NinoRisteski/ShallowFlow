from dataclasses import dataclass
from typing import Optional, List

@dataclass
class QuantizationConfig:
    dtype: str  # e.g., 'qint8', 'float16'
    method: str = "dynamic"  # e.g., 'dynamic', 'static'
    layers_to_quantize: Optional[List[str]] = None

@dataclass
class LLMConfig:
    model_name: str
    gpu_memory: int
    use_lora: bool
    device: str
    batch_size: int
    num_epochs: int
    lora_rank: int
    lora_config: 'LoRAConfig'
    quantization_config: QuantizationConfig
    use_quantization: bool = False
    learning_rate: float = 0.001

@dataclass
class TrainingConfig:
    quantization_config: QuantizationConfig
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32
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