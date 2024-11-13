from dataclasses import dataclass

@dataclass
class LLMConfig:
    device: str
    model_name: str
    batch_size: int
    num_epochs: int

@dataclass
class DDPConfig:
    world_size: int
    rank: int
    backend: str    

@dataclass
class FSDPConfig:
    min_num_params: int

@dataclass
class DeepSpeedConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int


