from dataclasses import dataclass

@dataclass
class QuantizationConfig:
    method: str  # e.g., 'dynamic', 'static'
    dtype: str    # e.g., 'qint8', 'float16'

class LLMConfig:
    def __init__(
        self,
        model_name: str,
        gpu_memory: int,
        use_lora: bool,
        device: str,
        batch_size: int,
        num_epochs: int,
        lora_rank: int,
        lora_config,
        quantization_config: QuantizationConfig,
        use_quantization: bool = False,
        learning_rate: float = 0.001
    ):
        self.model_name = model_name
        self.gpu_memory = gpu_memory
        self.use_lora = use_lora
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lora_rank = lora_rank
        self.lora_config = lora_config
        self.use_quantization = use_quantization
        self.quantization_config = quantization_config 
        self.learning_rate = learning_rate

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

class QuantizationConfig:
    def __init__(self, dtype, layers_to_quantize=None, some_other_param=None):
        self.dtype = dtype
        self.layers_to_quantize = layers_to_quantize or []
        self.some_other_param = some_other_param