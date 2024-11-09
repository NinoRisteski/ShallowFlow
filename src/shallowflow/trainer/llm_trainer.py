# src/shallowflow/trainer/llm_trainer.py
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..utils.config import TrainingConfig
from ..optimizations import LoRALayer, Quantizer
from ..utils.memory import MemoryTracker

class LLMTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.memory_tracker = MemoryTracker(config.gpu_memory)
        
        if config.use_lora:
            self._apply_lora()
        if config.use_quantization:
            self.quantizer = Quantizer(bits=8)
            
    def _apply_lora(self):
        # Apply LoRA to model layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get layer dimensions
                in_features = module.in_features
                out_features = module.out_features
                
                # Replace with LoRA layer
                lora_layer = LoRALayer(
                    in_features,
                    out_features,
                    rank=self.config.lora_rank
                )
                setattr(self.model, name, lora_layer)
                
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        **kwargs
    ):
        self.model.to(self.device)
        optimizer = self._setup_optimizer()
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for batch in train_dataset:
                loss = self._training_step(batch)
                self._optimization_step(loss, optimizer)
                
            if eval_dataset:
                eval_loss = self._evaluate(eval_dataset)
                print(f"Epoch {epoch}: eval_loss = {eval_loss}")