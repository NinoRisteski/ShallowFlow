import torch
import torch.distributed as dist
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..utils.config import TrainingConfig
from ..optimizations import LoRALayer, Quantizer
from ..utils.memory import MemoryTracker
from typing import Union, List, Dict, Any
import boto3

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)


        
        if config.use_lora:
            self._apply_lora()
        if config.use_quantization:
            self._apply_quantization()
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _apply_lora(self):
        # Apply LoRA to model layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get layer dimensions
                in_features = module.in_features
                out_features = module.out_features
                
                if not isinstance(in_features, int) or not isinstance(out_features, int):
                    raise TypeError("Features must be integers")
                if in_features <= 0 or out_features <= 0:
                    raise ValueError("Features must be positive")
                
                # Replace with LoRA layer
                lora_layer = LoRALayer(
                    in_features,
                    out_features,
                    rank=self.config.lora_config.rank
                )
                setattr(self.model, name, lora_layer)
                
    def _apply_quantization(self):
        """
        Apply quantization to the model based on the training configuration.
        """
        if hasattr(self.config, 'quantization_config'):
            quant_config = self.config.quantization_config
            # Example: Apply dynamic quantization
            import torch.quantization as quant
            self.model = quant.quantize_dynamic(
                self.model, 
                {torch.nn.Linear},  # Specify layers to quantize
                dtype=torch.qint8
            )
            print("Quantization applied to the model.")
        else:
            raise ValueError("Quantization configuration is missing in the training config.")
        
    def _setup_distributed(self):
        """Initialize distributed process group"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        torch.cuda.set_device(self.config.rank)
        dist.barrier()  # Add synchronization point
        
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
            for batch_idx, batch in enumerate(train_dataset):
                loss = self._training_step(batch)
                self._optimization_step(loss, optimizer)
                
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
            if eval_dataset:
                eval_loss = self._evaluate(eval_dataset)
                print(f"Epoch {epoch}: eval_loss = {eval_loss}")

    def process_text(
        self,
        text: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw text input
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens
        )

    def cleanup(self):
        torch.cuda.empty_cache()
        if hasattr(self, 'wrapped_model'):
            del self.wrapped_model

    def launch_instance(self) -> Dict:
        launch_template = {
            'InstanceType': self.config.instance_type,
            'ImageId': self._get_deep_learning_ami(),
            'BlockDeviceMappings': [{
                'DeviceName': '/dev/xvda',
                'Ebs': {
                    'VolumeSize': self.config.volume_size,
                    'VolumeType': 'gp3'
                }
            }]
        }
        
        if self.config.spot_instance:
            try:
                return self._launch_spot_instance(launch_template)
            except boto3.exceptions.Boto3Error as e:
                raise RuntimeError(f"AWS operation failed: {str(e)}")
        return self._launch_on_demand_instance(launch_template)

    def process_dataset(self, dataset):
        if self.max_samples:
            dataset = dataset[:self.max_samples]
        
        for batch in self._create_batches(dataset):
            yield self.text_processor.process_batch(batch)

    def training_step(self, batch):
        """
        Public method to handle a training step.
        """
        loss = self._training_step(batch)
        return loss

    def _training_step(self, batch):
        """
        Private method that performs the actual training step.
        """
        self.model.train()
        inputs = self.tokenizer(batch['input'], return_tensors='pt').to(self.config.device)
        labels = batch['labels'].to(self.config.device)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss