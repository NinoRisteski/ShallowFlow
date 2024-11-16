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
        config: TrainingConfig,
        use_wandb: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.config = config
        self.use_wandb = use_wandb
        gpu_memory = getattr(config, 'gpu_memory', 16)  # Default to 16GB for T4
        self.memory_tracker = MemoryTracker(gpu_memory)
        if not hasattr(config, 'learning_rate'):
            config.learning_rate = 1e-4  
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        self.use_lora = getattr(config, 'use_lora', False) 
        if self.use_lora:
            self._apply_lora()
        if hasattr(config, 'use_quantization'):
            self.use_quantization = config.use_quantization
            if self.use_quantization:
                self._apply_quantization()
        else:
            self.use_quantization = False 
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.use_wandb:
            self._setup_wandb()
        
        if not hasattr(config, 'num_epochs'):
            config.num_epochs = 3  
        
    def _apply_lora(self):
        # Default LoRA settings if not specified
        lora_rank = 8  # default rank
        if hasattr(self.config, 'lora_config'):
            lora_rank = self.config.lora_config.rank
        
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
                    rank=lora_rank
                )
                setattr(self.model, name, lora_layer)
                
    def _apply_quantization(self):
        """
        Apply quantization to the model based on the training configuration.
        """
        # Default quantization settings
        default_config = {
            'method': 'dynamic',
            'bits': 8
        }

        quant_config = default_config
        if hasattr(self.config, 'quantization_config'):
            quant_config.update(self.config.quantization_config)

        import torch.quantization as quant
        self.model = quant.quantize_dynamic(
            self.model, 
            {torch.nn.Linear},
            dtype=torch.qint8 if quant_config['bits'] == 8 else torch.qint4
        )
        print(f"Applied {quant_config['method']} quantization with {quant_config['bits']} bits")
        
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
        
        # Add debugging information
        print("Available batch keys:", batch.keys())
        
        # More flexible input handling
        if isinstance(batch, torch.Tensor):
            inputs = {'input_ids': batch}
        elif isinstance(batch, dict):
            if 'input_ids' in batch:
                inputs = {'input_ids': batch['input_ids']}
            elif 'input' in batch:
                inputs = self.tokenizer(batch['input'], return_tensors='pt')
            elif 'text' in batch:
                # Tokenize the text and create labels from it
                inputs = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024)
                labels = inputs['input_ids'].clone()  
            else:
                raise ValueError(f"Batch must contain either 'input_ids', 'input', or 'text' key. Got keys: {list(batch.keys())}")
        else:
            raise ValueError(f"Batch must be either a tensor or a dictionary. Got type: {type(batch)}")
        
        if 'attention_mask' in batch:
            inputs['attention_mask'] = batch['attention_mask']
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = labels.to(self.device) 
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss

    def _setup_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def finish_wandb(self):
        import wandb
        wandb.finish()

    def _setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        import wandb
        wandb.init(project="shallowflow")

    def _evaluate(self, eval_dataset):
        """
        Evaluate the model on the evaluation dataset
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                if isinstance(batch, torch.Tensor):
                    inputs = {'input_ids': batch}
                elif isinstance(batch, dict):
                    if 'input_ids' in batch:
                        inputs = {'input_ids': batch['input_ids']}
                    elif 'text' in batch:
                        inputs = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024)
                        labels = inputs['input_ids'].clone()
                    else:
                        raise ValueError(f"Batch must contain either 'input_ids' or 'text' key. Got keys: {list(batch.keys())}")
                else:
                    raise ValueError(f"Batch must be either a tensor or a dictionary. Got type: {type(batch)}")
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                outputs = self.model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')