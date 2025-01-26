from typing import Optional, Dict, List 
import torch
from ..optimizations import LoRALayer, Quantizer
from ..monitoring import MetricsTracker
from shallowflow.utils import MemoryManager, AWSManager
from shallowflow.utils.config import TrainingConfig, AWSConfig  
 

class TrainingPipeline:
    def __init__(
        self,
        model,
        config: TrainingConfig,
        aws_config: Optional[AWSConfig] = None
    ):
        self.model = model
        self.config = config
        self.memory_manager = MemoryManager(config.gpu_memory)
        self.metrics_tracker = MetricsTracker(config.project_name)
        
        self.aws_manager = AWSManager(aws_config) if aws_config else None
        self._setup_optimizations()
        
    def _setup_optimizations(self):
        """Setup model optimizations like LoRA and quantization."""
        if self.config.use_lora:
            self._apply_lora()
        if self.config.use_quantization:
            self.quantizer = Quantizer(bits=8)
            
    def _apply_lora(self):
        """Apply LoRA optimization to model layers."""
        # Get LoRA configuration
        lora_config = getattr(self.config, 'lora_config', {})
        lora_rank = lora_config.get('rank', 8)  # default rank
        lora_alpha = lora_config.get('alpha', 16)  # default scaling factor
        lora_dropout = lora_config.get('dropout', 0.1)  # default dropout rate
        target_modules = lora_config.get('target_modules', ['query', 'value'])
        
        # Track original parameters for potential restoration
        self._original_modules = {}
        
        # Apply LoRA to model layers
        for name, module in self.model.named_modules():
            # Check if module should be adapted
            if any(target in name for target in target_modules):
                if isinstance(module, torch.nn.Linear):
                    try:
                        # Get layer dimensions
                        in_features = module.in_features
                        out_features = module.out_features
                        
                        # Validate dimensions
                        if not isinstance(in_features, int) or not isinstance(out_features, int):
                            raise TypeError(f"Invalid features type for module {name}")
                        if in_features <= 0 or out_features <= 0:
                            raise ValueError(f"Invalid features dimensions for module {name}")
                        
                        # Store original module
                        self._original_modules[name] = module
                        
                        # Create and apply LoRA layer
                        lora_layer = LoRALayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=lora_rank,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            merge_weights=False
                        )
                        
                        # Copy original weights
                        with torch.no_grad():
                            lora_layer.linear.weight.copy_(module.weight)
                            if module.bias is not None:
                                lora_layer.linear.bias.copy_(module.bias)
                        
                        # Replace module with LoRA layer
                        self._set_module(self.model, name, lora_layer)
                        
                    except Exception as e:
                        print(f"Failed to apply LoRA to module {name}: {str(e)}")
                        continue
                        
        # Log LoRA application
        self.metrics_tracker.log_metrics({
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'lora_target_modules': len(target_modules)
        })
        
    def _set_module(self, model: torch.nn.Module, name: str, module: torch.nn.Module):
        """Helper method to set a module at a specific path in the model."""
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            model = getattr(model, part)
        setattr(model, name_parts[-1], module)
            
    def train(
        self,
        train_dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        callbacks: List = None
    ):
        try:
            # Setup cloud resources if needed
            instance_info = self._setup_training_environment()
            
            optimizer = self._setup_optimizer()
            scheduler = self._setup_scheduler(optimizer)
            
            for epoch in range(self.config.num_epochs):
                epoch_metrics = self._train_epoch(
                    train_dataset,
                    optimizer,
                    scheduler
                )
                
                if eval_dataset:
                    eval_metrics = self._evaluate(eval_dataset)
                    
                metrics = {
                    **epoch_metrics,
                    **(eval_metrics if eval_dataset else {})
                }
                
                if self.aws_manager:
                    metrics['training_costs'] = self.aws_manager.get_training_costs()
                    
                self.metrics_tracker.log_metrics(metrics)
                
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup all resources including cloud instances."""
        if self.aws_manager:
            self.aws_manager.terminate_instance()
        
        if hasattr(self, 'model'):
            self.model.cpu()
            torch.cuda.empty_cache()