from typing import Optional, List, Dict, Any
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from ..optimizations import LoRALayer, Quantizer
from ..monitoring import MetricsTracker
from shallowflow.utils import MemoryManager, AWSManager, GTX1660Optimizer
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
        self.gpu_optimizer = GTX1660Optimizer()
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
            
    def _setup_optimizer(self) -> Optimizer:
        """Setup the optimizer with proper parameters."""
        optimizer_config = getattr(self.config, 'optimizer_config', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')
        lr = optimizer_config.get('learning_rate', 1e-4)
        
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
        return optimizer
        
    def _setup_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """Setup the learning rate scheduler."""
        scheduler_config = getattr(self.config, 'scheduler_config', {})
        scheduler_type = scheduler_config.get('type', None)
        
        if not scheduler_type:
            return None
            
        if scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=scheduler_config.get('start_factor', 1.0),
                end_factor=scheduler_config.get('end_factor', 0.1),
                total_iters=scheduler_config.get('total_iters', self.config.num_epochs)
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', self.config.num_epochs),
                eta_min=scheduler_config.get('eta_min', 0)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
    def _train_epoch(
        self,
        train_dataset,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler]
    ) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create optimized dataloader
        train_loader = self.gpu_optimizer.create_dataloader(train_dataset)
        
        # Get mixed precision setup if enabled
        if self.config.use_mixed_precision:
            model, scaler = self.gpu_optimizer.setup_mixed_precision(self.model)
        else:
            model = self.model.cuda()
            scaler = None
            
        for batch_idx, batch in enumerate(train_loader):
            # Check memory usage
            self.memory_manager.check_memory()
            
            # Forward pass with mixed precision if enabled
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self._compute_loss(batch)
                loss.backward()
                optimizer.step()
                
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % self.config.log_interval == 0:
                self.metrics_tracker.log_metrics({
                    'batch_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'memory_usage': self.gpu_optimizer.get_memory_stats()
                })
                
        # Step scheduler if provided
        if scheduler:
            scheduler.step()
            
        return {
            'epoch_loss': total_loss / num_batches,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
    def _evaluate(self, eval_dataset) -> Dict[str, Any]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Create optimized dataloader
        eval_loader = self.gpu_optimizer.create_dataloader(eval_dataset)
        
        with torch.no_grad():
            for batch in eval_loader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
        return {
            'eval_loss': total_loss / num_batches
        }
        
    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch)
            
        # Default loss computation for language models
        if isinstance(batch, torch.Tensor):
            inputs = batch
            labels = batch
        elif isinstance(batch, dict):
            inputs = batch.get('input_ids', batch.get('inputs'))
            labels = batch.get('labels', inputs)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
            
        outputs = self.model(inputs, labels=labels)
        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
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