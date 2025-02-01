import torch
import wandb
from torch.cuda.amp import autocast
from typing import Optional, Dict
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim.swa_utils import AveragedModel, SWALR
import os

class LocalGPUTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config,
        project_name: str = "shallowflow-local",
        entity: Optional[str] = None,
        use_wandb: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.project_name = project_name
        self.entity = entity
        self.use_wandb = use_wandb
        
        # Initialize wandb if enabled
        if self.use_wandb:
            self.init_wandb()
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize optimizer with layer-wise learning rate decay
        self.optimizer = self._create_optimizer()
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join("checkpoints", self.project_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _create_optimizer(self):
        """Create optimizer with layer-wise learning rate decay"""
        layer_decay = getattr(self.config, 'layer_lr_decay', 0.8)
        
        if layer_decay < 1.0:
            # Get all transformer layers
            named_parameters = list(self.model.named_parameters())
            layers = [name for name, _ in named_parameters if 'transformer' in name]
            num_layers = len(set([name.split('.')[1] for name in layers if 'transformer.h.' in name]))
            
            # Create parameter groups with decaying learning rate
            optimizer_grouped_parameters = []
            # Track which parameters have been added
            params_set = set()
            
            # Group transformer layer parameters
            for layer_idx in range(num_layers):
                layer_params = {
                    name: param for name, param in named_parameters 
                    if f'transformer.h.{layer_idx}.' in name
                }
                if layer_params:
                    layer_lr = self.config.learning_rate * (layer_decay ** (num_layers - layer_idx - 1))
                    params = list(layer_params.values())
                    params_set.update(layer_params.keys())
                    optimizer_grouped_parameters.append({
                        "params": params,
                        "lr": layer_lr,
                        "weight_decay": self.config.weight_decay
                    })
            
            # Add embedding parameters
            embed_params = {
                name: param for name, param in named_parameters 
                if 'transformer.wte' in name or 'transformer.wpe' in name
            }
            if embed_params:
                params_set.update(embed_params.keys())
                optimizer_grouped_parameters.append({
                    "params": list(embed_params.values()),
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay
                })
            
            # Add remaining parameters that haven't been added yet
            remaining_params = {
                name: param for name, param in named_parameters 
                if name not in params_set
            }
            if remaining_params:
                optimizer_grouped_parameters.append({
                    "params": list(remaining_params.values()),
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay
                })
        else:
            # Simple AdamW without layer-wise decay
            optimizer_grouped_parameters = [
                {
                    "params": self.model.parameters(),
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay
                }
            ]
        
        return torch.optim.AdamW(optimizer_grouped_parameters)
        
    def init_wandb(self):
        """Initialize wandb tracking"""
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            config={
                "model_name": self.model.config.name_or_path,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_grad_norm": getattr(self.config, 'max_grad_norm', 1.0),
                "gradient_accumulation_steps": getattr(self.config, 'gradient_accumulation_steps', 1),
                "warmup_steps": getattr(self.config, 'warmup_steps', 0),
                "dropout": getattr(self.config, 'dropout', 0.1),
                "attention_dropout": getattr(self.config, 'attention_dropout', 0.1),
                "label_smoothing": getattr(self.config, 'label_smoothing', 0.1),
                "weight_decay": getattr(self.config, 'weight_decay', 0.01),
                "layer_lr_decay": getattr(self.config, 'layer_lr_decay', 1.0),
                "swa_start": getattr(self.config, 'swa_start', 0.75),
                "swa_lr": getattr(self.config, 'swa_lr', 2e-5),
                "gpu_type": "GTX 1660",
                "gpu_memory": "6GB",
                "mixed_precision": self.config.mixed_precision,
                "gradient_checkpointing": self.config.gradient_checkpointing
            }
        )
        
    def _compute_grad_norm(self):
        """Compute gradient norm safely"""
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters])
        ).item()
        return total_norm
        
    def _compute_loss(self, outputs, batch, label_smoothing=0.0):
        """Compute loss with label smoothing"""
        logits = outputs.logits
        labels = batch["labels"]
        
        if label_smoothing > 0:
            num_classes = logits.size(-1)
            smoothed_labels = torch.full_like(
                logits, label_smoothing / (num_classes - 1)
            ).to(logits.device)
            smoothed_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - label_smoothing)
            
            loss = -torch.sum(smoothed_labels * torch.log_softmax(logits, dim=-1), dim=-1)
            return loss.mean()
        else:
            return outputs.loss
        
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 3,
        num_training_steps: Optional[int] = None,
        num_warmup_steps: Optional[int] = None
    ):
        # Setup learning rate scheduler
        if num_training_steps is None:
            num_training_steps = len(train_dataset) * num_epochs
        if num_warmup_steps is None:
            num_warmup_steps = self.config.warmup_steps
            
        # Choose scheduler based on configuration
        if getattr(self.config, 'scheduler', 'linear') == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # Initialize SWA if enabled
        swa_start_step = int(num_training_steps * self.config.swa_start)
        if hasattr(self.config, 'swa_start'):
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=self.config.swa_lr
            )
        
        # Training loop with optional wandb logging
        best_eval_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_dataset):
                with autocast():
                    outputs = self.model(**batch)
                    loss = self._compute_loss(
                        outputs, 
                        batch, 
                        label_smoothing=self.config.label_smoothing
                    ) / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    grad_norm = self._compute_grad_norm()
                    if grad_norm > 0:  # Only clip if gradients exist
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Update SWA if enabled and after start step
                    if hasattr(self, 'swa_model') and global_step >= swa_start_step:
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()
                    else:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if self.use_wandb:
                        wandb.log({
                            "batch_loss": loss.item() * self.config.gradient_accumulation_steps,
                            "epoch": epoch,
                            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2,
                            "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "global_step": global_step,
                            "gradient_norm": grad_norm
                        })
                    else:
                        if batch_idx % 10 == 0:
                            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                                  f"LR: {self.scheduler.get_last_lr()[0]:.2e}, GradNorm: {grad_norm:.2f}")
                    
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(epoch, global_step)
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            avg_loss = total_loss / len(train_dataset)
            
            # Evaluate
            if eval_dataset:
                # Use SWA model for evaluation if enabled and after start step
                if hasattr(self, 'swa_model') and global_step >= swa_start_step:
                    eval_model = self.swa_model
                else:
                    eval_model = self.model
                    
                eval_loss = self._evaluate(eval_dataset, eval_model)
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, global_step, is_best=True)
                else:
                    patience_counter += 1
                
                if self.use_wandb:
                    wandb.log({
                        "eval_loss": eval_loss,
                        "epoch": epoch,
                        "epoch_loss": avg_loss
                    })
                else:
                    print(f"Epoch {epoch} completed. Train loss: {avg_loss:.4f}, Eval loss: {eval_loss:.4f}")
                
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                if self.use_wandb:
                    wandb.log({
                        "epoch_loss": avg_loss,
                        "epoch": epoch,
                    })
                else:
                    print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Update final model with SWA if enabled
        if hasattr(self, 'swa_model') and global_step >= swa_start_step:
            self.model = self.swa_model
    
    def _evaluate(self, eval_dataset, model=None):
        """Evaluate the model on the evaluation dataset"""
        if model is None:
            model = self.model
            
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                loss = self._compute_loss(outputs, batch, label_smoothing=self.config.label_smoothing)
                total_loss += loss.item()
                
        return total_loss / len(eval_dataset)
    
    def _save_checkpoint(self, epoch: int, global_step: int, is_best: bool = False):
        """Save a model checkpoint"""
        checkpoint_prefix = "best_model" if is_best else f"checkpoint-{global_step}"
        save_path = os.path.join(self.checkpoint_dir, checkpoint_prefix)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save optimizer and scheduler state
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'swa_model_state_dict': self.swa_model.state_dict() if hasattr(self, 'swa_model') else None,
        }, os.path.join(save_path, 'training_state.pt'))
    
    def finish(self):
        """Clean up wandb tracking if enabled"""
        if self.use_wandb:
            wandb.finish()