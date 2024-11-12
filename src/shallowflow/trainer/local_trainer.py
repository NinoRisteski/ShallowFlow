import torch
import wandb
from torch.cuda.amp import autocast
from typing import Optional, Dict

class LocalGPUTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config,
        project_name: str = "shallowflow-local",
        entity: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.project_name = project_name
        self.entity = entity
        
        # Initialize wandb
        self.init_wandb()
        
        self.scaler = torch.cuda.amp.GradScaler()
        
    def init_wandb(self):
        """Initialize wandb tracking"""
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            config={
                "model_name": self.model.config.name_or_path,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "gpu_type": "GTX 1660",
                "gpu_memory": "6GB",
                "mixed_precision": self.config.mixed_precision,
                "gradient_checkpointing": self.config.gradient_checkpointing
            }
        )
        
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 3
    ):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training loop with wandb logging
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataset):
                batch = {k: v.cuda() for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                # Log metrics to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batch_idx": batch_idx
                })
            
            avg_loss = total_loss / len(train_dataset)
            
            # Log epoch metrics
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch": epoch,
            })
            
            if eval_dataset:
                eval_loss = self._evaluate(eval_dataset)
                wandb.log({
                    "eval_loss": eval_loss,
                    "epoch": epoch
                })
                
    def _evaluate(self, eval_dataset):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
        return total_loss / len(eval_dataset)
    
    def finish(self):
        """Clean up wandb tracking"""
        wandb.finish()