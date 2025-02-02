import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from shallowflow.optimizations import LoRALayer
import wandb
import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional
import argparse

@dataclass
class Config:
    # Model parameters
    model_name: str = "gpt2"
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: list = None
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 5
    gradient_accumulation_steps: int = 4
    
    # Regularization
    label_smoothing: float = 0.05
    dropout: float = 0.2
    attention_dropout: float = 0.2
    
    # Misc
    seed: int = 42
    save_steps: int = 500
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 with LoRA')
    
    # Model parameters
    parser.add_argument('--model_name', default='gpt2', help='Base model to use')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank of LoRA adaptors')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha scaling factor for LoRA')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Fraction of steps for warmup')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping norm')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay factor')
    
    # Regularization
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--attention_dropout', type=float, default=0.2, help='Attention dropout rate')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity')
    parser.add_argument('--output_dir', default='outputs/lora', help='Output directory for checkpoints')
    
    return parser.parse_args()

def replace_attention_layers(model, rank, alpha):
    """Replace attention layers with LoRA layers"""
    for name, module in model.named_modules():
        if "c_attn" in name or "c_proj" in name:
            if isinstance(module, torch.nn.Linear):
                # Create LoRA layer using the existing linear layer
                lora_layer = LoRALayer(
                    base_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=0.1,
                )
                
                # Replace the layer
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, lora_layer)
    
    # Print parameter counts after replacement
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nAfter LoRA replacement:")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")
    print(f"LoRA params: {trainable:,} ({100 * trainable / total:.4f}% of total)")
    
    return model

def count_parameters(model):
    """Count trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, block_size):
        self.input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        self.block_size = block_size

    def __getitem__(self, idx):
        block_start = idx
        block_end = min(idx + self.block_size, len(self.input_ids))
        
        input_ids = self.input_ids[block_start:block_end]
        attention_mask = self.attention_mask[block_start:block_end]
        
        # Pad if necessary
        if len(input_ids) < self.block_size:
            pad_length = self.block_size - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

    def __len__(self):
        return len(self.input_ids) - self.block_size + 1

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update model configuration
    config = AutoConfig.from_pretrained(args.model_name)
    config.dropout = args.dropout
    config.attention_dropout = args.attention_dropout
    
    # Initialize base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
    )
    
    # Replace attention layers with LoRA layers
    model = replace_attention_layers(model, args.lora_rank, args.lora_alpha)
    
    # Count parameters
    trainable_params, total_params = count_parameters(model)
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing (note: this will increase computation time)")
        model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    # Load and preprocess Shakespeare dataset
    print("Loading Shakespeare dataset...")
    with open('data/datasets/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize the entire text
    print("Tokenizing text...")
    encodings = tokenizer(
        text,
        truncation=True,
        max_length=None,
        return_tensors=None
    )
    
    # Create dataset with overlapping blocks
    block_size = 128
    dataset = TextDataset(encodings, block_size)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Calculate number of update steps
    total_steps = len(train_dataset) // (args.batch_size * args.grad_accum_steps) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'input_ids': torch.stack([d['input_ids'] for d in x]).to(device),
            'attention_mask': torch.stack([d['attention_mask'] for d in x]).to(device),
            'labels': torch.stack([d['labels'] for d in x]).to(device)
        }
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.stack([d['input_ids'] for d in x]).to(device),
            'attention_mask': torch.stack([d['attention_mask'] for d in x]).to(device),
            'labels': torch.stack([d['labels'] for d in x]).to(device)
        }
    )
    
    print(f"Dataset sizes: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Initialize wandb
    if args.wandb_entity:
        wandb.init(
            project="gpt2-lora-shakespeare",
            entity=args.wandb_entity,
            config={
                # Model config
                "model_name": args.model_name,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "trainable_params": trainable_params,
                "total_params": total_params,
                
                # Training config
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "max_grad_norm": args.max_grad_norm,
                
                # Regularization
                "dropout": args.dropout,
                "attention_dropout": args.attention_dropout,
                "label_smoothing": args.label_smoothing,
                
                # Dataset info
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "sequence_length": block_size,
                
                # Hardware info
                "device": str(device),
                "mixed_precision": args.mixed_precision,
                "gradient_checkpointing": args.gradient_checkpointing
            },
            tags=["lora", "shakespeare", "gpt2"]
        )
        
        # Log model architecture diagram
        wandb.watch(model, log="all", log_freq=100)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        epoch_steps = 0
        
        # Track metrics for each epoch
        epoch_train_loss = 0
        epoch_grad_norm = 0
        
        for batch_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # Track loss
            total_loss += loss.item() * args.grad_accum_steps
            epoch_train_loss += loss.item() * args.grad_accum_steps
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                epoch_grad_norm += grad_norm
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                epoch_steps += 1
                
                if args.wandb_entity:
                    # Log training metrics
                    wandb.log({
                        "train/loss": loss.item() * args.grad_accum_steps,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/gradient_norm": grad_norm,
                        "train/epoch": epoch,
                        "train/step": epoch_steps,
                        "train/global_step": epoch * len(train_loader) + batch_idx,
                        
                        # Resource utilization
                        "system/gpu_memory_used": torch.cuda.memory_allocated() / 1024**2,
                        "system/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2
                    })
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.2e}, GradNorm: {grad_norm:.2f}")
        
        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_grad_norm = epoch_grad_norm / epoch_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_perplexity = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_perplexity += torch.exp(outputs.loss).item()
        
        val_loss /= len(val_loader)
        val_perplexity /= len(val_loader)
        
        print(f"Epoch {epoch} completed. Validation loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        if args.wandb_entity:
            # Log epoch metrics
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/validation_loss": val_loss,
                "epoch/perplexity": val_perplexity,
                "epoch/avg_gradient_norm": avg_grad_norm,
                "epoch/learning_rate": scheduler.get_last_lr()[0],
                "epoch/current": epoch,
                "epoch/total": args.num_epochs
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the LoRA weights
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            
            if args.wandb_entity:
                wandb.run.summary["best_validation_loss"] = best_val_loss
                wandb.run.summary["best_validation_perplexity"] = val_perplexity
                wandb.run.summary["best_model_epoch"] = epoch
    
    # Save the final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    if args.wandb_entity:
        # Log final metrics
        wandb.run.summary["final_validation_loss"] = val_loss
        wandb.run.summary["final_validation_perplexity"] = val_perplexity
        wandb.run.summary["total_training_steps"] = epoch_steps * args.num_epochs
        wandb.finish()

if __name__ == "__main__":
    main()