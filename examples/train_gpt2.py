import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from datasets import load_dataset
from shallowflow.trainer.local_trainer import LocalGPUTrainer
from dataclasses import dataclass
import argparse
import wandb
import os
import random
import numpy as np
from types import SimpleNamespace

@dataclass
class Config:
    batch_size: int
    learning_rate: float
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    patience: int = 3
    validation_split: float = 0.1
    save_steps: int = 500
    # Regularization parameters
    dropout: float = 0.1
    attention_dropout: float = 0.1
    label_smoothing: float = 0.1
    weight_decay: float = 0.01
    layer_lr_decay: float = 0.8  # Layer-wise learning rate decay
    swa_start: float = 0.75  # Start SWA at 75% of training
    swa_lr: float = 2e-5  # SWA learning rate

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Fraction of steps for warmup')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X steps')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--attention_dropout', type=float, default=0.2, help='Attention dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay factor')
    parser.add_argument('--layer_lr_decay', type=float, default=0.85, help='Layer-wise learning rate decay factor')
    parser.add_argument('--swa_start', type=float, default=0.75, help='When to start SWA (fraction of training)')
    parser.add_argument('--swa_lr', type=float, default=5e-5, help='SWA learning rate')
    
    # Model and data parameters
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine'], help='Learning rate scheduler')
    
    return parser.parse_args()

def load_shakespeare_dataset(file_path):
    """Load and preprocess the Shakespeare dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create a simple dataset with text chunks
    chunk_size = 1024
    chunks = []
    
    for i in range(0, len(text) - chunk_size + 1, chunk_size // 2):
        chunk = text[i:i + chunk_size]
        chunks.append({"text": chunk})
    
    # Split into train and validation
    random.shuffle(chunks)
    split_idx = int(len(chunks) * 0.9)
    return chunks[:split_idx], chunks[split_idx:]

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
            'labels': input_ids.clone()  # For causal language modeling
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update model configuration
    config = AutoConfig.from_pretrained("gpt2")
    config.dropout = args.dropout
    config.attention_dropout = args.attention_dropout
    
    # Initialize model with updated config
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        config=config,
    )
    
    # Only enable gradient checkpointing if explicitly requested
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
    
    # Create data loaders with simpler collate_fn since tensors are already created
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
    
    # Create trainer configuration
    trainer_config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_grad_norm': args.max_grad_norm,
        'gradient_accumulation_steps': args.grad_accum_steps,
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
        'patience': args.patience,
        'dropout': args.dropout,
        'attention_dropout': args.attention_dropout,
        'label_smoothing': args.label_smoothing,
        'weight_decay': args.weight_decay,
        'layer_lr_decay': args.layer_lr_decay,
        'swa_start': args.swa_start,
        'swa_lr': args.swa_lr,
        'save_steps': args.save_steps,
        'mixed_precision': args.mixed_precision,
        'gradient_checkpointing': args.gradient_checkpointing,
        'scheduler': args.scheduler
    }
    
    # Initialize trainer
    trainer = LocalGPUTrainer(
        model=model,
        tokenizer=tokenizer,
        config=SimpleNamespace(**trainer_config),
        entity=args.wandb_entity
    )
    
    # Train the model
    trainer.train(
        train_dataset=train_loader,
        eval_dataset=val_loader,
        num_epochs=args.num_epochs
    )
    
    trainer.finish()

if __name__ == '__main__':
    main()