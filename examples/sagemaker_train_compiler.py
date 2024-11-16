import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                       help="gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # SageMaker environment variables
    training_dir = os.environ.get("SM_CHANNEL_TRAINING", "data/datasets/tiny_shakespeare.txt")
    model_dir = os.environ.get("SM_MODEL_DIR", "model")
    num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load Tiny Shakespeare dataset
    dataset = load_dataset("tiny_shakespeare", split="train")
    
    # Training arguments optimized for compiler
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=16,
        optim="adamw_torch_xla",  # Optimized for Training Compiler
        dataloader_num_workers=4,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(model_dir)

# Required for distributed training
def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()