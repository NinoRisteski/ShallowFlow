import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shallowflow import LLMTrainer, TrainingConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 with ShallowFlow')
    parser.add_argument('--model_name', default='gpt2', help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--output_dir', default='outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize config
    config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize trainer
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Train
    trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    
    # Save model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()