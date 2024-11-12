import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shallowflow import LLMTrainer, TrainingConfig
from shallowflow.optimizations import QuantizationConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 with Quantization')
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--output_dir', default='outputs_quantized')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize configs
    training_config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_quantization=True
    )
    
    quant_config = QuantizationConfig(
        bits=args.bits,
        symmetric=True
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize trainer with quantization
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        quantization_config=quant_config
    )
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Train
    trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    
    # Save quantized model
    trainer.save_quantized_model(args.output_dir)

if __name__ == "__main__":
    main()