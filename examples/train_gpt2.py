import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shallowflow.trainer.llm_trainer import LLMTrainer
from shallowflow.utils.config import TrainingConfig, QuantizationConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 with ShallowFlow')
    parser.add_argument('--model_name', default='gpt2', help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--output_dir', default='outputs/test_run')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize config
    quantization_config = QuantizationConfig(dtype='qint8')
    config = TrainingConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=128,
        quantization_config=quantization_config,
        use_quantization=False
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load a small dataset for testing
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
    
    # Initialize trainer
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Process dataset
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        # Convert list of tensors to a single tensor
        if isinstance(outputs['input_ids'], list):
            outputs['input_ids'] = torch.tensor(outputs['input_ids'])
        if isinstance(outputs['attention_mask'], list):
            outputs['attention_mask'] = torch.tensor(outputs['attention_mask'])
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Train
    trainer.train(
        train_dataset=tokenized_dataset
    )
    
    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()