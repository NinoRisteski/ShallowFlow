import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shallowflow.trainer import LocalGPUTrainer, LLMTrainer
from shallowflow.utils.gpu_optimizations import GTX1660Config
from shallowflow.utils.aws_utils import AWSConfig

def parse_args():
    parser = argparse.ArgumentParser(description='ShallowFlow Training')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model name or path')
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare',
                       help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--use_aws', action='store_true',
                       help='Use AWS training')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases tracking')
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA optimization')
    parser.add_argument('--use_quantization', action='store_true',
                       help='Use 8-bit quantization')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Configure training based on hardware
    if args.use_aws:
        # AWS Training Configuration
        config = AWSConfig(
            instance_type="g4dn.xlarge",
            spot_instance=True
        )
        trainer = LLMTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
    else:
        # Local GTX 1660 Configuration
        config = GTX1660Config(
            batch_size=args.batch_size,
            mixed_precision=True,
            gradient_checkpointing=True
        )
        trainer = LocalGPUTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            enable_wandb=args.use_wandb
        )

    # Apply optimizations
    if args.use_lora:
        trainer.enable_lora()
    if args.use_quantization:
        trainer.enable_quantization()

    try:
        # Train model
        trainer.train(
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )

        # Save model
        trainer.save_model(args.output_dir)

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        trainer.save_model(args.output_dir + "_interrupted")

    finally:
        if args.use_wandb:
            trainer.finish_wandb()

if __name__ == "__main__":
    main()
