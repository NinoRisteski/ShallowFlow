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
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate for training')
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
    parser.add_argument('--quantization_bits', type=int, default=8,
                       help='Number of bits for quantization')
    parser.add_argument('--quantization_method', type=str, default='dynamic',
                       choices=['dynamic', 'static'],
                       help='Quantization method to use')
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
            spot_instance=True,
            quantization_config={
                'use_quantization': args.use_quantization,
                'bits': args.quantization_bits,
                'method': args.quantization_method
            }
        )
        trainer = LLMTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            use_wandb=args.use_wandb
        )
    else:
        # Local GTX 1660 Configuration
        config = GTX1660Config(
            batch_size=args.batch_size,
            mixed_precision=True,
            gradient_checkpointing=True,
            quantization_config={
                'use_quantization': args.use_quantization,
                'bits': args.quantization_bits,
                'method': args.quantization_method
            }
        )
        trainer = LocalGPUTrainer(
            model_name=args.model_name,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            use_wandb=args.use_wandb
        )

    # Apply optimizations
    if args.use_lora:
        trainer._apply_lora()
    if args.use_quantization:
        trainer._apply_quantization()

    try:
        # Train model
        trainer.train(
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )

        # Save model
        if hasattr(trainer, 'save_model'):
            trainer.save_model(args.output_dir)
        else:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Model and tokenizer saved to {args.output_dir}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        if hasattr(trainer, 'save_model'):
            trainer.save_model(args.output_dir + "_interrupted")
        else:
            model.save_pretrained(args.output_dir + "_interrupted")
            tokenizer.save_pretrained(args.output_dir + "_interrupted")

    finally:
        if args.use_wandb:
            if hasattr(trainer, 'finish_wandb'):
                trainer.finish_wandb()
            else:
                import wandb
                wandb.finish()

if __name__ == "__main__":
    main()
