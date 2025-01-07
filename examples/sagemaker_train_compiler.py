import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shallowflow.trainer import LLMTrainer
from shallowflow.utils.config import TrainingConfig, QuantizationConfig
from shallowflow.utils.sagemaker_utils import SageMakerConfig, SageMakerManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2",
                       help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA optimization")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 8-bit quantization")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # SageMaker environment variables
    training_dir = os.environ.get("SM_CHANNEL_TRAINING", "data")
    model_dir = os.environ.get("SM_MODEL_DIR", "model")
    
    # Configure quantization
    quantization_config = QuantizationConfig(
        method="dynamic",
        dtype="float16"
    )
    
    # Configure training settings
    training_config = TrainingConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_lora=args.use_lora,
        use_quantization=args.use_quantization,
        aws_instance="ml.g4dn.xlarge",
        gpu_memory=16,  # G4dn.xlarge has 16GB GPU memory
        quantization_config=quantization_config
    )

    # Configure SageMaker
    sagemaker_config = SageMakerConfig(
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Initialize SageMaker manager
    sagemaker_manager = SageMakerManager(sagemaker_config)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize trainer
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # Load dataset
    dataset = load_dataset("tiny_shakespeare", split="train")
    
    # Setup SageMaker training
    estimator = sagemaker_manager.setup_training(
        model_name=args.model_name,
        script_path=__file__
    )
    
    try:
        # Train the model
        trainer.train(
            train_dataset=dataset,
            num_epochs=args.epochs
        )
        
        # Save the model
        trainer.save_model(model_dir)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if hasattr(trainer, 'cleanup'):
            trainer.cleanup()

if __name__ == "__main__":
    main()