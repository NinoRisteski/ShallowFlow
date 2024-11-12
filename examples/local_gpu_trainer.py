from transformers import AutoModelForCausalLM, AutoTokenizer
from shallowflow.trainer import LocalGPUTrainer, GTX1660Config
from datasets import load_dataset

def main():
    # Configure for GTX 1660
    config = GTX1660Config(
        batch_size=8,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Initialize trainer with wandb tracking
    trainer = LocalGPUTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        project_name="shallowflow-local",
        entity="your-wandb-username"  # Optional
    )
    
    # Load tiny shakespeare dataset
    dataset = load_dataset("tiny_shakespeare")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    try:
        # Train with monitoring
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=3
        )
    finally:
        # Ensure wandb tracking is properly closed
        trainer.finish()

if __name__ == "__main__":
    main()