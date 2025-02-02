from transformers import AutoModelForCausalLM, AutoTokenizer
from shallowflow.trainer import LocalGPUTrainer
from datasets import load_dataset
from types import SimpleNamespace

def main():
    # Configure for GTX 1660
    config = SimpleNamespace(
        batch_size=8,
        mixed_precision=True,
        gradient_checkpointing=True,
        learning_rate=5e-5
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Initialize trainer (wandb tracking disabled)
    trainer = LocalGPUTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        use_wandb=False  # Disable wandb tracking
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