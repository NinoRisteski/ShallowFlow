# ShallowFlow

Efficient LLM training framework optimized for AWS T4 instances.

## Project Structure

```plaintext
shallowflow/
├── src/
│   ├── shallowflow/
│   │   ├── __init__.py
│   │   ├── trainer/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Base trainer class
│   │   │   └── llm_trainer.py   # AWS-optimized LLM trainer
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── ddp.py          # Single GPU T4 strategy
│   │   │   └── fsdp.py         # Future multi-GPU support
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── memory.py       # T4 memory optimization
│   │   │   ├── metrics.py      # Training metrics
│   │   │   └── aws_utils.py    # AWS-specific utilities
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       └── trackers.py     # AWS CloudWatch integration
├── examples/
│   ├── train_gpt2.py          # Single T4 GPU example
│   └── finetune_bert.py       # Fine-tuning example
├── tests/
├── pyproject.toml
└── README.md