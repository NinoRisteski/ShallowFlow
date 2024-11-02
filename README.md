# ShallowFlow
A simplified distributed training framework focusing on modern deep learning models with an emphasis on ease of use and monitoring.


shallowflow/
├── src/
│   ├── shallowflow/
│   │   ├── __init__.py
│   │   ├── trainer/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          
│   │   │   └── llm_trainer.py   
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── ddp.py          
│   │   │   └── fsdp.py         
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── memory.py       
│   │   │   ├── metrics.py      
│   │   │   └── aws_utils.py   
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       └── trackers.py     # AWS CloudWatch integration
├── examples/
│   ├── train_gpt2.py          # Single T4 GPU example
│   └── finetune_bert.py       # Fine-tuning example
├── tests/
├── pyproject.toml
└── README.md