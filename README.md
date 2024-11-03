# ShallowFlow

Efficient LLM training framework optimized for AWS T4 instances.

## Project Structure

```plaintext
shallowflow/
├── .github/
│   ├── workflows/
│   │   ├── tests.yml          # CI pipeline
│   │   └── lint.yml           # Code quality checks
│   └── assets/
│       └── project_diagram.png # Architecture diagram
├── data/
│   ├── __init__.py
│   ├── processors/
│   │   ├── __init__.py
│   │   └── text_processor.py
│   └── datasets/
│       └── .gitkeep           # For dataset caching
├── scripts/
│   ├── install_deps.sh
│   ├── setup_aws.sh
│   └── run_experiments.sh
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
│   │       └── trackers.py
├── tests/
│   ├── __init__.py
│   ├── test_trainer/
│   │   ├── test_base.py
│   │   └── test_llm_trainer.py
│   ├── test_strategies/
│   └── conftest.py
├── examples/
│   ├── train_gpt2.py
│   └── finetune_bert.py
├── .gitignore
├── .pre-commit-config.yaml
├── environment.yml
├── pyproject.toml
├── setup.py
└── README.md
```

## ShallowFlow Overview

ShallowFlow is a distributed training framework specifically designed for LLM training on cost-effective AWS GPU instances (g4dn.xlarge with NVIDIA T4). The project aims to make LLM training and fine-tuning accessible to developers with limited GPU resources.

## Core Purpose

**Primary Objectives**
- Enable efficient LLM training on single T4 GPU setups
- Optimize memory usage for resource-constrained environments
- Provide simplified interfaces for LLM fine-tuning
- Implement cost-effective training strategies

**Key Features**
- Parameter-efficient fine-tuning (PEFT) support
- Memory optimization techniques for T4 GPU
- AWS integration and cost monitoring
- Support for smaller, efficient models
- Built-in monitoring and evaluation tools

## Technical Benefits

**Resource Optimization**
- Utilizes 8-bit quantization for memory efficiency
- Implements gradient checkpointing
- Supports efficient model parallelism
- Optimizes for T4 GPU's 16GB memory constraint

**Cost Efficiency**
- Leverages AWS g4dn.xlarge ($0.526/hour)
- Implements spot instance support
- Provides cost monitoring and optimization
- Enables efficient resource utilization

## Project Goals

1. **Accessibility**: Make LLM training accessible to developers with limited resources
2. **Efficiency**: Optimize training for cost-effective GPU instances
3. **Simplicity**: Provide easy-to-use interfaces for LLM fine-tuning
4. **Scalability**: Enable scaling from single GPU to larger setups when needed

ShallowFlow fills a specific niche by providing a practical solution for ML engineers and researchers who want to work with LLMs but don't have access to high-end GPU clusters, making distributed training more accessible and cost-effective.

Citations:
[1] https://www.atlassian.com/work-management/project-management/project-objectives
[2] https://arxiv.org/html/2404.08573v1
[3] https://www.superannotate.com/blog/llm-fine-tuning
[4] https://www.anodot.com/learning-center/aws-cost-optimization/ec2/g4/
[5] https://hyperight.com/4-pillars-to-effective-training-of-large-language-models/