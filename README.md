<p align="center">
  <img src="assets/shallowflow1.png" alt="Alt text">
</p>

ShallowFlow is a distributed training framework designed for LLM training on cost-effective AWS GPU instances (g4dn.xlarge with NVIDIA T4). The project aims to make LLM training and fine-tuning accessible to developers with limited GPU resources.

## Core

**Objectives**
- Enable LLM training on single T4 GPU setups
- Optimize memory usage for resource-constrained environments
- Provide simplified interfaces for LLM fine-tuning
- Implement cost-effective training strategies

**Features**
- Parameter-efficient fine-tuning (PEFT) support
- Memory optimization techniques for T4 GPU
- AWS integration and cost monitoring
- Support for smaller, efficient models
- Built-in monitoring and evaluation tools

## Benefits

**Optimization**
- Utilizes 8-bit quantization for memory efficiency
- Implements gradient checkpointing
- Supports efficient model parallelism
- Optimizes for T4 GPU's 16GB memory constraint

**Efficiency**
- Leverages AWS g4dn.xlarge ($0.526/hour)
- Implements spot instance support
- Provides cost monitoring and optimization
- Enables efficient resource utilization

## Goals

1. **Accessibility**: Make LLM training accessible to developers with limited resources
2. **Efficiency**: Optimize training for cost-effective GPU instances
3. **Simplicity**: Provide easy-to-use interfaces for LLM fine-tuning
4. **Scalability**: Enable scaling from single GPU to larger setups when needed

ShallowFlow fills a specific niche by providing a practical solution for ML engineers and researchers who want to work with LLMs but don't have access to high-end GPU clusters, making distributed training more accessible and cost-effective.

## Install
``` 
# Clone repository
git clone https://github.com/yourusername/shallowflow.git
cd shallowflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# For conda environment
conda env create -f environment.yml
conda activate shallowflow
```
## Setup
1. Your GPU
```
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="shallowflow-training"  # Optional for tracking
```

2. AWS
```
# Configure AWS credentials
aws configure

# Set AWS environment variables
export AWS_REGION=us-west-2
export AWS_INSTANCE_TYPE=g4dn.xlarge
```
3. WandB
```
# Set WandB API key
export WANDB_API_KEY="your-wandb-api-key"
```

## Running ShallowFlow

1. Train on local GPU
```
# Train on Tiny Shakespeare dataset
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --num_epochs 3 \
    --use_wandb
```
2. Train with Optimizations
```
# Train with LoRA and Quantization
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --use_lora \
    --use_quantization \
    --use_wandb
```
3. Train on AWS
```
# Train on AWS
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --use_aws \
    --use_wandb
```
## Monitoring 

```
# Set up wandb
wandb login

# Run training with monitoring
python train.py \
    --model_name gpt2 \
    --use_wandb \
    --wandb_project "my-project" \
    --wandb_entity "my-username"
```
Or: 
```
# Monitor GPU usage
nvidia-smi

# Check training logs
tail -f logs/training.log
```
## Example Test Run:
```
# Fast testing configuration
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 4 \
    --num_epochs 1 \
    --use_quantization
```
```
# Complete training configuration
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --num_epochs 3 \
    --use_lora \
    --use_quantization \
    --use_wandb \
    --output_dir trained_models
```

```
Citations:
[1] https://www.atlassian.com/work-management/project-management/project-objectives
[2] https://arxiv.org/html/2404.08573v1
[3] https://www.superannotate.com/blog/llm-fine-tuning
[4] https://www.anodot.com/learning-center/aws-cost-optimization/ec2/g4/
[5] https://hyperight.com/4-pillars-to-effective-training-of-large-language-models/
```