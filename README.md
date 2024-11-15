<p align="center">
  <img src="assets/shallowflow1.png" alt="Alt text">
</p>

ShallowFlow is a distributed training framework designed for LLM training on cost-effective AWS GPU instances (g4dn.xlarge with NVIDIA T4). The project aims to make LLM training and fine-tuning accessible to developers with limited GPU resources.

## Core

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


## Install
```python
# Clone repository
git clone https://github.com/NinoRisteski/ShallowFlow.git
cd shallowflow

# Create virtual env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Conda env
conda env create -f environment.yml
conda activate shallowflow
```
## Setup
#### Your GPU
```python
# Environment var
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="shallowflow-training"  # Optional for tracking
```

#### AWS
```python
# Configure AWS credentials
aws configure

# Set AWS environment variables
export AWS_REGION=us-west-2
export AWS_INSTANCE_TYPE=g4dn.xlarge
```
#### WandB
```python
# Set WandB API key
export WANDB_API_KEY="your-wandb-api-key"
```

## Running ShallowFlow

#### Train on local GPU
```python
# Train on Tiny Shakespeare dataset
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --num_epochs 3 \
    --use_wandb
```
#### Train with Optimizations
```python
# Train with LoRA and Quantization
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --use_lora \
    --use_quantization \
    --use_wandb
```
#### Train on AWS
```python
# Train on AWS
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 8 \
    --use_aws \
    --use_wandb
```
## Monitoring 

```python
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
```python
# Monitor GPU usage
nvidia-smi

# Check training logs
tail -f logs/training.log
```
## Test Runs:
```python
# Fast testing configuration
python train.py \
    --model_name gpt2 \
    --dataset tiny_shakespeare \
    --batch_size 4 \
    --num_epochs 1 \
    --use_quantization
```
```python
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
ShallowFlow fills a specific niche by providing a practical solution for ML engineers and researchers who want to work with LLMs but don't have access to high-end GPU clusters, making distributed training more accessible and cost-effective.

## References

[1] Atlassian, "How to write project objectives and project goals," Atlassian Work Management Guide, 2024. [Online]. Available: https://www.atlassian.com/work-management/project-management/project-objectives

[2] A. Kumar et al., "Efficient Large Language Model Training Techniques," arXiv:2404.08573v1 [cs.LG], Apr. 2024.

[3] SuperAnnotate, "A Comprehensive Guide to LLM Fine-Tuning," SuperAnnotate Technical Blog, Mar. 2024. [Online]. Available: https://www.superannotate.com/blog/llm-fine-tuning

[4] Anodot, "AWS G4 Instance Cost Optimization Guide," Anodot Learning Center, 2024. [Online]. Available: https://www.anodot.com/learning-center/aws-cost-optimization/ec2/g4/

[5] Hyperight, "The 4 Pillars of Effective LLM Training," Hyperight Technical Resources, Feb. 2024. [Online]. Available: https://hyperight.com/4-pillars-to-effective-training-of-large-language-models/

[6] S. Böhm, "ShallowSpeed: Small scale distributed training of sequential deep learning models," GitHub Repository, 2024. [Online]. Available: https://github.com/siboehm/ShallowSpeed

Note: ShallowSpeed served as inspiration for this project, implementing similar concepts for distributed training but focused specifically on LLM training on cost-effective GPU setups.