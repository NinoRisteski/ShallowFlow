<p align="center">
  <img src="assets/shallowflow.png" alt="Alt text">
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

Citations:
[1] https://www.atlassian.com/work-management/project-management/project-objectives
[2] https://arxiv.org/html/2404.08573v1
[3] https://www.superannotate.com/blog/llm-fine-tuning
[4] https://www.anodot.com/learning-center/aws-cost-optimization/ec2/g4/
[5] https://hyperight.com/4-pillars-to-effective-training-of-large-language-models/
