#!/bin/bash

# Exit on error
set -e

# Default values
MODEL="gpt2"
DATASET="wikitext"
OUTPUT_DIR="results"
USE_LORA="true"
USE_QUANTIZATION="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL="$2"
            shift
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --no-lora)
            USE_LORA="false"
            shift
            ;;
        --no-quantization)
            USE_QUANTIZATION="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p $OUTPUT_DIR

# Activate virtual environment
source venv/bin/activate

# Run experiments
echo "Starting experiments..."

# Basic training
echo "Running basic training..."
python examples/basic/train_gpt2.py \
    --model $MODEL \
    --dataset $DATASET \
    --output-dir $OUTPUT_DIR/basic

# Training with LoRA
if [ "$USE_LORA" = "true" ]; then
    echo "Running LoRA training..."
    python examples/optimized/train_gpt2_lora.py \
        --model $MODEL \
        --dataset $DATASET \
        --output-dir $OUTPUT_DIR/lora
fi

# Training with Quantization
if [ "$USE_QUANTIZATION" = "true" ]; then
    echo "Running Quantized training..."
    python examples/optimized/train_quantized.py \
        --model $MODEL \
        --dataset $DATASET \
        --output-dir $OUTPUT_DIR/quantized
fi

# Combined optimization
if [ "$USE_LORA" = "true" ] && [ "$USE_QUANTIZATION" = "true" ]; then
    echo "Running combined optimization training..."
    python examples/optimized/train_gpt2_advanced.py \
        --model $MODEL \
        --dataset $DATASET \
        --output-dir $OUTPUT_DIR/combined \
        --use-lora \
        --use-quantization
fi

# Generate report
echo "Generating experiment report..."
python scripts/generate_report.py \
    --input-dir $OUTPUT_DIR \
    --output-file $OUTPUT_DIR/report.md

echo "Experiments complete! Results saved in $OUTPUT_DIR"