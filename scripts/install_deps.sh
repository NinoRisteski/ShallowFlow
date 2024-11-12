#!/bin/bash

# Exit on error
set -e

echo "Installing ShallowFlow dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install basic requirements
pip install -r requirements.txt

# Install development requirements if --dev flag is passed
if [ "$1" == "--dev" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project in development mode
pip install -e .

echo "Installation complete!"