#!/bin/bash

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Conda first."
    exit 1
fi

# Create a venv
conda env create -f environment.yml
conda activate folding

# Install auxiliary packages
apt-get update
apt-get install build-essential cmake libfftw3-dev vim npm -y
npm install -g pm2 -y

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
else
    CUDA_AVAILABLE=false
    echo "You are creating an environment without a CUDA compatible GPU. This will cause issues."
fi

# Install folding
pip install -e .