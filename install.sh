#!/bin/bash


# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
else
    CUDA_AVAILABLE=false
    echo "❌ You are creating an environment without a CUDA compatible GPU. CUDA is a requirement. Not installing folding."
    exit 1
fi

# Check GCC version
REQUIRED_MAJOR_GCC_VERSION=11
GCC_VERSION=$(gcc --version | grep ^gcc | awk '{print $NF}')
GCC_MAJOR_VERSION=$(echo $GCC_VERSION | cut -d. -f1)
GCC_MINOR_VERSION=$(echo $GCC_VERSION | cut -d. -f2)

# Determine if the version is between 6 and 13.2 (inclusive) as recommended from cuda/gcc/openmm matching
if [ "$GCC_MAJOR_VERSION" -lt 6 ] || [ "$GCC_MAJOR_VERSION" -gt 13 ] || \
   { [ "$GCC_MAJOR_VERSION" -eq 13 ] && [ "$GCC_MINOR_VERSION" -gt 2 ]; }; then
    echo "❌ Warning: Your GCC version is $GCC_VERSION. Folding requires GCC version between 6 and 13.2 (inclusive). Not installing folding."
    exit 1
fi

# Check if GCC major version is NOT $REQUIRED_MAJOR_GCC_VERSION
if [ "$GCC_MAJOR_VERSION" -ne $REQUIRED_MAJOR_GCC_VERSION ]; then
    echo "❌ Warning: Your GCC version is $GCC_VERSION. The script expects GCC version $REQUIRED_MAJOR_GCC_VERSION. Not installing folding."
    exit 1
fi
# ensure pip is installed
python3.11 -m ensurepip

# Install poetry
pip install poetry

# Install the project dependencies
poetry install

# Install auxiliary packages
sudo apt-get update
sudo apt-get install build-essential cmake libfftw3-dev vim npm -y
sudo npm install -g pm2 -y
chmod +x install_rqlite.sh
./install_rqlite.sh
