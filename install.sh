#!/bin/bash

# Function to install dependencies using apt-get
install_dependencies_apt() {
    apt-get update
    apt-get install build-essential cmake libfftw3-dev vim -y
}

# Function to install dependencies using Homebrew
install_dependencies_brew() {
    brew update
    brew install cmake fftw vim gromacs
}

# Check if make is available
if command -v make &> /dev/null
then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS. Installing dependencies using Homebrew."
        install_dependencies_brew
    else
        echo "Detected Linux. Installing dependencies using apt-get."
        install_dependencies_apt
    fi
else
    echo "make is not installed. Please install make and re-run the script."
    exit 1
fi

# Install pm2
npm install -g pm2 -y

# Create a venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt



# Install GROMACS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "GROMACS installed via Homebrew."
else
    # download and unpack gromacs for Linux
    wget ftp://ftp.gromacs.org/gromacs/gromacs-2024.1.tar.gz
    tar xfz gromacs-2024.1.tar.gz
    cd gromacs-2024.1
    mkdir build
    cd build
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
    make
    make check
    make install
    cd ../..
    
    echo "source /usr/local/gromacs/bin/GMXRC" >> ~/.bashrc
    
    # Add GROMACS initialization to venv/bin/activate
    COMMAND="source /usr/local/gromacs/bin/GMXRC"
    # Check if the command is already in the venv/bin/activate to avoid duplication
    if ! grep -Fxq "$COMMAND" .venv/bin/activate
    then
        # Append the command to .venv/bin/activate if it's not already there
        echo "$COMMAND" >> .venv/bin/activate
        echo "Added GROMACS initialization to .venv/bin/activate"
    else
        echo "GROMACS initialization already in .venv/bin/activate"
    fi
fi

# Install folding
pip install -e .