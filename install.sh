# Create a venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install auxiliary packages
apt-get update
apt-get install build-essential cmake libfftw3-dev vim npm -y
npm install -g pm2 -y

# Determine the number of physical cores
if [ "$(uname)" == "Linux" ]; then
    NUM_CORES=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
    NUM_SOCKETS=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
    NUM_PHYSICAL_CORES=$((NUM_CORES * NUM_SOCKETS))
elif [ "$(uname)" == "Darwin" ]; then
    NUM_PHYSICAL_CORES=$(sysctl -n hw.physicalcpu)
else
    NUM_PHYSICAL_CORES=10
fi

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
else
    CUDA_AVAILABLE=false
fi

# Download and unpack GROMACS
wget ftp://ftp.gromacs.org/gromacs/gromacs-2024.1.tar.gz
tar xfz gromacs-2024.1.tar.gz
cd gromacs-2024.1
mkdir build
cd build

# Configure GROMACS with or without CUDA support based on availability
if [ "$CUDA_AVAILABLE" = true ]; then
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON -DGMX_GPU=CUDA -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
else
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
fi

make -j$NUM_PHYSICAL_CORES
make check
make install

echo "source /usr/local/gromacs/bin/GMXRC" >> ~/.bashrc
source ~/.bashrc

# Add GROMACS initialization to venv/bin/activate
COMMAND="source /usr/local/gromacs/bin/GMXRC"
cd ../..
# Check if the command is already in the venv/bin/activate to avoid duplication
if ! grep -Fxq "$COMMAND" .venv/bin/activate
then
    # Append the command to .venv/bin/activate if it's not already there
    echo "$COMMAND" >> .venv/bin/activate
    echo "Added GROMACS initialization to .venv/bin/activate"
else
    echo "GROMACS initialization already in .venv/bin/activate"
fi

# Install folding
pip install -e .