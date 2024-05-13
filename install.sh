# Create a venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install auxiliary packages
apt-get update
apt-get install build-essential cmake libfftw3-dev vim npm -y
npm install -g pm2 -y

# download and unpack gromacs
wget ftp://ftp.gromacs.org/gromacs/gromacs-2024.1.tar.gz
tar xfz gromacs-2024.1.tar.gz
cd gromacs-2024.1
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
make
make check
make install

# Add GROMACS initialization to venv/bin/activate
COMMAND="source /usr/local/gromacs/bin/GMXRC"
cd ../..
# Check if the command is already in the venv/bin/activate to avoid duplication
if ! grep -Fxq "$COMMAND" .venv/bin/activate
then
    # Append the command to venv/bin/activate if it's not already there
    echo "$COMMAND" >> .venv/bin/activate
    echo "Added GROMACS initialization to .venv/bin/activate"
else
    echo "GROMACS initialization already in .venv/bin/activate"
fi

# Install folding
pip install -e .

deactivate
