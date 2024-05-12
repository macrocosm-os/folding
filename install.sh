pip install -r requirements.txt
apt-get update
apt-get install build-essential cmake libfftw3-dev vim npm
npm install -g pm2
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
source /usr/local/gromacs/bin/GMXRC

# Command to be added to .bashrc
COMMAND="source /usr/local/gromacs/bin/GMXRC"

# Check if the command is already in the .bashrc to avoid duplication
if ! grep -Fxq "$COMMAND" ~/.bashrc
then
    # Append the command to .bashrc if it's not already there
    echo "$COMMAND" >> ~/.bashrc
    echo "Added GROMACS initialization to .bashrc"
else
    echo "GROMACS initialization already in .bashrc"
fi
pip install -e .
