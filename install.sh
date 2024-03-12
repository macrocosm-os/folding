apt-get update
apt-get install build-essential cmake libfftw3-dev vim npm
npm install pm2
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
# TODO: copy the last command into the bashrc file so it runs on every startup