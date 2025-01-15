# Download and extract rqlite
curl -L https://github.com/rqlite/rqlite/releases/download/v8.36.3/rqlite-v8.36.3-linux-amd64.tar.gz -o rqlite-v8.36.3-linux-amd64.tar.gz
tar xvfz rqlite-v8.36.3-linux-amd64.tar.gz

# Create a directory for binaries in your home folder if it doesn't exist
mkdir -p ~/bin

# Copy rqlite executables to your bin directory
cp rqlite-v8.36.3-linux-amd64/rqlite* ~/bin/

# Add ~/bin to PATH if not already present
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc

# Reload bash configuration
source ~/.bashrc

# Clean up downloaded files
rm -rf rqlite-v8.36.3-linux-amd64*