# Download and extract rqlite
curl -L https://github.com/rqlite/rqlite/releases/download/v8.36.3/rqlite-v8.36.3-linux-amd64.tar.gz -o rqlite-v8.36.3-linux-amd64.tar.gz
tar xvfz rqlite-v8.36.3-linux-amd64.tar.gz  --no-same-owner

# Create a directory for binaries in your home folder if it doesn't exist
mkdir -p ~/bin

# Copy rqlite executables to your bin directory
cp rqlite-v8.36.3-linux-amd64/rqlite* ~/bin/

# Add $HOME/bin to PATH in .bashrc if not already present
echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc

# Update PATH for the current session to include $HOME/bin
export PATH="$PATH:$HOME/bin"

# Attempt to reload .bashrc (may not work in non-interactive shells, but PATH is already set)
source ~/.bashrc >/dev/null 2>&1 || true

# Clean up downloaded files
rm -rf rqlite-v8.36.3-linux-amd64*