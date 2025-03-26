#!/bin/bash

# Check if DiffDock directory already exists
if [ ! -d "DiffDock" ]; then
    git clone https://github.com/macrocosm-os/DiffDock.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone DiffDock repository. Please check your connection and permissions."
        exit 1
    else
        echo "Successfully cloned DiffDock repository."
    fi
else
    echo "DiffDock directory already exists. Skipping clone."
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."
    
    # Check if system is Ubuntu
    if [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then
        echo "Ubuntu detected. Installing Docker..."
        
        # Add Docker's official GPG key:
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl
        sudo install -m 0755 -d /etc/apt/keyrings
        sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc

        # Add the repository to Apt sources:
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
          $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
          sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        
        # Install Docker packages
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        
        # Start and enable Docker service
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Add current user to docker group to avoid using sudo
        sudo usermod -aG docker $USER
        echo "Docker installed successfully. You may need to log out and back in for group changes to take effect."
    else
        echo "This script only supports automatic Docker installation on Ubuntu."
        echo "Please install Docker manually according to your OS instructions."
        exit 1
    fi
else
    echo "Docker is already installed."
fi

# Continue with the rest of the script below
docker pull mccrinbcmacrocosmos/diffdock-api:latest
docker run -d --gpus all -p 8000:8000 -p 7860:7860 -p 8501:8501 macrocosmos/diffdock