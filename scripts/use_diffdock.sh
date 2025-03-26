#!/bin/bash

# Check if DiffDock directory already exists
if [ ! -d "DiffDock" ]; then
    git clone git@github.com:macrocosm-os/DiffDock.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone DiffDock repository. Please check your connection and permissions."
        exit 1
    else
        echo "Successfully cloned DiffDock repository."
    fi
else
    echo "DiffDock directory already exists. Skipping clone."
fi

# Continue with the rest of the script below
docker pull mccrinbcmacrocosmos/diffdock-api:latest
docker run -d --gpus all -p 8000:8000 -p 7860:7860 -p 8501:8501 macrocosmos/diffdock