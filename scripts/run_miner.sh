#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Execute the Python script
python ../neurons/miner.py \
    --netuid 141 \
    --subtensor.network test \
    --wallet.name <your_coldkey> \
    --wallet.hotkey <your_hotkey> \
    --neuron.max_workers <number of processes to run on your machine> \
    --axon.port <your_port>