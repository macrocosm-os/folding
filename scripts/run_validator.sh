#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Execute the Python script
python ../neurons/validator.py \
    --netuid 141 \
    --subtensor.network test \
    --wallet.name <test_coldkey> \
    --wallet.hotkey <test_hotkey> \
    --axon.port <your_port> \
    --neuron.queue_size <number of pdb_ids to submit> \
    --neuron.sample_size <number of miners per pdb_id> \
