#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Execute the Python script
python3 ./neurons/validator.py \
    --netuid 25 \
    --neuron.disable_set_weights \
    --subtensor.network finney \
    --wallet.name opentensor \
    --wallet.hotkey main \
    --neuron.queue_size 1 \
    --neuron.sample_size 1 \
    --neuron.update_interval 10 \ 
