#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Execute the Python script
python ../neurons/validator.py \
    --netuid 141 \
    --subtensor.network test \
    --wallet.name test_wallet \
    --wallet.hotkey test_hotkey \
    --axon.port 8091
