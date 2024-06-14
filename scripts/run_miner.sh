#!/bin/bash

# Activate the virtual environment
source /home/spunion/folding/venv/bin/activate

# Execute the Python script
python ./neurons/miner.py \
    --netuid 141 \
    --subtensor.network test \
    --wallet.name testy_wally \
    --wallet.hotkey m1 \
    --neuron.max_workers 2 \
    --axon.port 3001 \
    --logging.debug \