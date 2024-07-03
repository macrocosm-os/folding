#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Execute the Python script
python ./neurons/validator.py \
    --netuid 141 \
    --subtensor.network test \
    --wallet.name testnet \
    --wallet.hotkey v1 \
    --neuron.queue_size 1 \
    --neuron.sample_size 2 \
    --neuron.update_interval 20 \
    --protein.max_steps 50000 \
    --wandb.off \
    # --mdrun_args.maxh 0.006 #about 20s
