#!/bin/bash
# Execute the Python script
python ./neurons/validator.py \
    --netuid 141 \
    --subtensor.network test \
    --wallet.name folding_testnet \
    --wallet.hotkey v2 \
    --neuron.queue_size 1 \
    --neuron.sample_size 1 \
    --neuron.update_interval 60 \
    --neuron.epoch_length 100 \
    --protein.pdb_id 1ubq \
    --wandb.off true \
    --axon.port 8030 \
    --neuron.gjp_address 167.99.209.27:8030
