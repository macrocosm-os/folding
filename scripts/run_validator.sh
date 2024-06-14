#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Execute the Python script
python ./neurons/validator.py \
	--netuid 141 \
	--subtensor.network test \
	--wallet.name testy_wally \
	--wallet.hotkey m1 \
	--axon.port 3000 \
	--neuron.queue_size 5 \
	--neuron.sample_size 5 \