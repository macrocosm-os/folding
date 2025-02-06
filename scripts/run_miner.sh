

# Execute the Python script
python3 ./neurons/miner.py \
    --netuid 25 \
    --subtensor.network finney \
    --wallet.name <your_coldkey> \
    --wallet.hotkey <your_hotkey> \
    --neuron.max_workers <number of processes to run on your machine> \
    --axon.port <your_port>

