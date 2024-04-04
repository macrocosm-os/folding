module.exports = {
    apps: [
        {
            name: 'v1',
            script: 'neurons/validator.py',
            interpreter: '.venv/bin/python3',
            args: ["--netuid", "89",  "--subtensor.network", "test", "--wallet.name", "testy_wally", "--wallet.hotkey", "testy_hotkey5", "--axon.port", "40125", "--protein.pdb_id", "5oxe", 
            "--protein.max_steps", "10"],
        },
        // {
        //     name: 'm1',
        //     script: 'neurons/miner.py',
        //     interpreter: 'venv/bin/python3',
        //     args: ["--netuid", "89",  "--subtensor.network", "test", "--wallet.name", "testy_wally", "--wallet.hotkey", "testy_hotkey6", "--axon.port", "40126",],   
        // },
        {
            name: 'm2',
            script: 'neurons/miner.py',
            interpreter: '.venv/bin/python3',
            args: ["--netuid", "89",  "--subtensor.network", "test", "--wallet.name", "testy_wally", "--wallet.hotkey", "testy_hotkey7", "--axon.port", "40127",],   
        },
        // {
        //     name: 'm3',
        //     script: 'neurons/miner.py',
        //     interpreter: 'venv/bin/python3',
        //     args: ["--netuid", "89",  "--subtensor.network", "test", "--wallet.name", "testy_wally", "--wallet.hotkey", "testy_hotkey8", "--axon.port", "40128",],   
        // },
        // {
        //     name: 'm4',
        //     script: 'neurons/miner.py',
        //     interpreter: 'venv/bin/python3',
        //     args: ["--netuid", "89",  "--subtensor.network", "test", "--wallet.name", "testy_wally", "--wallet.hotkey", "testy_hotkey9", "--axon.port", "40129",],   
        // },
    ]
};

