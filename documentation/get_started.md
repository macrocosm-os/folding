# Get Started with SN25 

## Installation

Core software requirements include:
```
conda
poetry 
```

Core hard/firmware requirements include: 
```
linux
CUDA 
RTX 4090
```

To start, simply clone this repository and run the [install.sh](./install.sh) script. Below are steps to ensure your machine is running properly.

Firstly, you must install conda: 
```bash 
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```

Install wandb (optional as a miner): 
```bash
pip install wandb 
wandb login
```

We use a combination of `conda` and `poetry` to manage our environments. It is very important to create the environment with python 3.11 as this is necesssary for `bittensor` and `openmm`
```bash 
git clone https://github.com/macrocosm-os/folding.git
cd folding

conda create --name folding python=3.11
bash install.sh
```

Bash install will use poetry to build the environment correctly. 

## Launch Commands
### Validator

SN25 uses DigitalOcean S3 data buckets for data transfer. Therefore, the following environment variables need to be set in your system or application environment (`.env` file):
- `S3_REGION = "nyc3"`: The AWS region or S3-compatible region where the bucket is located.
- `S3_ENDPOINT = "https://nyc3.digitaloceanspaces.com"`: The endpoint URL for your S3-compatible service.
- `S3_BUCKET = sn25-folding-mainnet`: The name of the s3 bucket. 
- `S3_KEY`: Your S3 access key ID.
- `S3_SECRET`: Your S3 secret access key.

**As a validator, you must ask the Macrocosmos SN25 team for your personalized S3 access key and secret.** Reference `.env.example` for an example of how to set these variables.

There are many parameters that one can configure for a simulation. The base command-line args that are needed to run the validator are below. 
```bash
python neurons/validator.py
    --netuid <25/141>
    --subtensor.network <finney/test>
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --axon.port <your axon port> #VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```

### Miner
There are many parameters that one can configure for a simulation. The base command-line args that are needed to run the miner are below. 
```bash
python neurons/miner.py
    --netuid <25/141>
    --subtensor.network <finney/test>
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --neuron.max_workers <number of processes to run on your machine>
    --axon.port <your axon port> #VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```

Optionally, pm2 can be run for both the validator and the miner using our utility scripts found in pm2_configs. 
```bash 
pm2 start pm2_configs/miner.config.js
```
or 
```bash 
pm2 start pm2_configs/validator.config.js
```