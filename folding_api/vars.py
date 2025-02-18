import argparse

import bittensor as bt
import structlog
from slowapi import Limiter
from slowapi.util import get_remote_address

from folding.utils.config import add_args, config
from folding_api.auth import APIKeyManager
from folding_api.chain import SubtensorService
from folding_api.config import Settings

# Create parser and add arguments
parser = argparse.ArgumentParser()
bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
bt.axon.add_args(parser)
add_args(None, parser)  # Add base neuron args

# Parse arguments and create config
bt_config = config(parser)

settings = Settings()
logger = structlog.get_logger()
# Add rate limiting
limiter = Limiter(key_func=get_remote_address)

api_key_manager = APIKeyManager()
subtensor_service = SubtensorService(
    config=bt_config
)  # Pass config to SubtensorService
