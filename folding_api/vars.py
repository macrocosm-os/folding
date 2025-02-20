import argparse
import bittensor as bt
import structlog
from slowapi import Limiter
from slowapi.util import get_remote_address

# from folding_api.auth import APIKeyManager
from folding_api.chain import SubtensorService
from folding_api.validator_registry import ValidatorRegistry

parser = argparse.ArgumentParser()

# Add arguments
bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
parser.add_argument("--netuid", type=int, help="Subnet netuid", default=25)


bt_config = bt.config(parser)

logger = structlog.get_logger()
# Add rate limiting
limiter = Limiter(key_func=get_remote_address)

# api_key_manager = APIKeyManager()
subtensor_service = SubtensorService(
    config=bt_config
)  # Pass config to SubtensorService

validator_registry = ValidatorRegistry(metagraph=subtensor_service.metagraph)
