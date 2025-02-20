import bittensor as bt
from loguru import logger
import tenacity


class SubtensorService:
    def __init__(self, config=None):
        self.config = config
        # Initialize subtensor with config
        self.subtensor: bt.Subtensor = (
            bt.subtensor(config=self.config) if config else bt.subtensor()
        )
        self.metagraph: bt.Metagraph = self.subtensor.metagraph(
            self.config.netuid if config else 25, lite=False
        )
        self.wallet: bt.Wallet = (
            bt.wallet(config=self.config) if config else bt.wallet()
        )
        self.dendrite: bt.Dendrite = bt.dendrite(wallet=self.wallet)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    )
    def resync_metagraph(self):
        self.metagraph.sync(subtensor=self.subtensor)
        logger.info("metagraph_reloaded")
