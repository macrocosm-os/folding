# The MIT License (MIT)
# Copyright © 2024 Macrocosmos AI.

import time
import asyncio
import threading
import argparse
import traceback

import bittensor as bt

from folding.utils.logger import logger
from folding.protocol import PingSynapse
from folding.base.neuron import BaseFolding
from folding.utils.config import add_miner_args

from atom.base.miner import BaseMinerNeuron


class BaseMinerNeuron(BaseMinerNeuron, BaseFolding):
    """
    Base class for Bittensor miners.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def ping_forward(self, synapse: PingSynapse):
        """Respond to the validator with the necessary information about serving

        Args:
            self (PingSynapse): must attach "can_serve" and "available_compute"
        """

        logger.info(f"Received ping request from {synapse.dendrite.hotkey[:8]}")

        synapse.available_compute = self.max_workers - len(self.simulations)

        # TODO: add more conditions.
        if synapse.available_compute > 0:
            synapse.can_serve = True
            logger.success("Telling validator you can serve ✅")
        return synapse
