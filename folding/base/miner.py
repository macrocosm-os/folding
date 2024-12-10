# The MIT License (MIT)
# Copyright © 2024 Macrocosmos AI.

import argparse
from folding.utils.logger import logger
from folding.protocol import PingSynapse
from folding.base.neuron import BaseFolding
from folding.__init__ import __spec_version__

from atom.base.miner import BaseMinerNeuron as AtomBaseMinerNeuron


class BaseMinerNeuron(AtomBaseMinerNeuron, BaseFolding):
    """
    Base class for Bittensor miners.
    """

    def spec_version(self):
        return __spec_version__

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
