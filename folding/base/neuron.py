# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import bittensor as bt
from abc import ABC, abstractmethod
import os

import openmm
from tenacity import RetryError

# Sync calls set weights and also resyncs the metagraph.
from folding.utils.config import check_config, add_args, config
from folding.utils.misc import ttl_get_block
from folding import __spec_version__ as spec_version
from folding import __OPENMM_VERSION_TAG__
from folding.utils.ops import OpenMMException, load_pkl, write_pkl
from folding.mock import MockSubtensor, MockMetagraph
from folding.utils.logger import logger


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        logger.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        logger.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        if self.config.mock:
            self.wallet = bt.MockWallet(config=self.config)
            self.subtensor = MockSubtensor(self.config.netuid, wallet=self.wallet)
            self.metagraph = MockMetagraph(self.config.netuid, subtensor=self.subtensor)
        else:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)

            # Check OpenMM version if we are not in mock mode.
            self.check_openmm_version()
            self.setup_wandb_logging()

        logger.info(f"Wallet: {self.wallet}")
        logger.info(f"Subtensor: {self.subtensor}")
        logger.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0

    def check_openmm_version(self):
        """
        A method that enforces that the OpenMM version that is running the version specified in the __OPENMM_VERSION_TAG__.
        """
        try:
            self.openmm_version = openmm.__version__

            if __OPENMM_VERSION_TAG__ != self.openmm_version:
                raise OpenMMException(
                    f"OpenMM version mismatch. Installed == {self.openmm_version}. Please install OpenMM {__OPENMM_VERSION_TAG__}.*"
                )

        except Exception as e:
            raise e

        logger.success(f"Running OpenMM version: {self.openmm_version}")

    def setup_wandb_logging(self):
        if os.path.isfile(f"{self.config.neuron.full_path}/wandb_ids.pkl"):
            self.wandb_ids = load_pkl(
                f"{self.config.neuron.full_path}/wandb_ids.pkl", "rb"
            )
        else:
            self.wandb_ids = {}

    def add_wandb_id(self, pdb_id: str, wandb_id: str):
        self.wandb_ids[pdb_id] = wandb_id
        write_pkl(self.wandb_ids, f"{self.config.neuron.full_path}/wandb_ids.pkl", "wb")

    def remove_wandb_id(self, pdb_id: str):
        self.wandb_ids.pop(pdb_id)
        write_pkl(self.wandb_ids, f"{self.config.neuron.full_path}/wandb_ids.pkl", "wb")

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            try:
                logger.info("Attempting to set weights...")
                self.set_weights()
                logger.success("Weight setting successful!")
            except RetryError as e:
                logger.error(
                    f"Failed to set weights after retry attempts. Skipping for {self.config.neuron.epoch_length} blocks."
                )

        # Always save state.
        self.save_state()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.metagraph_resync_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Do not allow weight setting if the neuron is not a validator.
        if not self.metagraph.validator_permit[self.uid]:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def save_state(self):
        pass

    def load_state(self):
        logger.warning(
            "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        )
