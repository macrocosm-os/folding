import time
import asyncio
import threading
import argparse
import traceback

import bittensor as bt

from folding.base.neuron import BaseNeuron
from folding.protocol import PingSynapse, ParticipationSynapse
from folding.utils.config import add_miner_args
from folding.utils.logger import logger


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            logger.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            logger.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(
            wallet=self.wallet,
            config=self.config,
        )

        # Attach determiners which functions are called when servicing a request.
        logger.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        ).attach(
            forward_fn=self.ping_forward,  # not sure if we need blacklist on this.
        ).attach(
            forward_fn=self.participation_forward,
        )
        logger.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

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

    def participation_forward(self, synapse: ParticipationSynapse):
        """Respond to the validator with the necessary information about participating in a specified job

        Args:
            self (ParticipationSynapse): must attach "is_participating"
        """
        pass

    def run(self):
        pass

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            logger.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            logger.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()

        # Load existing jobs from database to utilize available workers
        added_jobs = self.add_active_jobs_from_db()
        if added_jobs > 0:
            logger.success(f"Loaded {added_jobs} active jobs from database on startup")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        logger.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

    def set_weights(self):
        pass
