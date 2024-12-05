# The MIT License (MIT)
# Copyright © 2024 Macrocosmos AI

import os
import json
import copy
import torch
import asyncio
import argparse
import threading
import bittensor as bt
from pathlib import Path
from typing import List, Optional

from atom.base.validator import BaseValidatorNeuron as AtomBaseValidatorNeuron

from folding.utils.logger import logger
from folding.base.neuron import BaseFolding
from folding.utils.ops import print_on_retry
from folding.utils.config import add_validator_args
from folding.organic.validator import OrganicValidator

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_result

ROOT_DIR = Path(__file__).resolve().parents[2]


class BaseValidatorNeuron(AtomBaseValidatorNeuron, BaseFolding):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        self._organic_scoring: Optional[OrganicValidator] = None
        if not self.config.neuron.axon_off and not self.config.neuron.organic_disabled:
            self._organic_scoring = OrganicValidator(
                axon=self.axon,
                validator=self,
                synth_dataset=None,
                trigger=self.config.neuron.organic_trigger,
                trigger_frequency=self.config.neuron.organic_trigger_frequency,
                trigger_frequency_min=self.config.neuron.organic_trigger_frequency_min,
            )

            self.loop.create_task(self._organic_scoring.start_loop())
        else:
            logger.warning(
                "Organic scoring is not enabled. To enable, remove '--neuron.axon_off' and '--neuron.organic_disabled'"
            )

        self.load_and_merge_configs()

    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times
        wait=wait_fixed(1),  # Wait 1 second between retries
        retry=retry_if_result(
            lambda result: result is False
        ),  # Retry if the result is False
        after=print_on_retry,
    )
    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            logger.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = (
            torch.nn.functional.normalize(self.scores, p=1, dim=0).to("cpu").numpy()
        )

        logger.debug("raw_weights", raw_weights)
        logger.debug("raw_weight_uids", self.metagraph.uids)
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        logger.debug("processed_weights", processed_weights)
        logger.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        logger.debug("uint_weights", uint_weights)
        logger.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )

        return result

    async def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            logger.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Check if `uids` is already a tensor and clone it to avoid the warning.
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, uids_tensor, rewards
        ).to(self.device)
        logger.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)
        logger.debug(f"Updated moving avg scores: {self.scores}")

    def load_config_json(self):
        config_json_path = os.path.join(
            str(ROOT_DIR), "folding/utils/config_input.json"
        )
        with open(config_json_path, "r") as file:
            config = json.load(file)
        return config

    def load_and_merge_configs(self):
        json_config = self.load_config_json()
        self.config.protein.update(json_config)
