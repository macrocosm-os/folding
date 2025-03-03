import random
import time
from typing import ClassVar, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, model_validator
import bittensor as bt
from folding_api.vars import subtensor_service


class Validator(BaseModel):
    uid: int
    stake: float
    address: str
    hotkey: str
    timeout: int = (
        1  # starting cooldown in seconds; doubles on failure (capped at 86400)
    )
    available_at: float = (
        0.0  # Unix timestamp indicating when the validator is next available
    )

    def update_failure(self, status_code: int) -> int:
        """
        Update the validator's failure count based on the operation status.

        - If the operation was successful (status_code == 200), decrease the failure count (ensuring it doesn't go below 0).
        - If the operation failed, increase the failure count.
        - If the failure count exceeds MAX_FAILURES, return 1 to indicate the validator should be deactivated.
        Otherwise, return 0.
        """
        current_time = time.time()
        if status_code == 200:
            self.timeout = 1
            self.available_at = current_time
        else:
            self.timeout = min(self.timeout * 4, 86400)
            self.available_at = current_time + self.timeout

    def is_available(self):
        """
        Check if the validator is available based on its cooldown.
        """
        return time.time() >= self.available_at

    def update_validator_info(self, stake: float, address: str, hotkey: str):
        self.stake = stake
        self.address = address
        self.hotkey = hotkey


class ValidatorRegistry(BaseModel):
    """
    Class to store the success of forwards to validator axons.
    Validators that routinely fail to respond to scoring requests are removed.
    """

    # Using a default factory ensures validators is always a dict.
    validators: dict[int, Validator] = Field(default_factory=dict)
    spot_checking_rate: ClassVar[float] = 0.0
    max_retries: ClassVar[int] = 4

    @model_validator(mode="after")
    def create_validator_list(
        cls, v: "ValidatorRegistry", metagraph=subtensor_service.metagraph
    ) -> "ValidatorRegistry":
        validator_uids = np.where(metagraph.stake >= 10000)[0].tolist()
        validator_addresses = [
            subtensor_service.get_commitment(uid) for uid in validator_uids
        ]
        validator_stakes = [metagraph.stake[uid] for uid in validator_uids]
        validator_hotkeys = [metagraph.hotkeys[uid] for uid in validator_uids]
        v.validators = {
            uid: Validator(uid=uid, stake=stake, address=address, hotkey=hotkey)
            for uid, stake, address, hotkey in zip(
                validator_uids, validator_stakes, validator_addresses, validator_hotkeys
            )
        }
        return v

    def update_validators(self):
        metagraph = subtensor_service.metagraph
        validator_uids = np.where(metagraph.stake >= 10000)[0].tolist()
        validator_addresses = [
            subtensor_service.get_commitment(uid) for uid in validator_uids
        ]
        validator_stakes = [metagraph.stake[uid] for uid in validator_uids]
        validator_hotkeys = [metagraph.hotkeys[uid] for uid in validator_uids]
        for uid, stake, address, hotkey in zip(
            validator_uids, validator_stakes, validator_addresses, validator_hotkeys
        ):
            self.validators[uid].update_validator_info(stake, address, hotkey)

    def get_available_validators(self) -> List[int]:
        """
        Given a list of validators, return only those that are not in their cooldown period.
        """
        return [
            uid
            for uid, validator in self.validators.items()
            if validator.is_available()
        ]

    def get_available_axons(self, k=1) -> Optional[dict[int, Validator]]:
        """
        Returns a list of tuples (uid, axon, hotkey) for a randomly selected validator based on stake weighting,
        if spot checking conditions are met. Otherwise, returns None.
        """
        for _ in range(self.max_retries):
            validator_list = self.get_available_validators()
            if validator_list:
                break
            else:
                time.sleep(5)
        if not validator_list:
            logger.error(f"Could not find available validator after {self.max_retries}")
            return None
        weights = [self.validators[uid].stake for uid in validator_list]
        chosen = random.choices(validator_list, weights=weights, k=k)
        return {uid: self.validators[uid] for uid in chosen}

    def update_validators_failure(self, uid: int, response_code: int) -> None:
        """
        Update a specific validator's failure count based on the response code.
        If the validator's failure count exceeds the maximum allowed failures,
        the validator is removed from the registry.
        """
        if uid in self.validators:
            self.validators[uid].update_failure(response_code)
