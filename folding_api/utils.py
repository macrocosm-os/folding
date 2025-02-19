from collections import defaultdict
from typing import List

from bittensor import OrganicSynapse
from loguru import logger

from folding_api.schemas import FoldingSchema, FoldingReturn
from folding_api.vars import subtensor_service
from folding.protocol import OrganicSynapse

import bittensor as bt
from typing import Any, Dict


def is_validator(uid: int, metagraph: bt.metagraph, stake_needed=10_000) -> bool:
    """Checks if a UID on the subnet is a validator."""
    return metagraph.S[uid] >= stake_needed


def get_validator_data(metagraph: bt.metagraph) -> Dict[str, Dict[str, Any]]:
    """Retrieve validator data (hotkey, percent stake) from metagraph

    Returns:
        Dict[str, Dict[str, Any]]: Sorted dictionary of validator data based on stake.
    """

    total_stake = sum(
        stake for uid, stake in enumerate(metagraph.S) if is_validator(uid, metagraph)
    )

    validator_data = {
        hotkey: {
            "percent_stake": float(stake / total_stake),
            "stake": stake,
            "uid": uid,
        }
        for uid, (hotkey, stake) in enumerate(zip(metagraph.hotkeys, metagraph.S))
        if is_validator(uid, metagraph)
    }

    sorted_data = dict(
        sorted(
            validator_data.items(),
            key=lambda item: item[1]["percent_stake"],
            reverse=True,
        )
    )
    return sorted_data
