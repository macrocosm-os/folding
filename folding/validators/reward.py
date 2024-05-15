import os
import pandas as pd
import numpy as np
import bittensor as bt
from typing import List, Dict

from folding.validators.protein import Protein
from folding.utils.data import DataExtractor
from folding.protocol import FoldingSynapse
from folding.rewards.reward import RewardEvent
from folding.rewards.energy import EnergyRewardModel
from folding.rewards.rmsd import RMSDRewardModel


def parsing_miner_data(
    miner_data_directory: str, validator_data_directory: str
) -> pd.DataFrame:
    """Runs specific GROMACS commands to extract physical properties from the simulation data. Each command produces a tabular file which is loaded as a pandas DataFrame.

    Args:
        miner_data_directory (str): _description_
        validator_data_directory (str): _description_

    Returns:
        pd.DataFrame: Contains the tabular data extracted from the simulation.
    """
    data_extractor = DataExtractor(
        miner_data_directory=miner_data_directory,
        validator_data_directory=validator_data_directory,
    )

    data_extractor.energy(data_type="Potential")
    return data_extractor.data["energy"]


def get_energies(protein: Protein, responses: List[FoldingSynapse], uids: List[int]):
    """Takes all the data from reponse synapses, applies the reward pipeline, and aggregates the rewards
    into a single torch.FloatTensor.

    Returns:
        torch.FloatTensor: A tensor of rewards for each miner.
    """

    energies = np.zeros(len(uids))
    for i, (uid, resp) in enumerate(zip(uids, responses)):
        # Ensures that the md_outputs from the miners are parsed correctly
        try:
            if not protein.process_md_output(
                md_output=resp.md_output, hotkey=resp.axon.hotkey
            ):
                continue

            if resp.dendrite.status_code != 200:
                bt.logging.info(
                    f"uid {uid} responded with status code {resp.dendrite.status_code}"
                )
                continue

            output_data = parsing_miner_data(
                miner_data_directory=protein.get_miner_data_directory(resp.axon.hotkey),
                validator_data_directory=protein.validator_directory,
            )
            energies[i] = output_data.iloc[-1]["energy"]

        except Exception as E:
            # If any of the above methods have an error, we will catch here.
            bt.logging.error(
                f"Failed to parse miner data for uid {uid} with error: {E}"
            )
            continue

    return energies
