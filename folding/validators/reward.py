import os
import torch
import bittensor as bt
from typing import List, Dict

from folding.validators.protein import Protein
from folding.utils.data import DataExtractor
from folding.protocol import FoldingSynapse
from folding.rewards.energy import EnergyRewardModel


def parsing_reward_data(miner_data_directory: str, validator_data_directory: str):
    data_extractor = DataExtractor(
        miner_data_directory=miner_data_directory,
        validator_data_directory=validator_data_directory,
    )
    # run all methods
    data_extractor.energy(data_type="Potential")
    data_extractor.temperature(data_type="T-rest")
    data_extractor.pressure(data_type="Pressure")
    data_extractor.density(data_type="Density")
    data_extractor.prod_energy(data_type="Potential")
    # data_extractor.rmsd()

    return data_extractor.data


def apply_reward_pipeline(data: Dict):
    reward_pipeline = [EnergyRewardModel()]
    reward_events = []

    for model in reward_pipeline:
        event = model.apply(data=data)
        reward_events.append(event)

    return reward_events


def get_rewards(
    protein: Protein, responses: List[FoldingSynapse], uids: List[int]
) -> Dict:
    """Applies the reward model across each call. Unsuccessful responses are zeroed."""

    reward_data = {}
    for uid, resp in zip(uids, responses):
        # Output the reward data information into the terminal
        md_output_summary = {k: len(v) for k, v in resp.md_output.items()}
        bt.logging.info(
            f"uid {uid}:\nDendrite: {resp.dendrite}\nMD Output: {md_output_summary}\n"
        )

        miner_data_directory = os.path.join(
            protein.validator_directory, resp.axon.hotkey[:8]
        )

        if resp.dendrite.status_code != 200:
            bt.logging.error(
                f"uid {uid} failed with status code {resp.dendrite.status_code}"
            )
            reward_data[uid] = None
            continue

        # Must be done because gromacs only *reads* data, cannot take it in directly
        protein.save_files(
            files=resp.md_output,
            output_directory=miner_data_directory,
        )

        bt.logging.info(f"Parsing data for miner {miner_data_directory}")
        output_data = parsing_reward_data(
            miner_data_directory=miner_data_directory,
            validator_data_directory=protein.validator_directory,
        )
        reward_data[uid] = output_data

    reward_events = apply_reward_pipeline(data=reward_data)

    return reward_events

    # # Softmax rewards across samples.
    # successful_rewards_normalized = torch.softmax(
    #     torch.tensor(successful_rewards), dim=0
    # )

    # # Init zero rewards for all calls.
    # filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
    # filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

    # # Return the filled rewards.
    # return filled_rewards, filled_rewards_normalized
