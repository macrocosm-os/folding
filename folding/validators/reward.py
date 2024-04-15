import os
import torch
import bittensor as bt
from typing import List, Dict

from folding.validators.protein import Protein
from folding.utils.data import DataExtractor
from folding.protocol import FoldingSynapse


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

    return reward_data

    # # Softmax rewards across samples.
    # successful_rewards_normalized = torch.softmax(
    #     torch.tensor(successful_rewards), dim=0
    # )

    # # Init zero rewards for all calls.
    # filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
    # filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

    # # Fill reward tensor.
    # for idx, reward, reward_normalized in zip(
    #     successful_response_indices,
    #     successful_rewards,
    #     successful_rewards_normalized,
    # ):
    #     filled_rewards[idx] = reward
    #     filled_rewards_normalized[idx] = reward_normalized

    # # Return the filled rewards.
    # return filled_rewards, filled_rewards_normalized
