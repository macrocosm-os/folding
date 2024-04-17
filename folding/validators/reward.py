import os
import torch
import bittensor as bt
from typing import List, Dict

from folding.validators.protein import Protein
from folding.utils.data import DataExtractor
from folding.protocol import FoldingSynapse
from folding.rewards.reward import RewardEvent
from folding.rewards.energy import EnergyRewardModel
from folding.rewards.rmsd import RMSDRewardModel


def parsing_miner_data(miner_data_directory: str, validator_data_directory: str):
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
    data_extractor.rmsd()

    return data_extractor.data


def apply_reward_pipeline(data: Dict) -> List[RewardEvent]:
    reward_events = []
    reward_pipeline = [EnergyRewardModel(), RMSDRewardModel()]

    for model in reward_pipeline:
        event = model.apply(data=data)
        reward_events.append(event)

    return reward_events


def aggregate_rewards(reward_events: List[RewardEvent]):
    """Currently, the reward mechanism takes a mean over all rewards"""

    rewards = []
    events = {}

    for event in reward_events:
        rewards.append(list(event.rewards.values()))
        events.update(event.asdict())

    return torch.FloatTensor(rewards).mean(dim=0), events


def get_rewards(protein: Protein, responses: List[FoldingSynapse], uids: List[int]):
    """Takes all the data from reponse synapses, applies the reward pipeline, and aggregates the rewards
    into a single torch.FloatTensor.

    Returns:
        rewards, events: torch.FloatTensor([reward_value_each_uid]), Dict[event_information]
    """

    reward_data = {}
    for uid, resp in zip(uids, responses):
        md_output_summary = {k: len(v) for k, v in resp.md_output.items()}

        if not protein.process_md_output(
            md_output=resp.md_output, hotkey=resp.axon.hotkey
        ):
            reward_data[uid] = None
            continue

        if resp.dendrite.status_code != 200:
            bt.logging.error(
                f"uid {uid} failed with status code {resp.dendrite.status_code}"
            )
            reward_data[uid] = None
            continue

        output_data = parsing_miner_data(
            miner_data_directory=protein.get_miner_data_directory(resp.axon.hotkey),
            validator_data_directory=protein.validator_directory,
        )
        reward_data[uid] = output_data

    reward_events = apply_reward_pipeline(data=reward_data)
    rewards, events = aggregate_rewards(reward_events=reward_events)

    events.update(md_output_summary)  # Record the size of the files in md_output

    return rewards, events
