import os
import pytest
import torch
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from folding.store import PandasJobStore, Job
from folding.rewards.linear_reward import divide_decreasing


ROOT_PATH = Path(__file__).parent
DB_PATH = os.path.join(ROOT_PATH, "mock_data")
PDB = "ab12"
FF = "charmm27"
BOX = "cubic"
WATER = "tip3p"


def pipeline(
    top_reward: float,
    energies: torch.Tensor,
    rewards: torch.Tensor,
    job: Job,
    fn: Callable = divide_decreasing,
) -> torch.Tensor:
    # Find if there are any indicies that are the same as the best value
    remaining_miners = {}
    for index in torch.nonzero(energies):
        # There could be multiple max energies.
        # The best energy could be the one that is saved in the store. We reward this old miner, as they don't need to reply anymore.
        if (energies[index] == job.best_loss) or (
            index == job.hotkeys.index(job.best_hotkey)
        ):
            rewards[index] = top_reward
        else:
            remaining_miners[index] = energies[index]

    # The amount of reward that is distributed to the remaining miners MUST be less than the reward given to the top miners.
    num_reminaing_miners = len(remaining_miners)
    if num_reminaing_miners > 1:
        sorted_remaining_miners = dict(
            sorted(remaining_miners.items(), key=lambda item: item[1])
        )  # sort smallest to largest

        # Apply a fixed decrease in reward on the remaining non-zero miners.
        rewards_per_miner = fn(
            amount_to_distribute=1 - top_reward,
            number_of_elements=num_reminaing_miners,
        )
        for index, r in zip(sorted_remaining_miners.keys(), rewards_per_miner):
            rewards[index] = r
    else:
        for index in remaining_miners.keys():
            rewards[index] = 1 - top_reward

    return rewards


def insert_single_job_in_store():
    store = PandasJobStore(db_path=DB_PATH, force_create=True)
    info = {
        "pdb": PDB,
        "ff": FF,
        "box": BOX,
        "water": WATER,
        "hotkeys": ["a", "b", "c", "d"],
        "created_at": pd.Timestamp.now().floor("s"),
        "updated_at": pd.Timestamp.now().floor("s"),
    }

    job = Job(**info)
    store.insert(
        pdb=job.pdb, ff=job.ff, box=job.box, water=job.water, hotkeys=job.hotkeys
    )

    return store, job


def determine_bests(job: Job, energies: torch.Tensor):
    best_index = energies.argmin()
    best_loss = energies[best_index].item()
    best_hotkey = job.hotkeys[best_index]

    return best_index, best_loss, best_hotkey


@pytest.mark.parametrize("top_reward", [0.80, 0.60])
def test_linear_decrease(top_reward: float):
    store, job = insert_single_job_in_store()

    energies = torch.Tensor([0, 0, 0, -10000])
    rewards = torch.zeros(len(energies))

    best_index, best_loss, best_hotkey = determine_bests(job=job, energies=energies)

    job.update(
        loss=best_loss,
        hotkey=best_hotkey,
        commit_hash="",
        gro_hash="",
    )

    assert job.best_loss == energies[-1]

    rewards = pipeline(
        top_reward=top_reward,
        energies=energies,
        rewards=rewards,
        job=job,
        fn=divide_decreasing,
    )

    assert torch.equal(rewards, torch.Tensor([0, 0, 0, top_reward]))

    # Apply a situation where the first returned set of energies is the best.
    energies = torch.Tensor([-9500, -3200, -1000, -9000])
    rewards = torch.zeros(len(energies))
    best_index, best_loss, best_hotkey = determine_bests(job=job, energies=energies)

    job.update(
        loss=best_loss,
        hotkey=best_hotkey,
        commit_hash="",
        gro_hash="",
    )

    assert job.best_loss == -10000

    rewards = pipeline(
        top_reward=top_reward,
        energies=energies,
        rewards=rewards,
        job=job,
        fn=divide_decreasing,
    )

    assert rewards[2] == 0
    assert rewards[-1] == top_reward
    assert np.isclose(1 - top_reward, sum(rewards[:-1]))

    energies = torch.Tensor([-10000, -9200, -9999, -9800])
    rewards = torch.zeros(len(energies))
    best_index, best_loss, best_hotkey = determine_bests(job=job, energies=energies)

    job.update(
        loss=best_loss,
        hotkey=best_hotkey,
        commit_hash="",
        gro_hash="",
    )

    assert job.best_loss == -10000

    rewards = pipeline(
        top_reward=top_reward,
        energies=energies,
        rewards=rewards,
        job=job,
        fn=divide_decreasing,
    )

    assert rewards[0] == top_reward
    assert rewards[-1] == top_reward

    energies = torch.Tensor([-11000, -9200, -9999, -9800])
    rewards = torch.zeros(len(energies))
    best_index, best_loss, best_hotkey = determine_bests(job=job, energies=energies)

    job.update(
        loss=best_loss,
        hotkey=best_hotkey,
        commit_hash="",
        gro_hash="",
    )

    assert job.best_loss == -11000

    rewards = pipeline(
        top_reward=top_reward,
        energies=energies,
        rewards=rewards,
        job=job,
        fn=divide_decreasing,
    )

    assert rewards[0] == top_reward
    assert best_hotkey == "a"


if __name__ == "__main__":
    test_linear_decrease(top_reward=0.80)
