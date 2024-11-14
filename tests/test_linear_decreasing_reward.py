import os
import pytest
import torch
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from folding.store import SQLiteJobStore, Job
from folding.rewards.reward_pipeline import reward_pipeline as pipeline

ROOT_PATH = Path(__file__).parent
DB_PATH = os.path.join(ROOT_PATH, "mock_data")
PDB = "ab12"
FF = "charmm27"
BOX = "cubic"
WATER = "tip3p"


def insert_single_job_in_store():
    epsilon = 1e-5

    store = SQLiteJobStore(db_path=DB_PATH, force_create=True)
    info = {
        "pdb": PDB,
        "ff": FF,
        "box": BOX,
        "water": WATER,
        "hotkeys": ["a", "b", "c", "d"],
        "created_at": pd.Timestamp.now().floor("s"),
        "updated_at": pd.Timestamp.now().floor("s"),
        "epsilon": epsilon,
    }

    job = Job(**info)
    store.insert(
        pdb=job.pdb,
        ff=job.ff,
        box=job.box,
        water=job.water,
        hotkeys=job.hotkeys,
        epsilon=epsilon,
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
    )

    assert torch.equal(rewards, torch.Tensor([0, 0, 0, 1]))

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
    )

    assert rewards[0] == top_reward
    assert best_hotkey == "a"

    energies = torch.Tensor([0, 0, 0, 0])
    rewards = torch.zeros(len(energies))
    best_index, best_loss, best_hotkey = determine_bests(job=job, energies=energies)

    job.update(
        loss=best_loss,
        hotkey=best_hotkey,
        commit_hash="",
        gro_hash="",
    )

    assert job.best_loss == -11000
    assert job.best_hotkey == "a"

    rewards = pipeline(
        top_reward=top_reward,
        energies=energies,
        rewards=rewards,
        job=job,
    )

    assert rewards[job.hotkeys.index(job.best_hotkey)] == 1

    energies = torch.Tensor([0, -9800, 0, 0])
    rewards = torch.zeros(len(energies))
    best_index, best_loss, best_hotkey = determine_bests(job=job, energies=energies)

    job.update(
        loss=best_loss,
        hotkey=best_hotkey,
        commit_hash="",
        gro_hash="",
    )

    assert job.best_loss == -11000
    assert job.best_hotkey == "a"

    rewards = pipeline(
        top_reward=top_reward,
        energies=energies,
        rewards=rewards,
        job=job,
    )

    assert rewards[job.hotkeys.index(job.best_hotkey)] == top_reward
    assert rewards[1] == 1 - top_reward


if __name__ == "__main__":
    test_linear_decrease(top_reward=0.80)
