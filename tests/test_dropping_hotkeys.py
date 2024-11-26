import os
import pytest
import shutil
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


def delete_db_path():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


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


def test_hotkey_drop():
    delete_db_path()

    top_reward = 0.8
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

    rewards = pipeline(
        energies=energies, rewards=rewards, top_reward=top_reward, job=job
    )

    assert job.best_hotkey == "d"

    # Now we need to remove the best hotkey from the store to ensure that the pipeline is working correctly
    job.hotkeys = ["a", "b", "c"]
    store.update(job=job)

    job = store.get_queue(ready=False).queue[0]

    energies = torch.Tensor([0, 0, 0])
    rewards = torch.zeros(len(energies))

    rewards = pipeline(
        energies=energies, rewards=rewards, top_reward=top_reward, job=job
    )
    assert (rewards == 0).all()
    assert job.best_hotkey == "d"
    assert len(job.hotkeys) == 3
