### This script is used to run a sensitivity search for a protein folding simulation.
### Used for benchmarking.
from folding.base.neuron import BaseNeuron
import os
import time
import wandb
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import bittensor as bt

import pandas as pd
from box import Box  # install using pip install box
import copy
import pickle as pkl

import plotly.express as px
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import traceback
import numpy as np

from folding.base.simulation import OpenMMSimulation

from folding.validators.hyperparameters import HyperParameters
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.validators.protein import Protein
from folding.validators.forward import (
    create_random_modifications_to_system_config,
    parse_config,
)
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter
import snowflake.connector
import random

from folding.utils.ops import (
    load_and_sample_random_pdb_ids,
    OpenMMException,
)

from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY

DIR_PATH = Path(__file__).resolve().parents[2]
INPUT_PDBS_PATH = "/home/paperspace/sergio/folding/subsets/large_proteins.csv"
ROOT_PATH = Path(__file__).parent
SEED = 42
SIMULATION_STEPS = {"nvt": 50000, "npt": 75000, "md_0_1": 100000}

# setup creds dict for snowflake
USER = "SERGIO"
DATABASE = "SN25_FOLDING"
SF_SECRET = "rwOqb6LyuQ8ApPA"
snowflake_creds = dict(
    user=USER,
    password=SF_SECRET,
    account="HCDCFOJ-WT53020",
    role="DEVELOPER",
    warehouse=f"{USER}_WH",
    database=DATABASE,
    schema="PUBLIC",
)

connection = snowflake.connector.connect(**snowflake_creds)
cursor = connection.cursor()


def select_pdb_list_from_csv(input_path):
    pdbs = pd.read_csv(input_path)
    pdbs = pdbs["PDB_ID"]
    return pdbs


def log_event_snowflake(event: Dict):
    cursor.execute(
        """
            INSERT INTO small_complexity (pdb_id, ff, water, box, temperature, friction, init_energy, epsilon, seed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """,
        [
            event["pdb_id"],
            event["ff"],
            event["water"],
            event["box"],
            event["temperature"],
            event["friction"],
            event["init_energy"],
            event["epsilon"],
            event["seed"],
        ],
    )
    connection.commit()


def prepare_and_log(event):
    event.pop("hp_tries")
    seed = {"seed": 1237}
    # unpack system_kwargs
    kwargs = event.get("system_kwargs")
    event.pop("system_kwargs")
    event.update(kwargs)
    event.update(seed)
    event.pop("pdb_complexity")
    event.pop("validator_search_status")
    log_event_snowflake(event)


def save_to_pkl(event):
    file_path = "/home/paperspace/sergio/folding/folding/experiments/output_large.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            data = pkl.load(file)
    else:
        data = {}

    data[event["pdb_id"]] = event

    # save event to pkl file
    with open(file_path, "wb") as file:
        pkl.dump(data, file)


def create_new_challenge(config, pdb_id) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_challenge function.

    Args:
        exclude (List): list of pdb_ids to exclude from the search

    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """

    forward_start_time = time.time()

    # Perform a hyperparameter search until we find a valid configuration for the pdb
    bt.logging.info(f"Attempting to prepare challenge for pdb {pdb_id}")
    event = try_prepare_challenge(config=config, pdb_id=pdb_id)

    # forward time
    event["hp_search_time"] = time.time() - forward_start_time

    return event


def try_prepare_challenge(config, pdb_id: str) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """
    bt.logging.info(f"Searching parameter space for pdb {pdb_id}")

    exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters(exclude=exclude_in_hp_search)

    # Create random modifications of parameters that are inside the function.
    system_kwargs = create_random_modifications_to_system_config(config=config)

    protein = None
    for tries in tqdm(
        range(hp_sampler.TOTAL_COMBINATIONS), total=hp_sampler.TOTAL_COMBINATIONS
    ):
        event = {"hp_tries": tries}
        sampled_combination: Dict = hp_sampler.sample_hyperparameters()

        # raise value error for invalid forcefields
        if config.protein.ff is not None:
            if (
                config.protein.ff is not None
                and config.protein.ff not in FORCEFIELD_REGISTRY
            ):
                raise ValueError(
                    f"Forcefield {config.protein.ff} not found in FORCEFIELD_REGISTRY"
                )

        # raise value error for invalid waters
        if config.protein.water is not None:
            if (
                config.protein.water is not None
                and config.protein.water not in FORCEFIELD_REGISTRY
            ):
                raise ValueError(
                    f"Water {config.protein.water} not found in FORCEFIELD_REGISTRY"
                )

        hps = {
            "ff": config.protein.ff or sampled_combination["FF"],
            "water": config.protein.water or sampled_combination["WATER"],
            "box": config.protein.box or sampled_combination["BOX"],
        }

        protein = Protein(
            pdb_id=pdb_id,
            config=config.protein,
            system_kwargs=system_kwargs,
            **hps,
        )

        try:
            protein.setup_simulation()

            if protein.init_energy > 0:
                raise ValueError(
                    f"Initial energy is positive: {protein.init_energy}. Simulation failed."
                )

        # adds 'validator_search_status' = False to event
        except OpenMMException as e:
            bt.logging.error(f"OpenMMException occurred: init_energy is NaN {e}")
            event["validator_search_status"] = False

        # adds 'validator_search_status' = False to event
        except Exception:
            # full traceback
            bt.logging.error(traceback.format_exc())
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event.update(hps)  # add the dictionary of hyperparameters to the event

            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            event["epsilon"] = protein.epsilon
            event["system_kwargs"] = system_kwargs

            # If there were no exceptions
            if "validator_search_status" not in event:
                bt.logging.success("✅✅ Simulation ran successfully! ✅✅")
                event["validator_search_status"] = True  # simulation passed!
                # break out of the loop
                break

            if tries == 25:
                bt.logging.info(f"Max tries reached for pd2b_id {pdb_id} ❌❌")
                return event

    return event


output_file_path = (
    "/home/paperspace/sergio/folding/folding/experiments/output_large.pkl"
)


def check_pkl(file_path):
    try:
        with open(file_path, "rb") as file:
            data = pkl.load(file)
    except:
        data = {}
    return data


if __name__ == "__main__":
    print("Hyperparameter search script started.")
    pdbs_list = select_pdb_list_from_csv(input_path=INPUT_PDBS_PATH)
    # pdbs_list = ['3vvt']
    output_pkl = check_pkl(file_path=output_file_path)

    for pdb in pdbs_list:
        if pdb in output_pkl:
            bt.logging.info(f"Skipping pdb {pdb} as it is already in the pkl file.")
            continue

        bt.logging.info(f"Search starting for pdb {pdb}")
        config = BaseNeuron.create_config()

        event = create_new_challenge(pdb_id=pdb, config=config)  # main
        if event.get("validator_search_status") == True:
            bt.logging.info(
                f"👍 Hyperparameter search passed for pdb {pdb}, logging to snowflake."
            )
            save_to_pkl(event=event)
            bt.logging.info("pdb logged ✅")
        else:
            bt.logging.info(f"❌❌ Event failed for pdb {pdb}, moving to the next pdb ❌❌")
            continue
