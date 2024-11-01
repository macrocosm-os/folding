### This script is used to run a sensitivity search for a protein folding simulation.
### Used for benchmarking.

import os
import time
import wandb
import random
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import bittensor as bt

import pandas as pd
from box import Box  # install using pip install box
import copy

import plotly.express as px
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from folding.base.simulation import OpenMMSimulation
from folding.validators.hyperparameters import HyperParameters
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.validators.protein import Protein
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter
from folding.utils.config import config
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

ROOT_PATH = Path(__file__).parent
SEED = 42
SIMULATION_STEPS = {"nvt": 1_000_000}


def log_event(event = None, protein_vis = None):
    """Log the event to the console and to the wandb logger."""
    # bt.logging.info(f"Event: {event}")
    if event:
        wandb.log(event)
    if protein_vis:
        wandb.log({"protein_vis": wandb.Molecule(protein_vis)})


def create_wandb_run(pdb_id: str, project: str = "folding-openmm", entity: str = "macrocosmos", tags: List = []):
    wandb.init(project=project, entity=entity, tags=tags, name=pdb_id)


def configure_commands(
    state: str,
    seed: int,
    system_config: SimulationConfig,
    pdb_obj: app.PDBFile,
    output_dir: str,
    pdb_id: str,
    angle: int,
    CHECKPOINT_INTERVAL: int = 10000,
    STATE_DATA_REPORTER_INTERVAL: int = 10,
    EXIT_REPORTER_INTERVAL: int = 10,
) -> Dict[str, List[str]]:
    simulation, _ = OpenMMSimulation().create_simulation(
        pdb=pdb_obj,
        system_config=system_config.get_config(),
        seed=seed,
        state=state,
    )
    simulation.reporters.append(
        LastTwoCheckpointsReporter(
            file_prefix=f"{output_dir}/{state}",
            reportInterval=CHECKPOINT_INTERVAL,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            file=f"{output_dir}/{state}_{seed}.log",
            reportInterval=STATE_DATA_REPORTER_INTERVAL,
            step=True,
            potentialEnergy=True,
        )
    )
    simulation.reporters.append(
        ExitFileReporter(
            filename=f"{output_dir}/{state}",
            reportInterval=EXIT_REPORTER_INTERVAL,
            file_prefix=state,
        )
    )
    simulation.reporters.append(
        app.PDBReporter(
            file=f"{output_dir}/folded_{angle}_{seed}.pdb",
            reportInterval=CHECKPOINT_INTERVAL,
        )
    ) 

    return simulation

def create_random_modifications_to_system_config() -> Dict:
    """create modifications of the desired parameters.

    Looks at the base bittensor config and parses parameters that we have deemed to be
    valid for random sampling. This is to increase the problem space for miners.
    """
    sampler = lambda min_val, max_val: round(np.random.uniform(min_val, max_val), 2)

    system_kwargs = {"temperature": sampler(200, 400), "friction": sampler(0.9, 1.1)}


    return system_kwargs

def create_new_challenge(pdb_id: str, angles=(180,180)) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_challenge function.
    Args:
        exclude (List): list of pdb_ids to exclude from the search
    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """

    forward_start_time = time.time()

    # Perform a hyperparameter search until we find a valid configuration for the pdb
    # bt.logging.warning(f"Attempting to prepare challenge for pdb {pdb_id}")
    protein, event = try_prepare_challenge(pdb_id=pdb_id, angles=angles)

    if event.get("validator_search_status"):
        bt.logging.success(f"✅✅ Successfully created challenge for pdb_id {pdb_id} ✅✅")
    else:
        # forward time if validator step fails
        event["hp_search_time"] = time.time() - forward_start_time
        bt.logging.error(
            f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
        )

    return protein, event


def try_prepare_challenge(pdb_id: str, angles=(180,180)) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """

    # exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters()

    bt.logging.info(f"Searching parameter space for pdb {pdb_id}")
    
    system_kwargs = create_random_modifications_to_system_config()

    for tries in tqdm(
        range(hp_sampler.TOTAL_COMBINATIONS), total=hp_sampler.TOTAL_COMBINATIONS
    ):
        hp_sampler_time = time.time()

        event = {"hp_tries": tries}
        sampled_combination: Dict = hp_sampler.sample_hyperparameters()

        hps = {
            "ff": sampled_combination["FF"],
            "water": sampled_combination["WATER"],
            "box": sampled_combination["BOX"],
        }

        config = Box({"force_use_pdb": False, "input_source": "rcsb"})

        protein = Protein(pdb_id=pdb_id, config=config, system_kwargs=system_kwargs, **hps)

        try:
            protein.setup_simulation(angles=angles)

        except Exception as e:
            bt.logging.error(f"Error occurred for pdb_id {pdb_id}: {e}")
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event.update(hps)  # add the dictionary of hyperparameters to the event
            event["hp_sample_time"] = time.time() - hp_sampler_time
            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            event["angle"] = angles[0]
            event['system_kwargs'] = system_kwargs


            if "validator_search_status" not in event:
                bt.logging.warning("✅✅ Simulation ran successfully! ✅✅")
                event["validator_search_status"] = True  # simulation passed!
                # break out of the loop if the simulation was successful
                break
            if tries == 3:
                bt.logging.error(f"Max tries reached for pdb_id {pdb_id} :x::x:")
                break

    return protein, event


def sample_pdb(exclude: List = [], pdb_id: str = None):
    return pdb_id or select_random_pdb_id(PDB_IDS, exclude=exclude)


def extact_energies(state: str, data_directory: str, num_experiments: int):
    check_log_file = pd.read_csv(os.path.join(data_directory, f"{state}_{num_experiments}.log"))

    return check_log_file["Potential Energy (kJ/mole)"].values


def cpt_file_mapper(output_dir: str, state: str, angles: tuple):
    if state == "nvt":
        return f"{output_dir}/em.cpt"

    if "npt" in state:
        state = "nvt" + state.split("npt")[-1]

    if "md" in state:
        state = "npt" + state.split("md_0_1")[-1]

    return f"{output_dir}/{state}.cpt"


if __name__ == "__main__":

    angles = [(0,0), (30,30), (60,60), (90,90), (120,120), (150,150)]

    pdbs_to_exclude = []
    pdb_ids = ['5kxs', '6qdy']
    for pdb_id in pdb_ids:
        create_wandb_run(project="folding-openmm", entity="macrocosmos", tags=["unfolding"], pdb_id=pdb_id)

        for angle in angles:
            num_experiments = 3
            while num_experiments > 0:
                

                try:
                    protein, event = create_new_challenge(pdb_id=pdb_id, angles=angle)
                except Exception as e:
                    bt.logging.error(f"Error occurred for pdb_id {pdb_id}: {e}")
                    continue

                event['num_experiment'] = num_experiments
                pdb_obj: app.PDBFile = protein.load_pdb_file(protein.pdb_location)
                output_dir = protein.validator_directory

                system_config = copy.deepcopy(protein.system_config)

                for state, steps_to_run in SIMULATION_STEPS.items():
                    # Creates the simulation object needed for the stage.
                    partitions = steps_to_run // 100_000
                    

                    temp_state = state + f"_angle_{angle}"

                    simulation = configure_commands(
                        state=temp_state,
                        seed=num_experiments,
                        system_config=system_config,
                        pdb_obj=pdb_obj,
                        output_dir=protein.validator_directory,
                        pdb_id=pdb_id,
                        angle=angle[0]
                    )

                    bt.logging.info(
                        f"Running {state} for {steps_to_run} steps for pdb {pdb_id}"
                    )
                    if state == "nvt":
                        mapper_state = state
                    else:
                        mapper_state = temp_state


                    simulation.loadCheckpoint(cpt_file_mapper(output_dir, mapper_state, angle))
                    
                    start_time = time.time()
                    for _ in range(partitions):
                        simulation.step(100_000)
                        event['energy'] = simulation.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoules_per_mole
                        event['time'] = time.time() - start_time
                        event['step'] = simulation.currentStep
                        log_event(event, protein_vis=f"{protein.validator_directory}/folded_{angle[0]}_{num_experiments}.pdb")
                    simulation_time = time.time() - start_time
                    event[f"{state}_time"] = simulation_time

                    energy_array = extact_energies(
                        state=temp_state, data_directory=protein.validator_directory, num_experiments=num_experiments
                    )
                    # event[f"{state}_energies_angle_{angle}"] = energy_array.tolist()

                    fig = px.scatter(
                        energy_array,
                        title=f"Energy array for {pdb_id} for state {state} for angle {angle}",
                        labels={"index": "Step", "value": "energy"},
                        height=600,
                        width=1400,
                    )
                    fig.write_image(os.path.join(output_dir, f"{mapper_state}_energy.png"))

                log_event(event)
                num_experiments = num_experiments - 1
