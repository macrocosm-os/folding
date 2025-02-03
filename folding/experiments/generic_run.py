### This script is used to run a sensitivity search for a protein folding simulation.
### Used for benchmarking.

import os
import time
import copy
import wandb
from tqdm import tqdm
from folding.utils.logger import logger
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from box import Box  # install using pip install box

import plotly.express as px
import openmm.app as app

from folding.validators.protein import Protein
from folding.base.simulation import OpenMMSimulation
from folding.validators.hyperparameters import HyperParameters
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.reporters import ExitFileReporter, LastTwoCheckpointsReporter

SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_PATH = Path(__file__).parent
# SIMULATION_STEPS = {"nvt": 1000, "npt": 1000, "md_0_1": 1000}
SIMULATION_STEPS = {"nvt": 50000, "npt": 75000, "md_0_1": 100000}

PDBS_TO_TEST = ["5v4y", "1o86", "2c6f", "1rtl", "3eYd"]
# PDBS_TO_TEST = ["1o86"]

# references: {
#     5v4y = P11485
#     1o86 = P12821 
#     2C6F = P12821
#     1RTL = P26664
#     3EYD = P26664
# }


def log_event(event: Dict):
    """Log the event to the console and to the wandb logger."""
    # logger.info(f"Event: {event}")
    wandb.log(event)


def create_wandb_run(project: str = "folding-openmm", entity: str = "macrocosmos"):
    wandb.init(project=project, entity=entity)


def configure_commands(
    state: str,
    seed: int,
    system_config: SimulationConfig,
    pdb_obj: app.PDBFile,
    output_dir: str,
    CHECKPOINT_INTERVAL: int = 1000,
    STATE_DATA_REPORTER_INTERVAL: int = 10,
    EXIT_REPORTER_INTERVAL: int = 10,
) -> Dict[str, List[str]]:
    simulation, _ = OpenMMSimulation().create_simulation(
        pdb=pdb_obj,
        system_config=system_config.get_config(),
        seed=seed,
    )
    simulation.reporters.append(
        LastTwoCheckpointsReporter(
            file_prefix=f"{output_dir}/{state}",
            reportInterval=CHECKPOINT_INTERVAL,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            file=f"{output_dir}/{state}.log",
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

    return simulation


async def create_new_challenge(pdb_id: str) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_challenge function.
    Args:
        exclude (List): list of pdb_ids to exclude from the search
    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """

    forward_start_time = time.time()

    # Perform a hyperparameter search until we find a valid configuration for the pdb
    # logger.warning(f"Attempting to prepare challenge for pdb {pdb_id}")
    protein, event = await try_prepare_challenge(pdb_id=pdb_id)

    if event.get("validator_search_status"):
        logger.success(f"✅✅ Successfully created challenge for pdb_id {pdb_id} ✅✅")
    else:
        # forward time if validator step fails
        event["hp_search_time"] = time.time() - forward_start_time
        logger.error(
            f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
        )

    return protein, event


async def try_prepare_challenge(pdb_id: str) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """

    # exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters()

    logger.info(f"Searching parameter space for pdb {pdb_id}")

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

        system_kwargs = create_random_modifications_to_system_config()

        config = Box({"force_use_pdb": True, "input_source" : "rcsb"})
        protein = Protein(pdb_id=pdb_id, config=config, system_kwargs=system_kwargs, **hps,)

        try:
            await protein.setup_simulation()

        except Exception as e:
            logger.error(f"Error occurred for pdb_id {pdb_id}: {e}")
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event.update(hps)  # add the dictionary of hyperparameters to the event
            event["hp_sample_time"] = time.time() - hp_sampler_time
            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            # event["epsilon"] = protein.epsilon

            if "validator_search_status" not in event:
                logger.warning("✅✅ Simulation ran successfully! ✅✅")
                event["validator_search_status"] = True  # simulation passed!
                # break out of the loop if the simulation was successful
                break
            if tries == 10:
                logger.error(f"Max tries reached for pdb_id {pdb_id} :x::x:")
                break

    return protein, event


def create_random_modifications_to_system_config() -> Dict:
    """create modifications of the desired parameters.

    Looks at the base bittensor config and parses parameters that we have deemed to be
    valid for random sampling. This is to increase the problem space for miners.
    """
    # sampler = lambda min_val, max_val: round(np.random.uniform(min_val, max_val), 2)
    # system_kwargs = {"temperature": sampler(200, 400), "friction": sampler(0.9, 1.1)}

    system_kwargs = {"temperature" : 300, "friction" : 1}
    return system_kwargs


def extact_energies(state: str, data_directory: str):
    check_log_file = pd.read_csv(os.path.join(data_directory, f"{state}.log"))

    return check_log_file["Potential Energy (kJ/mole)"].values


def cpt_file_mapper(output_dir: str, state: str):
    if state == "nvt":
        return f"{output_dir}/em.cpt"

    if "npt" in state:
        state = "nvt" + state.split("npt")[-1]

    if "md" in state:
        state = "npt" + state.split("md_0_1")[-1]

    return f"{output_dir}/{state}.cpt"


def run_simulation(
    protein, pdb_obj, output_dir: str
):
    system_config = copy.deepcopy(protein.system_config)

    for seed in [1, 42, 1337]:
        system_config.seed = seed

        for state, steps_to_run in SIMULATION_STEPS.items():
            # Creates the simulation object needed for the stage.

            temp_state = state + f"_temp_{system_config.temperature}_seed_{seed}"

            simulation = configure_commands(
                state=temp_state,
                seed=protein.system_config.seed,
                system_config=system_config,
                pdb_obj=pdb_obj,
                output_dir=protein.validator_directory,
            )

            logger.info(f"Running {state} for {steps_to_run} steps for pdb {protein.pdb_id}")

            if state == "nvt":
                mapper_state = state
            else:
                mapper_state = temp_state

            simulation.loadCheckpoint(cpt_file_mapper(output_dir, mapper_state))

            start_time = time.time()
            simulation.step(steps_to_run)
            simulation_time = time.time() - start_time

            energy_array = extact_energies(
                state=temp_state, data_directory=protein.validator_directory
            )

            fig = px.scatter(
                energy_array,
                title=f"Energy array for {protein.pdb_id} for state {state} for temperature {system_config.temperature}",
                labels={"index": "Step", "value": "energy"},
                height=600,
                width=1400,
            )
            fig.write_image(os.path.join(output_dir, f"{mapper_state}_energy.png"))


import asyncio 
async def process_pdb(pdb_id):
    """Process a single PDB ID asynchronously."""
    try:
        protein, event = await create_new_challenge(pdb_id=pdb_id)
    except Exception as e:
        logger.error(f"Error occurred for pdb_id {pdb_id}: {e}")
        return

    pdb_obj: app.PDBFile = protein.load_pdb_file(protein.pdb_location)
    output_dir = protein.validator_directory

    # If run_simulation is not async, use asyncio.to_thread
    await asyncio.to_thread(
        run_simulation,
        protein=protein,
        pdb_obj=pdb_obj,
        output_dir=output_dir,
    )

async def stop_pm2():
    """Stops all PM2 processes using a shell command."""
    process = await asyncio.create_subprocess_shell(
        "pm2 stop all",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if stdout:
        print(f"[PM2 Stop Output]:\n{stdout.decode()}")
    if stderr:
        print(f"[PM2 Stop Error]:\n{stderr.decode()}")

async def main():
    """Main async entry point."""
    for pdb_id in PDBS_TO_TEST:
        await process_pdb(pdb_id)  # Process one at a time sequentially
    
    # Run PM2 stop all after processing all PDBs
    await stop_pm2()

if __name__ == "__main__":
    asyncio.run(main())

