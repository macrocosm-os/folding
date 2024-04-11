import os
import pickle
import time
import argparse
from typing import List, Dict

import wandb
import bittensor as bt

from folding.validators.protein import Protein
from folding.utils.ops import select_random_pdb_id
from folding.validators.hyperparameters import HyperParameters

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDB_PATH = os.path.join(ROOT_DIR, "folding/pdb_ids.pkl")
if not os.path.exists(PDB_PATH):
    raise ValueError(
        f"Required Pdb file {PDB_PATH!r} was not found. Run `python scripts/gather_pdbs.py` first."
    )

with open(PDB_PATH, "rb") as f:
    PDB_IDS = pickle.load(f)


def init_wandb(config, reinit=False):
    """Starts a new wandb run."""

    wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=config.project_name,
        entity=config.entity,
        mode="offline" if config.offline else "online",
        notes=config.notes,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
    )


def parse_config(config) -> List[str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """
    ff = config.ff
    water = config.water
    box = config.box
    exclude_in_hp_search = []

    if ff is not None:
        exclude_in_hp_search.append("FF")
    if water is not None:
        exclude_in_hp_search.append("WATER")
    if box is not None:
        exclude_in_hp_search.append("BOX")

    return exclude_in_hp_search


def forward(config):
    init_wandb(config=config)

    num_pdbs_attempted = 0
    while True:
        num_pdbs_attempted += 1

        # Select a random pdb id to be iterating over.

        bt.logging.info(f"CONFIG: {config}")

        if config.pdb_id is None:
            pdb_id = select_random_pdb_id(PDB_IDS=PDB_IDS)
        else:
            pdb_id = config.pdb_id

        bt.logging.info(f"Starting pdb iteration {num_pdbs_attempted}")
        forward_start_time = time.time()
        exclude_in_hp_search = parse_config(config)

        hp_sampler = HyperParameters(exclude=exclude_in_hp_search)

        for ii in range(hp_sampler.TOTAL_COMBINATIONS):
            event = {}
            try:
                sampled_combination: Dict = hp_sampler.sample_hyperparameters()
                bt.logging.info(
                    f"pdb_id: {pdb_id}, Selected hyperparameters: {sampled_combination}, iteration {ii}"
                )

                protein = Protein(
                    pdb_id=pdb_id,
                    ff=config.ff
                    if config.ff is not None
                    else sampled_combination["FF"],
                    water=config.water
                    if config.water is not None
                    else sampled_combination["WATER"],
                    box=config.box
                    if config.box is not None
                    else sampled_combination["BOX"],
                    config=config,
                )

                hps = {
                    "FF": protein.ff,
                    "WATER": protein.water,
                    "BOX": protein.box,
                    "BOX_DISTANCE": sampled_combination["BOX_DISTANCE"],
                }

                bt.logging.info(f"Attempting to generate challenge: {protein}")
                protein.forward()

            except Exception as E:
                bt.logging.error(
                    f"❌❌ Error running hyperparameters {sampled_combination} for pdb_id {pdb_id} ❌❌"
                )
                bt.logging.warning(E)
                event["status"] = False

            finally:
                event["forward_time"] = time.time() - forward_start_time
                event["pdb_id"] = pdb_id
                event.update(hps)  # add the protein hyperparameters to the event

                if "status" not in event:
                    event["status"] = True  # simulation passed!
                    bt.logging.info("✅✅ Simulation ran successfully! ✅✅")

                wandb.log(event)

                # if event["status"] is True:
                #     break  # break out of the for loop.

        if (num_pdbs_attempted == config.num_pdbs) or (config.pdb_id is not None):
            bt.logging.success(f"Finished all possible pdbs.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
    )

    parser.add_argument(
        "--num_pdbs",
        type=int,
        help="Number of PDBs to test.",
        default=100,
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="folding-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="TEST_RUNS",
    )

    parser.add_argument(
        "--pdb_id",
        type=str,
        help="PDB ID for protein folding.",  # defaults to None
        default="5oxe",
    )

    parser.add_argument(
        "--ff",
        type=str,
        help="Force field for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--water",
        type=str,
        help="Water used for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--box",
        type=str,
        help="Box type for protein folding.",
        default=None,  #'dodecahedron',
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of steps for protein folding.",
        default=100,
    )

    parser.add_argument(
        "--suppress_cmd_output",
        action="store_true",
        help="If set, we suppress the text output of terminal commands to reduce terminal clutter.",
        default=True,
    )

    config = parser.parse_args()
    forward(config=config)
