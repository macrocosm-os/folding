import time
import argparse
from typing import List, Dict

import wandb
import bittensor as bt

from folding.validators.protein import Protein
from folding.validators.hyperparameters import HyperParameters


def init_wandb(config, reinit=False):
    """Starts a new wandb run."""

    run = wandb.init(
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
        exclude_in_hp_search.append("ff")
    if water is not None:
        exclude_in_hp_search.append("water")
    if box is not None:
        exclude_in_hp_search.append("box")

    return exclude_in_hp_search


def forward(config):
    init_wandb(config=config)

    for ii in range(2):
        print(config)
        bt.logging.info(f"Starting iteration {ii}")
        forward_start_time = time.time()
        exclude_in_hp_search = parse_config(config)

        hp_sampler = HyperParameters(
            pdb_id=config.pdb_id,
            exclude=exclude_in_hp_search,
        )

        try:
            sampled_combination: Dict[str, Dict] = hp_sampler.sample_hyperparameters()
            hp: Dict = sampled_combination[config.pdb_id]

            bt.logging.info(f"Selected hyperparameters: {hp}")

            protein = Protein(
                pdb_id=config.pdb_id,
                ff=config.ff if config.ff is not None else hp["FF"],
                water=config.water if config.water is not None else hp["WATER"],
                box=config.box if config.box is not None else hp["BOX"],
                config=config,
            )

            bt.logging.info(f"Attempting to generate challenge: {protein}")
            protein.forward()

        except Exception as E:
            bt.logging.error(f"Error running hyperparameters {hp}")

        finally:
            event = {}

            event["forward_time"] = time.time() - forward_start_time
            event["pdb_id"] = config.pdb_id
            event.update(hp)  # add the protein hyperparameters to the event

            wandb.log(event)

            bt.logging.success(f"Finished iteration {ii}")
        bt.logging.success(f"Finished all iterations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
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
        default=None,
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
        default=None,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of steps for protein folding.",
        default=10000,
    )

    parser.add_argument(
        "--suppress_cmd_output",
        action="store_true",
        help="If set, we suppress the text output of terminal commands to reduce terminal clutter.",
        default=True,
    )

    config = parser.parse_args()
    forward(config=config)
