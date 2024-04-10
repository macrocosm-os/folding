import time
import argparse
from typing import List, Dict

import wandb
import bittensor as bt

from folding.validators.protein import Protein
from folding.validators.hyperparameters import HyperParameters


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=self.config.wandb.project_name,
        entity=self.config.wandb.entity,
        mode="offline" if self.config.wandb.offline else "online",
        notes=self.config.wandb.notes,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )


def parse_config(config) -> List[str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """
    ff = config.protein.ff
    water = config.protein.water
    box = config.protein.box
    exclude_in_hp_search = []

    if ff is not None:
        exclude_in_hp_search.append("ff")
    if water is not None:
        exclude_in_hp_search.append("water")
    if box is not None:
        exclude_in_hp_search.append("box")

    return exclude_in_hp_search


def forward(config):
    for ii in range(100):
        forward_start_time = time.time()
        exclude_in_hp_search = parse_config(config)

        hp_sampler = HyperParameters(
            pdb_id=config.protein.pdb_id,
            exclude=exclude_in_hp_search,
        )

        try:
            sampled_combination: Dict[str, Dict] = hp_sampler.sample_hyperparameters()
            hp: Dict = sampled_combination[config.protein.pdb_id]

            protein = Protein(
                pdb_id=config.protein.pdb_id,
                ff=config.protein.ff if config.protein.ff is not None else hp["ff"],
                water=config.protein.water
                if config.protein.water is not None
                else hp["water"],
                box=config.protein.box if config.protein.box is not None else hp["box"],
                config=config.protein,
            )

            bt.logging.info(f"Attempting to generate challenge: {protein}")
            protein.forward()

        except Exception as E:
            bt.logging.error(f"Error running hyperparameters {hp}")

        finally:
            event = {}

            event["forward_time"] = time.time() - forward_start_time
            event["pdb_id"] = config.protein.pdb_id
            event.update(hp)  # add the protein hyperparameters to the event

            wandb.log(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a Bittensor synapse with given parameters."
    )

    parser.add_argument(
        "--pdb_id",
        type=str,
        default="5oxe",
        help="protein that you want to fold",
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

    parser.add_argument(
        "--protein.pdb_id",
        type=str,
        help="PDB ID for protein folding.",  # defaults to None
        default=None,
    )

    parser.add_argument(
        "--protein.ff",
        type=str,
        help="Force field for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--protein.water",
        type=str,
        help="Water used for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--protein.box",
        type=str,
        help="Box type for protein folding.",
        default=None,
    )

    parser.add_argument(
        "--protein.max_steps",
        type=int,
        help="Maximum number of steps for protein folding.",
        default=10000,
    )

    parser.add_argument(
        "--protein.suppress_cmd_output",
        action="store_true",
        help="If set, we suppress the text output of terminal commands to reduce terminal clutter.",
        default=True,
    )

    parser.add_argument(
        "--neuron.mock",
        action="store_true",
        help="Dry run.",
        default=False,
    )

    main_process(args=args)
