import copy
import wandb
from typing import List
from dataclasses import asdict, dataclass
import os

import folding
import bittensor as bt
from folding.utils.logger import logger


@dataclass
class Log:
    validator_model_id: str
    challenge: str
    challenge_prompt: str
    reference: str
    miners_ids: List[str]
    responses: List[str]
    miners_time: List[float]
    challenge_time: float
    reference_time: float
    rewards: List[float]
    task: dict
    # extra_info: dict


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return (
        not self.config.wandb.off
        and self.step
        and self.step % self.config.wandb.run_step_length == 0
    )


def init_wandb(self, pdb_id: str, reinit=True, failed=False):
    """Starts a new wandb run."""

    tags = [
        self.wallet.hotkey.ss58_address,
        folding.__version__,
        str(folding.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
    ]
    project = self.config.wandb.project_name
    if failed:
        tags.append("failed")

    if self.config.mock:
        tags.append("mock")
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")
    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    id = None if pdb_id not in self.wandb_ids.keys() else self.wandb_ids[pdb_id]

    run = wandb.init(
        anonymous="allow",
        name=pdb_id,
        reinit=reinit,
        project=project,
        id=id,
        entity=self.config.wandb.entity,
        config=wandb_config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
        resume="allow",
    )

    self.add_wandb_id(pdb_id, run.id)

    if id is None:
        logger.success(
            "Started a new wandb run",
            sufix=f"<blue> {pdb_id} </blue>",
        )
    else:
        logger.success(
            "updated a wandb run",
            sufix=f"<blue> {pdb_id} </blue>",
        )

    return run


def log_protein(run, pdb_id_path: str):
    """Logs the protein visualization to wandb.
    pdb_id_path: str: path to the pdb file on disk.
    """
    try:
        run.log({"protein_vis": wandb.Molecule(pdb_id_path)})
    except:
        logger.warning("Failed to log protein visualization")


def log_folded_protein(run, pdb_id_path: str):
    """Logs the folded protein visualization to wandb.
    pdb_id_path: str: path to the pdb file on disk.
    """
    try:
        run.log({"folded_protein_vis": wandb.Molecule(pdb_id_path)})
    except:
        logger.warning("Failed to log folded protein visualization")


def log_event(
    self,
    event,
    failed=False,
    pdb_location: str = None,
    folded_protein_location: str = None,
):
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    if self.config.wandb.off:
        return
    pdb_id = event["pdb_id"]

    run = init_wandb(self, pdb_id=pdb_id, failed=failed)

    # Log the event to wandb.
    run.log(event)
    wandb.save(os.path.join(self.config.neuron.full_path, f"events.log"))

    if pdb_location is not None:
        log_protein(run, pdb_id_path=pdb_location)
    if folded_protein_location is not None:
        log_folded_protein(run, pdb_id_path=folded_protein_location)
        wandb.save(folded_protein_location)

    run.finish()

    if (event["validator_search_status"] == False) or (
        "active" in event and event["active"] == False
    ):
        self.remove_wandb_id(pdb_id)
