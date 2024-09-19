import copy
import wandb
from typing import List
from loguru import logger
from dataclasses import asdict, dataclass

import folding
import bittensor as bt


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
    project = self.config.wandb.project_name
    id = None if pdb_id not in self.wandb_ids.keys() else self.wandb_ids[pdb_id]

    if pdb_id in self.wandb_ids.keys():
        id = self.wandb_ids[pdb_id]["wandb_id"]
        run_status = self.wandb_ids[pdb_id]["status"]
    else:
        id = None
        run_status = "active"

    tags = [
        self.wallet.hotkey.ss58_address,
        folding.__version__,
        str(folding.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
        run_status,
    ]

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

    if run_status == "active":
        self.add_wandb_id(pdb_id, run.id)

    if id is None:
        bt.logging.success(
            prefix="Started a new wandb run",
            sufix=f"<blue> {pdb_id} </blue>",
        )
    else:
        bt.logging.success(
            prefix="updated a wandb run",
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
        bt.logging.warning("Failed to log protein visualization")


def log_event(self, event, failed=False, pdb_location: str = None):
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    if self.config.wandb.off:
        return
    pdb_id = event["pdb_id"]

    run = init_wandb(self, pdb_id=pdb_id, failed=failed)

    # Log the event to wandb.
    run.log(event)

    if pdb_location is not None:
        log_protein(run, pdb_id_path=pdb_location)

    run.finish()
