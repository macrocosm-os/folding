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


def init_wandb(self, pdb_id: str, reinit=True):
    """Starts a new wandb run."""
    tags = [
        self.wallet.hotkey.ss58_address,
        folding.__version__,
        str(folding.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
    ]

    if self.config.mock:
        tags.append("mock")
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    if pdb_id not in self.wandb_ids.keys():
        run = wandb.init(
            anonymous="allow",
            name=pdb_id,
            reinit=reinit,
            project=self.config.wandb.project_name,
            entity=self.config.wandb.entity,
            config=wandb_config,
            mode="offline" if self.config.wandb.offline else "online",
            dir=self.config.neuron.full_path,
            tags=tags,
            notes=self.config.wandb.notes,
            resume="allow",
        )
        self.wandb_ids[pdb_id] = run.id
    else:
        run = wandb.init(
            anonymous="allow",
            name=pdb_id,
            id=self.wandb_ids[pdb_id],
            reinit=reinit,
            project=self.config.wandb.project_name,
            entity=self.config.wandb.entity,
            config=wandb_config,
            mode="offline" if self.config.wandb.offline else "online",
            dir=self.config.neuron.full_path,
            tags=tags,
            notes=self.config.wandb.notes,
            resume="allow",
        )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )
    return run


def log_event(self, event):
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    if self.config.wandb.off:
        return
    pdb_id = event["pdb_id"]
    if not getattr(self, "wandb_ids", None):
        self.wandb_ids = {}

    run = init_wandb(self, pdb_id=pdb_id)

    # Log the event to wandb.
    run.log(event)
    run.finish()
