import subprocess
import tqdm
from typing import List, Dict
from collections import defaultdict
import bittensor as bt
import copy
import wandb
from loguru import logger
from dataclasses import asdict, dataclass

import folding


"""
We need something thet gets returned (result dict) 
We don't want it to grow each time its called
We want it to be independent of gromacs
We dont need to be worried about pdb_id 
This needs to log to the event dict 
"""


def output_dict():
    return defaultdict(str)


class CommandFailedException(Exception):
    pass


class RunAndLog:
    def __init__(self):
        self.command_dict = defaultdict(output_dict)

    def run_commands(
        self,
        commands: List[str],
        suppress_cmd_output: bool = True,
        verbose: bool = False,
    ) -> Dict[str, str]:
        for cmd in tqdm.tqdm(commands):  # set tuple
            bt.logging.debug(f"Running command {cmd}")

            try:
                # result = run the command (cmd), check that it succedded, executed it through the shell, captures its output and error messages.
                result = subprocess.run(
                    cmd,
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if not suppress_cmd_output:
                    bt.logging.info(result.stdout.decode())

                    # no need to record and log successful command executions

            except subprocess.CalledProcessError as e:
                # this will always print to the terminal regardless if verbose.
                bt.logging.error(f"❌ Failed to run command ❌: {cmd}")

                if verbose:
                    bt.logging.error(f"Output: {e.stdout.decode()}")
                    bt.logging.error(f"Error: {e.stderr.decode()}")

                # update command_dict
                self.command_dict[str(cmd = ' '.join(cmd.split(' ', 2)[:2]))]["error"] = e.stderr.decode()

                # call log_event with the command and error message
                self.log_event(event=self.command_dict[str(cmd)]["error"])

                command_dict = (
                    self.command_dict
                )  # we have the option of using the command dict before exception is raised

                # Raise an exception during a failure event
                raise subprocess.CalledProcessError(
                    returncode=f"Failed to run command: {cmd}"
                )

    def should_reinit_wandb(self):
        # Check if wandb run needs to be rolled over.
        return (
            not self.config.wandb.off
            and self.step
            and self.step % self.config.wandb.run_step_length == 0
        )

    def init_wandb(self, reinit=False):
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

        self.wandb = wandb.init(
            anonymous="allow",
            reinit=reinit,
            project=self.config.wandb.project_name,
            entity=self.config.wandb.entity,
            config=wandb_config,
            mode="offline" if self.config.wandb.offline else "online",
            dir=self.config.neuron.full_path,
            tags=tags,
            notes=self.config.wandb.notes,
        )
        bt.logging.success(
            prefix="Started a new wandb run",
            sufix=f"<blue> {self.wandb.name} </blue>",
        )

    def log_event(self, event):
        if not self.config.neuron.dont_save_events:
            logger.log("EVENTS", "events", **event)

        if self.config.wandb.off:
            return

        if not getattr(self, "wandb", None):
            init_wandb(self)

        # Log the event to wandb.
        self.wandb.log(event)


current_directory = "/home/spunion/folding/data/8emf/deliverable1/no_seed_1"
gmx_command_1 = "gmx grompp -f e.mdp -c 8.gro -p topol.top -o e.tpr -maxwarn 100"

test_commands = [
    "cd /home/spunion/folding/data/8emf/deliverable1/no_seed_1",
    gmx_command_1,
]

RunAndLog_instance = RunAndLog()
RunAndLog_instance.run_commands(
    commands=test_commands, suppress_cmd_output=True, verbose=False
)
