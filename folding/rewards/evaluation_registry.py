import os
from typing import Dict, Any

import pandas as pd

from folding.utils.logger import logger
from folding.base.evaluation import BaseEvaluator
from folding.base.simulation import OpenMMSimulation
from folding.utils import constants as c
from folding.utils.ops import (
    ValidationError,
    write_pkl,
    load_pdb_file,
    save_files,
)


class SyntheticMDEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs
        self.md_simulator = OpenMMSimulation()

    def process_md_output(
        self, md_output: dict, seed: int, state: str, hotkey: str, basepath: str, pdb_location: str, **kwargs
    ) -> bool:
        required_files_extensions = ["cpt", "log"]
        self.hotkey_alias = hotkey[:8]
        self.current_state = state
        self.miner_seed = seed

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {k.split(".")[-1]: k for k, v in md_output.items() if len(v) > 0}

        if len(md_output.keys()) == 0:
            logger.warning(f"Miner {self.hotkey_alias} returned empty md_output... Skipping!")
            return False

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                logger.error(f"Missing file with extension {ext} in md_output")
                return False

        self.miner_data_directory = os.path.join(basepath, hotkey[:8])

        # Save files so we can check the hash later.
        save_files(
            files=md_output,
            output_directory=self.miner_data_directory,
        )

        try:
            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.

            logger.info(f"Recreating miner {self.hotkey_alias} simulation in state: {self.current_state}")
            self.simulation, self.system_config = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
            )

            checkpoint_path = os.path.join(self.miner_data_directory, f"{self.current_state}.cpt")
            state_xml_path = os.path.join(self.miner_data_directory, f"{self.current_state}.xml")
            log_file_path = os.path.join(self.miner_data_directory, self.md_outputs_exts["log"])

            self.simulation.loadCheckpoint(checkpoint_path)

            self.log_file = pd.read_csv(log_file_path)
            self.log_step = self.log_file['#"Step"'].iloc[-1]

            # Checks to see if we have enough steps in the log file to start validation
            if len(self.log_file) < c.MIN_LOGGING_ENTRIES:
                raise ValidationError(
                    f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                )

            # Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
            # Checks if log_file is MIN_STEPS steps ahead of checkpoint
            if (self.log_step - self.simulation.currentStep) < c.MIN_SIMULATION_STEPS:
                # If the miner did not run enough steps, we will load the old checkpoint
                checkpoint_path = os.path.join(self.miner_data_directory, f"{self.current_state}_old.cpt")
                if os.path.exists(checkpoint_path):
                    logger.warning(
                        f"Miner {self.hotkey_alias} did not run enough steps since last checkpoint... Loading old checkpoint"
                    )
                    self.simulation.loadCheckpoint(checkpoint_path)
                    # Checking to see if the old checkpoint has enough steps to validate
                    if (self.log_step - self.simulation.currentStep) < c.MIN_SIMULATION_STEPS:
                        raise ValidationError(
                            f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                        )
                else:
                    raise ValidationError(
                        f"Miner {self.hotkey_alias} did not run enough steps and no old checkpoint found... Skipping!"
                    )

            self.cpt_step = self.simulation.currentStep
            self.checkpoint_path = checkpoint_path
            self.state_xml_path = state_xml_path

            # Create the state file here because it could have been loaded after MIN_SIMULATION_STEPS check
            self.simulation.saveState(self.state_xml_path)

            # Save the system config to the miner data directory
            system_config_path = os.path.join(self.miner_data_directory, f"miner_system_config_{seed}.pkl")
            if not os.path.exists(system_config_path):
                write_pkl(
                    data=self.system_config,
                    path=system_config_path,
                    write_mode="wb",
                )

        except ValidationError as E:
            logger.warning(f"{E}")
            return False

        except Exception as e:
            logger.error(f"Failed to recreate simulation: {e}")
            return False

        return True

    def _evaluate(self, data: Dict[str, Any]) -> float:
        if self.process_md_output(**data):
            return 1.0  # Reward for successful simulation
        return 0.0

    def name(self) -> str:
        return "SyntheticMDReward"


class OrganicMDEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMDReward"


class SyntheticMLEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "SyntheticMLReward"


class OrganicMLEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMLReward"


class EvaluationRegistry:
    """
    Handles the organization of all tasks that we want inside of SN25, which includes:
        - Molecular Dynamics (MD)
        - ML Inference

    It also attaches its corresponding reward pipelines.
    """

    def __init__(self):
        evaluation_pipelines = [SyntheticMDEvaluator, OrganicMDReward, SyntheticMLReward, OrganicMLReward]

        self.tasks = []
        for pipe in evaluation_pipelines:
            self.tasks.append(pipe().name())
