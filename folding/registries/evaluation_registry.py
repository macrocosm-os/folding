import os
from typing import Any, Dict, List, Union
import traceback

import numpy as np
import pandas as pd
from openmm import app
import bittensor as bt
import plotly.graph_objects as go

from folding.base.evaluation import BaseEvaluator
from folding.base.simulation import OpenMMSimulation
from folding.utils import constants as c
from folding.utils.logger import logger
from folding.utils.ops import (
    ValidationError,
    create_velm,
    load_pdb_file,
    load_pkl,
    save_files,
    save_pdb,
    write_pkl,
)
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.protocol import IntermediateSubmissionSynapse


class SyntheticMDEvaluator(BaseEvaluator):
    def __init__(
        self,
        pdb_id: str,
        pdb_location: str,
        hotkey: str,
        state: str,
        seed: int,
        md_output: dict,
        basepath: str,
        system_config: SimulationConfig,
        velm_array_pkl_path: str,
        **kwargs,
    ):
        self.pdb_id = pdb_id
        self.kwargs = kwargs
        self.md_simulator = OpenMMSimulation()
        self.pdb_location = pdb_location
        self.hotkey_alias = hotkey[:8]
        self.current_state = state
        self.miner_seed = seed
        self.md_output = md_output
        self.basepath = basepath

        self.system_config = system_config

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {
            k.split(".")[-1]: k for k, v in self.md_output.items() if len(v) > 0
        }
        self.miner_data_directory = os.path.join(self.basepath, self.hotkey_alias)
        self.velm_array_pkl_path = velm_array_pkl_path

    def process_md_output(self) -> bool:
        """Method to process molecular dynamics data from a miner and recreate the simulation.

        Args:
            md_output (dict): Data from the miner containing the output files.
            seed (int): Miner's seed used in the simulation.
            state (str): The state of the simulation.
            hotkey (str): The hotkey of the miner.
            basepath (str): The basepath of the validator to hold all the miner data.
            pdb_location (str): The location of the pdb file.

        Raises:
            ValidationError: Miner not running enough simulation steps

        Returns:
            bool: True if the simulation was successfully recreated, False otherwise.
        """

        required_files_extensions = ["cpt", "log"]

        if len(self.md_output.keys()) == 0:
            logger.warning(
                f"Miner {self.hotkey_alias} returned empty md_output... Skipping!"
            )
            return False

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                logger.error(f"Missing file with extension {ext} in md_output")
                return False

        # Save files so we can check the hash later.
        save_files(
            files=self.md_output,
            output_directory=self.miner_data_directory,
        )

        try:
            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.

            logger.info(
                f"Recreating miner {self.hotkey_alias} simulation in state: {self.current_state}"
            )
            simulation, self.system_config = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
                initialize_with_solvent=False,
            )

            checkpoint_path = os.path.join(
                self.miner_data_directory, f"{self.current_state}.cpt"
            )
            state_xml_path = os.path.join(
                self.miner_data_directory, f"{self.current_state}.xml"
            )
            log_file_path = os.path.join(
                self.miner_data_directory, self.md_outputs_exts["log"]
            )

            simulation.loadCheckpoint(checkpoint_path)

            self.log_file = pd.read_csv(log_file_path)
            self.log_step = self.log_file['#"Step"'].iloc[-1]

            # Checks to see if we have enough steps in the log file to start validation
            if len(self.log_file) < c.MIN_LOGGING_ENTRIES:
                raise ValidationError(
                    f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                )

            # Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
            # Checks if log_file is MIN_STEPS steps ahead of checkpoint
            if (self.log_step - simulation.currentStep) < c.MIN_SIMULATION_STEPS:
                # If the miner did not run enough steps, we will load the old checkpoint
                checkpoint_path = os.path.join(
                    self.miner_data_directory, f"{self.current_state}_old.cpt"
                )
                if os.path.exists(checkpoint_path):
                    logger.warning(
                        f"Miner {self.hotkey_alias} did not run enough steps since last checkpoint... Loading old checkpoint"
                    )
                    simulation.loadCheckpoint(checkpoint_path)
                    # Checking to see if the old checkpoint has enough steps to validate
                    if (
                        self.log_step - simulation.currentStep
                    ) < c.MIN_SIMULATION_STEPS:
                        raise ValidationError(
                            f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                        )
                else:
                    raise ValidationError(
                        f"Miner {self.hotkey_alias} did not run enough steps and no old checkpoint found... Skipping!"
                    )

            self.cpt_step = simulation.currentStep
            self.checkpoint_path = checkpoint_path
            self.state_xml_path = state_xml_path

            # Save the system config to the miner data directory
            system_config_path = os.path.join(
                self.miner_data_directory, f"miner_system_config_{self.miner_seed}.pkl"
            )

            self.final_miner_energies = self.log_file[
                (self.log_file['#"Step"'] > self.cpt_step)
            ]["Potential Energy (kJ/mole)"].values

            if not os.path.exists(system_config_path):
                write_pkl(
                    data=self.system_config,
                    path=system_config_path,
                    write_mode="wb",
                )

            miner_velm_data = create_velm(simulation=simulation)

            if not self.check_masses(miner_velm_data):
                raise ValidationError(
                    f"Miner {self.hotkey_alias} has modified the system in unintended ways... Skipping!"
                )
            self.number_of_checkpoints = (
                int(self.log_file['#"Step"'].iloc[-1] / 10000) - 1
            )
            if self.number_of_checkpoints <= c.MAX_CHECKPOINTS_TO_VALIDATE:
                raise ValidationError(
                    f"Not enough intermediate checkpoints generated by miner. Need {c.MAX_CHECKPOINTS_TO_VALIDATE} but miner has generated {self.number_of_checkpoints}."
                )

        except ValidationError as E:
            logger.warning(f"{E}")
            return False

        except Exception as e:
            logger.error(f"Failed to recreate simulation: {e}")
            return False

        return True

    def get_reported_energy(self) -> float:
        """Get the energy from the simulation"""

        return float(np.median(self.final_miner_energies[-c.ENERGY_WINDOW_SIZE :]))

    def check_masses(self, miner_velm_data) -> bool:
        """
        Check if the masses reported in the miner file are identical to the masses given
        in the initial pdb file. If not, they have modified the system in unintended ways.

        Reference:
        https://github.com/openmm/openmm/blob/53770948682c40bd460b39830d4e0f0fd3a4b868/platforms/common/src/kernels/langevinMiddle.cc#L11
        """

        validator_velm_data = load_pkl(self.velm_array_pkl_path, "rb")

        validator_masses = validator_velm_data["pdb_masses"]
        miner_masses = miner_velm_data["pdb_masses"]

        for i, (v_mass, m_mass) in enumerate(zip(validator_masses, miner_masses)):
            if v_mass != m_mass:
                logger.error(
                    f"Masses for atom {i} do not match. Validator: {v_mass}, Miner: {m_mass}"
                )
                return False
        return True

    def check_gradient(self, check_energies: np.ndarray) -> bool:
        """This method checks the gradient of the potential energy within the first
        WINDOW size of the check_energies array. Miners that return gradients that are too high,
        there is a *high* probability that they have not run the simulation as the validator specified.
        """
        mean_gradient = np.diff(check_energies[: c.GRADIENT_WINDOW_SIZE]).mean().item()
        return (
            mean_gradient <= c.GRADIENT_THRESHOLD
        )  # includes large negative gradients is passible

    def compare_state_to_cpt(
        self, state_energies: list, checkpoint_energies: list
    ) -> bool:
        """
        Check if the state file is the same as the checkpoint file by comparing the median of the first few energy values
        in the simulation created by the checkpoint and the state file respectively.
        """

        state_energies = np.array(state_energies)
        checkpoint_energies = np.array(checkpoint_energies)

        state_median = np.median(state_energies[: c.GRADIENT_WINDOW_SIZE])
        checkpoint_median = np.median(checkpoint_energies[: c.GRADIENT_WINDOW_SIZE])

        percent_diff = abs((state_median - checkpoint_median) / checkpoint_median) * 100

        if percent_diff > c.XML_CHECKPOINT_THRESHOLD:
            return False
        return True

    async def is_run_valid(self, validator=None, job_id=None, axon=None):
        """
        Checks if the run is valid by evaluating a set of logical conditions:

        1. comparing the potential energy values between the current simulation and a reference log file.
        2. ensuring that the gradient of the minimization is within a certain threshold to prevent exploits.
        3. ensuring that the masses of the atoms in the simulation are the same as the masses in the original pdb file.
        4. If validator, job_id, and axon are provided, it also validates intermediate checkpoints from the miner.

        Args:
            validator: Optional validator instance to query for intermediate checkpoints
            job_id: Optional job ID to pass to get_intermediate_checkpoints
            axon: Optional axon to query for intermediate checkpoints

        Returns:
            Tuple[bool, dict, dict, str]: True if the run is valid, False otherwise.
                The two dicts contain the potential energy values from the current simulation and the reference log file.
                The string contains the reason for the run being invalid.
        """
        try:
            checked_energies_dict = {}
            miner_energies_dict = {}

            logger.info(f"Checking if run is valid for {self.hotkey_alias}...")
            logger.info(f"Checking final checkpoint...")
            # Check the final checkpoint
            (
                is_valid,
                checked_energies,
                miner_energies,
                result,
            ) = self.is_checkpoint_valid(
                checkpoint_path=self.checkpoint_path,
                steps_to_run=3000,
                checkpoint_num="final",
            )
            checked_energies_dict["final"] = checked_energies
            miner_energies_dict["final"] = miner_energies

            if not is_valid:
                return False, checked_energies_dict, miner_energies_dict, result

            # Check the intermediate checkpoints
            if validator is not None and job_id is not None and axon is not None:
                checkpoint_numbers = np.random.randint(
                    0,
                    self.number_of_checkpoints,
                    size=c.MAX_CHECKPOINTS_TO_VALIDATE,
                ).tolist()

                # Get intermediate checkpoints from the miner
                intermediate_checkpoints = await self.get_intermediate_checkpoints(
                    validator=validator,
                    job_id=job_id,
                    axon=axon,
                    checkpoint_numbers=checkpoint_numbers,
                )

                if not intermediate_checkpoints:
                    return (
                        False,
                        checked_energies_dict,
                        miner_energies_dict,
                        "no-intermediate-checkpoints",
                    )

                # Validate each checkpoint
                for checkpoint_num, checkpoint_data in intermediate_checkpoints.items():
                    logger.info(f"Checking intermediate checkpoint {checkpoint_num}...")
                    if checkpoint_data is None:
                        return (
                            False,
                            checked_energies_dict,
                            miner_energies_dict,
                            f"checkpoint-is-none",
                        )

                    # Save the checkpoint data to a temporary file
                    temp_checkpoint_path = os.path.join(
                        self.miner_data_directory, f"intermediate_{checkpoint_num}.cpt"
                    )
                    with open(temp_checkpoint_path, "wb") as f:
                        f.write(checkpoint_data)
                    (
                        is_valid,
                        checked_energies,
                        miner_energies,
                        result,
                    ) = self.is_checkpoint_valid(
                        checkpoint_path=temp_checkpoint_path,
                        steps_to_run=c.INTERMEDIATE_CHECKPOINT_STEPS,
                        checkpoint_num=checkpoint_num,
                    )

                    checked_energies_dict[checkpoint_num] = checked_energies
                    miner_energies_dict[checkpoint_num] = miner_energies

                    if not is_valid:
                        return False, checked_energies_dict, miner_energies_dict, result

                return True, checked_energies_dict, miner_energies_dict, "valid"

        except ValidationError as E:
            logger.warning(f"{E}")
            return False, {}, {}, E.message

        return True, checked_energies_dict, miner_energies_dict, "valid"

    def evaluate(self) -> bool:
        """Checks to see if the miner's data can be passed for validation"""
        if not self.process_md_output():
            return False

        # Check to see if we have a logging resolution of 10 or better, if not the run is not valid
        if (self.log_file['#"Step"'][1] - self.log_file['#"Step"'][0]) > 10:
            return False

        return True

    async def validate(self, validator=None, job_id=None, axon=None):
        """
        Validate the run by checking if it's valid and returning the appropriate metrics.

        Args:
            validator: Optional validator instance to query for intermediate checkpoints
            job_id: Optional job ID to pass to get_intermediate_checkpoints
            axon: Optional axon to query for intermediate checkpoints

        Returns:
            Tuple containing the median energy, checked energies, miner energies, and result message
        """
        (
            is_valid,
            checked_energies_dict,
            miner_energies_dict,
            result,
        ) = await self.is_run_valid(validator=validator, job_id=job_id, axon=axon)

        if not is_valid:
            # Return empty dictionaries to maintain consistency
            return 0.0, checked_energies_dict, miner_energies_dict, result

        # Use the final checkpoint's energy for the score
        if "final" in checked_energies_dict and checked_energies_dict["final"]:
            final_energies = checked_energies_dict["final"]
            # Take the median of the last ENERGY_WINDOW_SIZE values
            median_energy = np.median(final_energies[-c.ENERGY_WINDOW_SIZE :])
            return median_energy, checked_energies_dict, miner_energies_dict, result
        else:
            # This should not happen if is_valid is True, but handle it just in case
            return (
                0.0,
                checked_energies_dict,
                miner_energies_dict,
                "missing-final-energies",
            )

    def _evaluate(self, data: Dict[str, Any]):
        pass

    def _validate(self):
        pass

    def get_ns_computed(self):
        """Calculate the number of nanoseconds computed by the miner."""

        return (self.cpt_step * self.system_config.time_step_size) / 1e3

    async def get_intermediate_checkpoints(
        self,
        validator: "Validator",
        job_id: str,
        axon: bt.Axon,
        checkpoint_numbers: list[int],
    ):
        """Get the intermediate checkpoints from the miner.

        Args:
            validator: The validator instance
            job_id: The job ID
            axon: The axon to query
            checkpoint_numbers: List of checkpoint numbers to retrieve
        """
        synapse = IntermediateSubmissionSynapse(
            job_id=job_id,
            checkpoint_numbers=checkpoint_numbers,
            pdb_id=self.pdb_id,
        )
        responses = await validator.dendrite.forward(
            synapse=synapse, axons=[axon], deserialize=True
        )
        return responses[0].cpt_files

    def name(self) -> str:
        return "SyntheticMD"

    def is_checkpoint_valid(
        self,
        checkpoint_path: str,
        steps_to_run: int = c.MIN_SIMULATION_STEPS,
        checkpoint_num: str = "final",
    ):
        """Validate a checkpoint by comparing energy values from simulation to reported values.

        This method avoids recreating simulation objects where possible by reusing them
        and resetting their state as needed.

        Args:
            checkpoint_path: Path to the checkpoint file to validate
            steps_to_run: Number of steps to run in the validation simulation

        Returns:
            Tuple of (is_valid, checked_energies, miner_energies, result_message)
        """
        # Create simulation once
        logger.info(
            f"Recreating simulation for {self.pdb_id}, checkpoint_num: {checkpoint_num} for state-based analysis..."
        )
        state_xml_path = os.path.join(
            self.miner_data_directory, f"{checkpoint_num}.xml"
        )

        # Load PDB file once
        pdb = load_pdb_file(pdb_file=self.pdb_location)
        system_config_dict = self.system_config.get_config()

        # Create initial simulation
        simulation, _ = self.md_simulator.create_simulation(
            pdb=pdb,
            system_config=system_config_dict,
            seed=self.miner_seed,
            initialize_with_solvent=False,
        )

        # Load checkpoint
        simulation.loadCheckpoint(checkpoint_path)
        current_cpt_step = simulation.currentStep

        if current_cpt_step + steps_to_run > self.log_step:
            raise ValidationError(message="simulation-step-out-of-range")

        # Save state to XML file
        simulation.saveState(state_xml_path)

        simulation, _ = self.md_simulator.create_simulation(
            pdb=pdb,
            system_config=system_config_dict,
            seed=self.miner_seed,
            initialize_with_solvent=False,
        )
        simulation.loadState(state_xml_path)

        # Run state simulation and collect energies
        state_energies = []
        for _ in range(steps_to_run // 10):
            simulation.step(10)
            energy = (
                simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            )
            state_energies.append(energy)

        try:
            if not self.check_gradient(check_energies=np.array(state_energies)):
                logger.warning(f"state energies: {state_energies}")
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-gradient check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, ... Skipping!"
                )
                raise ValidationError(message="state-gradient")

            # Create a reporter for the checkpoint simulation
            current_state_logfile = os.path.join(
                self.miner_data_directory, f"check_{checkpoint_num}.log"
            )

            simulation, _ = self.md_simulator.create_simulation(
                pdb=pdb,
                system_config=system_config_dict,
                seed=self.miner_seed,
                initialize_with_solvent=False,
            )

            # Load checkpoint
            simulation.loadCheckpoint(checkpoint_path)

            simulation.reporters.append(
                app.StateDataReporter(
                    current_state_logfile,
                    10,
                    step=True,
                    potentialEnergy=True,
                )
            )

            logger.info(
                f"Running {steps_to_run} steps. log_step: {self.log_step}, cpt_step: {current_cpt_step}"
            )

            # Run the checkpoint simulation
            simulation.step(steps_to_run)

            # Process results
            check_log_file = pd.read_csv(current_state_logfile)
            check_energies: np.ndarray = check_log_file[
                "Potential Energy (kJ/mole)"
            ].values

            max_step = current_cpt_step + steps_to_run

            miner_energies: np.ndarray = self.log_file[
                (self.log_file['#"Step"'] > current_cpt_step)
                & (self.log_file['#"Step"'] <= max_step)
            ]["Potential Energy (kJ/mole)"].values

            if len(np.unique(check_energies)) == 1:
                logger.warning(
                    "All energy values in reproduced simulation are the same. Skipping!"
                )
                raise ValidationError(message="reprod-energies-identical")

            if not self.check_gradient(check_energies=np.array(check_energies)):
                logger.warning(f"check_energies: {check_energies}")
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed cpt-gradient check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, ... Skipping!"
                )
                raise ValidationError(message="cpt-gradient")

            if not self.compare_state_to_cpt(
                state_energies=state_energies, checkpoint_energies=check_energies
            ):
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-checkpoint comparison for {self.pdb_id}, checkpoint_num: {checkpoint_num}, ... Skipping!"
                )
                raise ValidationError(message="state-checkpoint")

            # calculating absolute percent difference per step
            percent_diff = abs(
                ((check_energies - miner_energies) / miner_energies) * 100
            )
            median_percent_diff = np.median(percent_diff)

            if median_percent_diff > c.ANOMALY_THRESHOLD:
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed anomaly check for {self.pdb_id}, checkpoint_num: {checkpoint_num}, with median percent difference: {median_percent_diff} ... Skipping!"
                )
                raise ValidationError(message="anomaly")

            # Save the folded pdb file if the run is valid
            positions = simulation.context.getState(getPositions=True).getPositions()
            topology = simulation.topology

            save_pdb(
                positions=positions,
                topology=topology,
                output_path=os.path.join(
                    self.miner_data_directory, f"{self.pdb_id}_folded.pdb"
                ),
            )

            return True, check_energies.tolist(), miner_energies.tolist(), "valid"

        except ValidationError as E:
            logger.warning(f"{E}")
            return False, [], [], E.message


class OrganicMDEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMD"


class SyntheticMLEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "SyntheticML"


class OrganicMLEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicML"


EVALUATION_REGISTRY = {
    "SyntheticMD": SyntheticMDEvaluator,
    "OrganicMD": OrganicMDEvaluator,
    "SyntheticML": SyntheticMLEvaluator,
    "OrganicML": OrganicMLEvaluator,
}
