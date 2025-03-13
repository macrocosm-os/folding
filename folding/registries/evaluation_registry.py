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

    # TODO: Refactor this method to be more modular, seperate getting energy and setting up simulations and files

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

            self.steps_to_run = min(
                c.MAX_SIMULATION_STEPS_FOR_EVALUATION, self.log_step - self.cpt_step
            )

            # Create the state file here because it could have been loaded after MIN_SIMULATION_STEPS check
            simulation.saveState(self.state_xml_path)

            # Save the system config to the miner data directory
            system_config_path = os.path.join(
                self.miner_data_directory, f"miner_system_config_{self.miner_seed}.pkl"
            )
            if not os.path.exists(system_config_path):
                write_pkl(
                    data=self.system_config,
                    path=system_config_path,
                    write_mode="wb",
                )

            self.max_step = self.cpt_step + self.steps_to_run
            self.miner_energies: np.ndarray = self.log_file[
                (self.log_file['#"Step"'] > self.cpt_step)
                & (self.log_file['#"Step"'] <= self.max_step)
            ]["Potential Energy (kJ/mole)"].values

            miner_velm_data = create_velm(simulation=simulation)

            if not self.check_masses(miner_velm_data):
                raise ValidationError(
                    f"Miner {self.hotkey_alias} has modified the system in unintended ways... Skipping!"
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
        return float(np.median(self.miner_energies[-c.ENERGY_WINDOW_SIZE :]))

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
            Tuple[bool, list, list, str]: True if the run is valid, False otherwise.
                The two lists contain the potential energy values from the current simulation and the reference log file.
                The string contains the reason for the run being invalid.
        """

        # This is where we are going to check the xml files for the state.
        logger.info(
            f"Recreating simulation for {self.pdb_id} for state-based analysis..."
        )
        simulation, system_config = self.md_simulator.create_simulation(
            pdb=load_pdb_file(pdb_file=self.pdb_location),
            system_config=self.system_config.get_config(),
            seed=self.miner_seed,
            initialize_with_solvent=False,
        )
        simulation.loadState(self.state_xml_path)
        state_energies = []
        for _ in range(self.steps_to_run // 10):
            simulation.step(10)
            energy = (
                simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            )
            state_energies.append(energy)

        try:
            if not self.check_gradient(check_energies=state_energies):
                logger.warning(f"state energies: {state_energies}")
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-gradient check for {self.pdb_id}, ... Skipping!"
                )
                raise ValidationError(message="state-gradient")

            # Reload in the checkpoint file and run the simulation for the same number of steps as the miner.
            simulation, system_config = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
                initialize_with_solvent=False,
            )
            simulation.loadCheckpoint(self.checkpoint_path)

            current_state_logfile = os.path.join(
                self.miner_data_directory, f"check_{self.current_state}.log"
            )
            simulation.reporters.append(
                app.StateDataReporter(
                    current_state_logfile,
                    10,
                    step=True,
                    potentialEnergy=True,
                )
            )

            logger.info(
                f"Running {self.steps_to_run} steps. log_step: {self.log_step}, cpt_step: {self.cpt_step}"
            )

            simulation.step(self.steps_to_run)

            check_log_file = pd.read_csv(current_state_logfile)
            check_energies: np.ndarray = check_log_file[
                "Potential Energy (kJ/mole)"
            ].values

            if len(np.unique(check_energies)) == 1:
                logger.warning(
                    "All energy values in reproduced simulation are the same. Skipping!"
                )
                raise ValidationError(message="reprod-energies-identical")

            if not self.check_gradient(check_energies=check_energies):
                logger.warning(f"check_energies: {check_energies}")
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed cpt-gradient check for {self.pdb_id}, ... Skipping!"
                )
                raise ValidationError(message="cpt-gradient")

            if not self.compare_state_to_cpt(
                state_energies=state_energies, checkpoint_energies=check_energies
            ):
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-checkpoint comparison for {self.pdb_id}, ... Skipping!"
                )
                raise ValidationError(message="state-checkpoint")

            # calculating absolute percent difference per step
            percent_diff = abs(
                ((check_energies - self.miner_energies) / self.miner_energies) * 100
            )
            median_percent_diff = np.median(percent_diff)

            if median_percent_diff > c.ANOMALY_THRESHOLD:
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed anomaly check for {self.pdb_id}, with median percent difference: {median_percent_diff} ... Skipping!"
                )
                raise ValidationError(message="anomaly")

            # Validate intermediate checkpoints if the required parameters are provided
            if validator is not None and job_id is not None and axon is not None:
                (
                    intermediate_valid,
                    intermediate_result,
                ) = await self.validate_intermediate_checkpoints(
                    validator=validator, job_id=job_id, axon=axon
                )
                if not intermediate_valid:
                    raise ValidationError(message=intermediate_result)

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

            return True, check_energies.tolist(), self.miner_energies.tolist(), "valid"

        except ValidationError as E:
            logger.warning(f"{E}")
            return False, [], [], E.message

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
        is_valid, checked_energies, miner_energies, result = await self.is_run_valid(
            validator=validator, job_id=job_id, axon=axon
        )
        if not is_valid:
            return 0.0, checked_energies, miner_energies, result

        return (
            np.median(checked_energies[-c.ENERGY_WINDOW_SIZE :]),
            checked_energies,
            miner_energies,
            result,
        )  # Last portion of the reproduced energy vector

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
        """Get the intermediate checkpoints from the miner."""

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

    async def validate_intermediate_checkpoints(self, validator, job_id, axon):
        """
        Validate intermediate checkpoints from the miner.

        Args:
            validator: Validator instance to query for intermediate checkpoints
            job_id: Job ID to pass to get_intermediate_checkpoints
            axon: Axon to query for intermediate checkpoints

        Returns:
            bool: True if validation passes, False if it fails
            str: Reason for failure or "valid"
        """
        logger.info(f"Validating intermediate checkpoints for {self.pdb_id}...")
        try:
            # k random numbers indicating which checkpoints we want
            checkpoint_numbers = np.random.randint(
                0,
                int(self.log_file['#"Step"'].iloc[-1] / 10000) - 1,
                size=c.MAX_CHECKPOINTS_TO_VALIDATE,
            ).tolist()

            # Get intermediate checkpoints from the miner
            intermediate_checkpoints = await self.get_intermediate_checkpoints(
                validator=validator,
                job_id=job_id,
                axon=axon,
                checkpoint_numbers=checkpoint_numbers,
            )

            # If we have intermediate checkpoints, validate them
            if (
                intermediate_checkpoints
                and len(intermediate_checkpoints) == c.MAX_CHECKPOINTS_TO_VALIDATE
            ):
                # Track if any intermediate checkpoint fails
                failed_checkpoints = []

                for checkpoint_num, checkpoint_data in intermediate_checkpoints.items():
                    if checkpoint_data is None:
                        logger.warning(
                            f"Checkpoint {checkpoint_num} is None. Skipping!"
                        )
                        failed_checkpoints.append(checkpoint_num)
                        continue

                    # Save the checkpoint data to a temporary file
                    temp_checkpoint_path = os.path.join(
                        self.miner_data_directory, f"intermediate_{checkpoint_num}.cpt"
                    )
                    with open(temp_checkpoint_path, "wb") as f:
                        f.write(checkpoint_data)

                    # Create a simulation with the checkpoint
                    intermediate_simulation, _ = self.md_simulator.create_simulation(
                        pdb=load_pdb_file(pdb_file=self.pdb_location),
                        system_config=self.system_config.get_config(),
                        seed=self.miner_seed,
                        initialize_with_solvent=False,
                    )

                    # Load the checkpoint and run a short simulation
                    try:
                        intermediate_simulation.loadCheckpoint(temp_checkpoint_path)

                        # Create a log file for the intermediate checkpoint
                        intermediate_logfile = os.path.join(
                            self.miner_data_directory,
                            f"intermediate_{checkpoint_num}.log",
                        )
                        intermediate_simulation.reporters.append(
                            app.StateDataReporter(
                                intermediate_logfile,
                                10,
                                step=True,
                                potentialEnergy=True,
                            )
                        )

                        # Run a short simulation from the intermediate checkpoint
                        steps_to_run = c.INTERMEDIATE_CHECKPOINT_STEPS
                        intermediate_simulation.step(steps_to_run)

                        # Read the log file and check the energies
                        intermediate_log = pd.read_csv(intermediate_logfile)

                        # Find entries in the log file where intermediate_log['#"Step"'] is equal to self.log_file['#"Step"']
                        relevant_log_entries = pd.merge(
                            self.log_file,
                            intermediate_log,
                            on='#"Step"',
                            how="inner",
                            suffixes=("_original", "_intermediate"),
                        )

                        intermediate_energies = np.array(
                            relevant_log_entries[
                                "Potential Energy (kJ/mole)_intermediate"
                            ].values
                        )

                        # Check that there's some variation in the energies
                        if len(np.unique(intermediate_energies)) == 1:
                            logger.warning(
                                f"All energy values in intermediate checkpoint {checkpoint_num} simulation are the same. Skipping!"
                            )
                            failed_checkpoints.append(checkpoint_num)
                            continue

                        # Check gradient on intermediate checkpoint energies
                        if not self.check_gradient(
                            check_energies=intermediate_energies
                        ):
                            logger.warning(
                                f"intermediate_energies: {intermediate_energies}"
                            )
                            logger.warning(
                                f"hotkey {self.hotkey_alias} failed gradient check for intermediate checkpoint {checkpoint_num}, ... Skipping!"
                            )
                            failed_checkpoints.append(checkpoint_num)
                            continue

                        # Extract the original reported energies for this checkpoint range
                        original_energies = np.array(
                            relevant_log_entries[
                                "Potential Energy (kJ/mole)_original"
                            ].values
                        )

                        if len(original_energies) == 0:
                            logger.warning(
                                f"No original energies found for checkpoint {checkpoint_num}. Skipping percent diff check."
                            )
                        else:

                            # plotting intermediate log and original log in the same plot and save it as an image
                            fig = go.Figure()

                            # Add first trace with name for legend
                            fig.add_trace(
                                go.Scatter(
                                    x=relevant_log_entries['#"Step"'],
                                    y=relevant_log_entries[
                                        "Potential Energy (kJ/mole)_original"
                                    ],
                                    name="Original",  # This will appear in the legend
                                    line=dict(color="red"),
                                )
                            )

                            # Add second trace with name for legend
                            fig.add_trace(
                                go.Scatter(
                                    x=relevant_log_entries['#"Step"'],
                                    y=relevant_log_entries[
                                        "Potential Energy (kJ/mole)_intermediate"
                                    ],
                                    name="Intermediate",  # This will appear in the legend
                                    line=dict(color="blue"),
                                )
                            )

                            # Update layout
                            fig.update_layout(
                                title=f"Intermediate Checkpoint {checkpoint_num}",
                                xaxis_title="Step",
                                yaxis_title="Energy (kJ/mole)",
                                legend_title="Data Source",
                            )

                            fig.write_image(
                                os.path.join(
                                    self.miner_data_directory,
                                    f"checkpoint_{checkpoint_num}.png",
                                )
                            )
                            # Calculate the percent difference between simulated and reported energies
                            # Trim to the shortest length to ensure we're comparing corresponding steps
                            min_length = min(
                                len(intermediate_energies), len(original_energies)
                            )
                            intermediate_energies_trimmed = intermediate_energies[
                                :min_length
                            ]
                            original_energies_trimmed = original_energies[:min_length]

                            # Calculate absolute percent difference
                            percent_diff = abs(
                                (
                                    (
                                        intermediate_energies_trimmed
                                        - original_energies_trimmed
                                    )
                                    / original_energies_trimmed
                                )
                                * 100
                            )
                            median_percent_diff = np.median(percent_diff)

                            # Check if the percent difference is within acceptable limits
                            if median_percent_diff > c.ANOMALY_THRESHOLD:
                                logger.warning(
                                    f"hotkey {self.hotkey_alias} failed anomaly check for intermediate checkpoint {checkpoint_num}, "
                                    f"with median percent difference: {median_percent_diff} ... Skipping!"
                                )
                                failed_checkpoints.append(checkpoint_num)
                                continue
                            logger.info(
                                f"Intermediate checkpoint {checkpoint_num} percent diff check passed with median diff: {median_percent_diff}%"
                            )

                        logger.info(
                            f"Intermediate checkpoint {checkpoint_num} validation passed."
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error validating intermediate checkpoint {checkpoint_num}: {e}"
                        )
                        failed_checkpoints.append(checkpoint_num)

                # If more than half of the checkpoints failed, consider the miner's work invalid
                if len(failed_checkpoints) > len(intermediate_checkpoints) / 2:
                    logger.warning(
                        f"More than half of intermediate checkpoints failed validation: {failed_checkpoints}"
                    )
                    return False, "intermediate-checkpoints-majority-failed"

                logger.info("All intermediate checkpoints passed validation.")
                return True, "valid"
            else:
                logger.warning(
                    f"Not enough intermediate checkpoints received from miner. Asked for cpts {checkpoint_numbers} but got {intermediate_checkpoints.keys()}"
                )
                return False, "not-enough-intermediate-checkpoints"

        except Exception as e:
            logger.warning(
                f"Error during intermediate checkpoint validation: {traceback.format_exc()}"
            )
            return False, "intermediate-checkpoints-error"


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
