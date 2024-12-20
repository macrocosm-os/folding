import os
import time
import glob
import base64
import random
import shutil
import asyncio
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Literal, Any

import numpy as np
import pandas as pd
from openmm import app, unit
from pdbfixer import PDBFixer

from folding.base.simulation import OpenMMSimulation
from folding.store import Job
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.ops import (
    OpenMMException,
    ValidationError,
    write_pkl,
    load_pkl,
    check_and_download_pdbs,
    check_if_directory_exists,
    plot_miner_validator_curves,
)

from folding.utils.logger import logger

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Protein(OpenMMSimulation):
    PDB_RECORDS = ("ATOM", "ANISOU", "REMARK", "HETATM", "CONECT")

    @property
    def name(self):
        return self.protein_pdb.split(".")[0]

    def __init__(
        self,
        pdb_id: str,
        ff: str,
        water: str,
        box: Literal["cube", "dodecahedron", "octahedron"],
        config: Dict,
        system_kwargs: Dict,
        load_md_inputs: bool = False,
        epsilon: float = 1,  # percentage
        **kwargs,
    ) -> None:
        """The Protein class is responsible for handling the protein simulation.
        It attempts to setup the simulation environment to ensure that it can be run
        on the miner side. It also contains methods to validate the simulation outputs.

        Args:
            pdb_id (str): pdb_id of the protein, typically a 4 letter string.
            ff (str): the forcefield that will be used for simulation.
            water (str): water model that will be used for simulation.
            box (Literal): the shape of the box that will be used for simulation.
            config (Dict): bittensor config object.
            system_kwargs (Dict): system kwargs for the SimualtionConfig object.
            load_md_inputs (bool, optional): If we should load pre-comuted files for the protein. Defaults to False.
            epsilon (float, optional): The percentage improvement that must be achieved for this protein to be considered "better". Defaults to 1.
        """

        self.base_directory = os.path.join(str(ROOT_DIR), "data")

        self.pdb_id: str = pdb_id.lower()
        self.simulation_cpt = "em.cpt"
        self.simulation_pkl = f"config_{self.pdb_id}.pkl"

        self.setup_filepaths()

        self.ff: str = ff
        self.box: Literal["cube", "dodecahedron", "octahedron"] = box
        self.water: str = water

        # The dict is saved as a string in the event, so it needs to be evaluated.
        if isinstance(system_kwargs, str):
            system_kwargs = eval(system_kwargs)

        self.system_config = SimulationConfig(
            ff=self.ff,
            water=self.water,
            box=self.box,
            seed=1337,
            **system_kwargs,
        )

        self.config = config
        self.simulation: app.Simulation = None
        self.input_files = [self.em_cpt_location]

        self.md_inputs = (
            self.read_and_return_files(filenames=self.input_files)
            if load_md_inputs
            else {}
        )

        # set to an arbitrarily high number to ensure that the first miner is always accepted.
        self.init_energy = 0
        self.pdb_complexity = defaultdict(int)
        self.epsilon = epsilon

    def setup_filepaths(self):
        self.pdb_file = f"{self.pdb_id}.pdb"
        self.pdb_directory = os.path.join(self.base_directory, self.pdb_id)
        self.pdb_location = os.path.join(self.pdb_directory, self.pdb_file)

        self.validator_directory = os.path.join(self.pdb_directory, "validator")
        self.em_cpt_location = os.path.join(
            self.validator_directory, self.simulation_cpt
        )
        self.simulation_pkl_location = os.path.join(
            self.validator_directory, self.simulation_pkl
        )

        self.velm_array_pkl = os.path.join(
            self.pdb_directory, "velm_array_indicies.pkl"
        )

    @staticmethod
    async def from_job(job: Job, config: Dict):
        # Load_md_inputs is set to True to ensure that miners get files every query.
        protein = Protein(
            pdb_id=job.pdb,
            ff=job.ff,
            box=job.box,
            water=job.water,
            config=config,
            load_md_inputs=True,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
        )

        try:
            protein.pdb_complexity = Protein._get_pdb_complexity(protein.pdb_location)
            protein._calculate_epsilon()

            protein.pdb_contents = protein.load_pdb_as_string(protein.pdb_location)

        except Exception as E:
            logger.error(f"from_job failed for {protein.pdb_id} with Exception {E}.")
            return None
        return protein

    @staticmethod
    def load_pdb_as_string(pdb_path: str) -> str:
        with open(pdb_path, "r") as f:
            return f.read()

    @staticmethod
    def _get_pdb_complexity(pdb_path):
        """Get the complexity of the pdb file by counting the number of atoms, residues, etc."""
        pdb_complexity = defaultdict(int)
        with open(pdb_path, "r") as f:
            for line in f.readlines():
                # Check if the line starts with any of the PDB_RECORDS
                for key in Protein.PDB_RECORDS:
                    if line.strip().startswith(key):
                        pdb_complexity[key] += 1
        return pdb_complexity

    async def setup_pdb_directory(self):
        # if directory doesn't exist, download the pdb file and save it to the directory
        if not os.path.exists(self.pdb_directory):
            os.makedirs(self.pdb_directory)

        if not os.path.exists(os.path.join(self.pdb_directory, self.pdb_file)):
            if not await check_and_download_pdbs(
                pdb_directory=self.pdb_directory,
                pdb_id=self.pdb_file,
                input_source=self.config.input_source,
                download=True,
                force=self.config.force_use_pdb,
            ):
                raise Exception(
                    f"Failed to download {self.pdb_file} to {self.pdb_directory}"
                )
            await self.fix_pdb_file()
        else:
            logger.info(
                f"PDB file {self.pdb_file} already exists in path {self.pdb_directory!r}."
            )

    def read_and_return_files(self, filenames: List) -> Dict:
        """Read the files and return them as a dictionary."""
        files_to_return = {}
        for filename in filenames:
            for file in glob.glob(os.path.join(self.validator_directory, filename)):
                try:
                    # A bit of a hack to load in the data correctly depending on the file ext
                    name = file.split("/")[-1]
                    with open(file, "rb") as f:
                        if "cpt" in name:
                            files_to_return[name] = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )
                        else:
                            files_to_return[
                                name
                            ] = f.read()  # This would be the pdb file.

                except Exception:
                    continue
        return files_to_return

    async def setup_simulation(self):
        """forward method defines the following:
        1. gather the pdb_id and setup the namings.
        2. setup the pdb directory and download the pdb file if it doesn't exist.
        3. check for missing files and generate the input files if they are missing.
        4. edit the necessary config files and add them to the synapse object self.md_inputs[file] = content
        4. save the files to the validator directory for record keeping.
        """

        logger.info(
            f"Launching {self.pdb_id} Protein Job with the following configuration\nff : {self.ff}\nbox : {self.box}\nwater : {self.water}"
        )

        await self.setup_pdb_directory()
        await self.generate_input_files()

        # Create a validator directory to store the files
        check_if_directory_exists(output_directory=self.validator_directory)

        # Read the files that should exist now based on generate_input_files.
        self.md_inputs = self.read_and_return_files(filenames=self.input_files)

        self.save_files(
            files=self.md_inputs,
            output_directory=self.validator_directory,
            write_mode="w",
        )

        self.pdb_complexity = Protein._get_pdb_complexity(self.pdb_location)
        self.init_energy = self.calc_init_energy()

        # Checking if init energy is nan
        if np.isnan(self.init_energy):
            raise OpenMMException(
                f"Failed to calculate initial energy for {self.pdb_id}"
            )

        self._calculate_epsilon()

    def __str__(self):
        return f"Protein(pdb_id={self.pdb_id}, ff={self.ff}, box={self.box}, water={self.water})"

    def __repr__(self):
        return self.__str__()

    def load_pdb_file(self, pdb_file: str) -> app.PDBFile:
        """Method to take in the pdb file and load it into an OpenMM PDBFile object."""
        return app.PDBFile(pdb_file)

    async def fix_pdb_file(self):
        """
        Protein Data Bank (PDB or PDBx/mmCIF) files often have a number of problems
        that must be fixed before they can be used in a molecular dynamics simulation.

        The fixer will remove metadata that is contained in the header of the original pdb, and we might
        want to keep this. Therefore, we will rename the original pdb file to *_original.pdb and make a new
        pdb file using the PDBFile.writeFile method.

        Reference docs for the PDBFixer class can be found here:
            https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
        """

        fixer = PDBFixer(filename=self.pdb_location)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        original_pdb = self.pdb_location.split(".")[0] + "_original.pdb"
        os.rename(self.pdb_location, original_pdb)

        app.PDBFile.writeFile(
            topology=fixer.topology,
            positions=fixer.positions,
            file=open(self.pdb_location, "w"),
        )

    def create_velm(self, simulation: app.Simulation) -> Dict[str, Any]:
        """Alters the initial state of the simulation using initial velocities
        at the beginning and the end of the protein chain to use as a lookup in memory.

        Args:
            simulation (app.Simulation): The simulation object.

        Returns:
            simulation: original simulation object.
        """

        mass_index = 0
        mass_indicies = []
        atom_masses: List[unit.quantity.Quantity] = []

        while True:
            try:
                atom_masses.append(simulation.system.getParticleMass(mass_index))
                mass_indicies.append(mass_index)
                mass_index += 1
            except:
                # When there are no more atoms.
                break

        velm = {
            "mass_indicies": mass_indicies,
            "pdb_masses": atom_masses,
        }

        return velm

    # Function to generate the OpenMM simulation state.
    @OpenMMSimulation.timeit
    async def generate_input_files(self):
        """Generate_input_files method defines the following:
        1. Load the pdb file and create the simulation object.
        2. Minimize the energy of the system.
        3. Save the checkpoint file.
        4. Save the system config to the validator directory.
        5. Move all files except the pdb file to the validator directory.
        """

        logger.info(f"Changing path to {self.pdb_directory}")
        os.chdir(self.pdb_directory)

        logger.info(
            f"pdb file is set to: {self.pdb_file}, and it is located at {self.pdb_location}"
        )

        self.simulation, self.system_config = await asyncio.to_thread(
            self.create_simulation,
            self.load_pdb_file(pdb_file=self.pdb_file),
            self.system_config.get_config(),
            "em",
        )

        # load in information from the velm memory
        velm = self.create_velm(simulation=self.simulation)
        write_pkl(
            data=velm,
            path=self.velm_array_pkl,
            write_mode="wb",
        )

        logger.info(f"Minimizing energy for pdb: {self.pdb_id} ...")
        start_time = time.time()
        await asyncio.to_thread(
            self.simulation.minimizeEnergy, maxIterations=100
        )  # TODO: figure out the right number for this
        logger.warning(f"Minimization took {time.time() - start_time:.4f} seconds")
        await asyncio.to_thread(self.simulation.step, 1000)

        self.simulation.saveCheckpoint("em.cpt")

        # This is only for the validators, as they need to open the right config later.
        # Only save the config if the simulation was successful.
        write_pkl(data=self.system_config, path=self.simulation_pkl, write_mode="wb")

        # Here we are going to change the path to a validator folder, and move ALL the files except the pdb file
        check_if_directory_exists(output_directory=self.validator_directory)
        # Move all files
        cmd = f'find . -maxdepth 1 -type f ! -name "*.pdb" -exec mv {{}} {self.validator_directory}/ \;'
        logger.debug(f"Moving all files except pdb to {self.validator_directory}")
        os.system(cmd)

    def gen_seed(self):
        """Generate a random seed"""
        return random.randint(1000, 999999)

    def save_files(
        self, files: Dict, output_directory: str, write_mode: str = "wb"
    ) -> Dict:
        """Save the simulation files generated on the validator side to a desired output directory.

        Args:
            files (Dict): Dictionary mapping between filename and content
            output_directory (str)
            write_mode (str, optional): How the file should be written. Defaults to "wb".

        Returns:
            _type_: _description_
        """
        logger.info(f"⏰ Saving files to {output_directory}...")
        check_if_directory_exists(output_directory=output_directory)

        filetypes = {}
        for filename, content in files.items():
            filetypes[filename.split(".")[-1]] = filename

            logger.info(f"Saving file {filename} to {output_directory}")
            if "em.cpt" in filename:
                filename = "em_binary.cpt"

            # loop over all of the output files and save to local disk
            with open(os.path.join(output_directory, filename), write_mode) as f:
                f.write(content)

        return filetypes

    def delete_files(self, directory: str):
        logger.info(f"Deleting files in {directory}")
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
        # os.rmdir(output_directory)

    def get_miner_data_directory(self, hotkey: str):
        self.miner_data_directory = os.path.join(self.validator_directory, hotkey[:8])

    def process_md_output(
        self, md_output: dict, seed: int, state: str, hotkey: str
    ) -> bool:
        MIN_LOGGING_ENTRIES = 500
        MIN_SIMULATION_STEPS = 5000

        required_files_extensions = ["cpt", "log"]
        self.hotkey_alias = hotkey[:8]
        self.current_state = state

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {
            k.split(".")[-1]: k for k, v in md_output.items() if len(v) > 0
        }

        if len(md_output.keys()) == 0:
            logger.warning(
                f"Miner {self.hotkey_alias} returned empty md_output... Skipping!"
            )
            return False

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                logger.error(f"Missing file with extension {ext} in md_output")
                return False

        self.get_miner_data_directory(hotkey=hotkey)

        # Save files so we can check the hash later.
        self.save_files(
            files=md_output,
            output_directory=self.miner_data_directory,
        )
        try:
            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.

            logger.info(
                f"Recreating miner {self.hotkey_alias} simulation in state: {self.current_state}"
            )
            self.simulation, self.system_config = self.create_simulation(
                pdb=self.load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=seed,
                state=state,
            )

            checkpoint_path = os.path.join(
                self.miner_data_directory, f"{self.current_state}.cpt"
            )

            log_file_path = os.path.join(
                self.miner_data_directory, self.md_outputs_exts["log"]
            )

            self.simulation.loadCheckpoint(checkpoint_path)
            self.log_file = pd.read_csv(log_file_path)
            self.log_step = self.log_file['#"Step"'].iloc[-1]

            # Checks to see if we have enough steps in the log file to start validation
            if len(self.log_file) < MIN_LOGGING_ENTRIES:
                raise ValidationError(
                    f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                )

            # Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
            # Checks if log_file is MIN_STEPS steps ahead of checkpoint
            if (self.log_step - self.simulation.currentStep) < MIN_SIMULATION_STEPS:
                # If the miner did not run enough steps, we will load the old checkpoint
                checkpoint_path = os.path.join(
                    self.miner_data_directory, f"{self.current_state}_old.cpt"
                )
                if os.path.exists(checkpoint_path):
                    logger.warning(
                        f"Miner {self.hotkey_alias} did not run enough steps since last checkpoint... Loading old checkpoint"
                    )
                    self.simulation.loadCheckpoint(checkpoint_path)
                    # Checking to see if the old checkpoint has enough steps to validate
                    if (
                        self.log_step - self.simulation.currentStep
                    ) < MIN_SIMULATION_STEPS:
                        raise ValidationError(
                            f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                        )
                else:
                    raise ValidationError(
                        f"Miner {self.hotkey_alias} did not run enough steps and no old checkpoint found... Skipping!"
                    )

            self.cpt_step = self.simulation.currentStep
            self.checkpoint_path = checkpoint_path

            # Save the system config to the miner data directory
            system_config_path = os.path.join(
                self.miner_data_directory, f"miner_system_config_{seed}.pkl"
            )
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

    def check_gradient(self, check_energies: np.ndarray) -> True:
        """This method checks the gradient of the potential energy within the first
        WINODW size of the check_energies array. Miners that return gradients that are too high,
        there is a *high* probability that they have not run the simulation as the validator specified.
        """
        WINDOW = 50  # Number of steps to calculate the gradient over
        GRADIENT_THRESHOLD = 10  # kJ/mol/nm

        mean_gradient = np.diff(check_energies[:WINDOW]).mean().item()
        return (
            mean_gradient <= GRADIENT_THRESHOLD
        )  # includes large negative gradients is passible

    def check_masses(self) -> bool:
        """
        Check if the masses reported in the miner file are identical to the masses given
        in the initial pdb file. If not, they have modified the system in unintended ways.

        Reference:
        https://github.com/openmm/openmm/blob/53770948682c40bd460b39830d4e0f0fd3a4b868/platforms/common/src/kernels/langevinMiddle.cc#L11
        """

        validator_velm_data = load_pkl(self.velm_array_pkl, "rb")
        miner_velm_data = self.create_velm(simulation=self.simulation)

        validator_masses = validator_velm_data["pdb_masses"]
        miner_masses = miner_velm_data["pdb_masses"]

        for i, (v_mass, m_mass) in enumerate(zip(validator_masses, miner_masses)):
            if v_mass != m_mass:
                logger.error(
                    f"Masses for atom {i} do not match. Validator: {v_mass}, Miner: {m_mass}"
                )
                return False
        return True

    def is_run_valid(self):
        """
        Checks if the run is valid by evaluating a set of logical conditions:

        1. comparing the potential energy values between the current simulation and a reference log file.
        2. ensuring that the gradient of the minimization is within a certain threshold to prevent exploits.
        3. ensuring that the masses of the atoms in the simulation are the same as the masses in the original pdb file.


        Returns:
            Tuple[bool, list, list]: True if the run is valid, False otherwise.
                The two lists contain the potential energy values from the current simulation and the reference log file.
        """

        # The percentage that we allow the energy to differ from the miner to the validator.
        ANOMALY_THRESHOLD = 0.5

        if not self.check_masses():
            return False, [], [], "masses"

        # Check to see if we have a logging resolution of 10 or better, if not the run is not valid
        if (self.log_file['#"Step"'][1] - self.log_file['#"Step"'][0]) > 10:
            return False, [], [], "logging_resolution"

        # Run the simulation at most 3000 steps
        steps_to_run = min(3000, self.log_step - self.cpt_step)

        self.simulation.reporters.append(
            app.StateDataReporter(
                os.path.join(
                    self.miner_data_directory, f"check_{self.current_state}.log"
                ),
                10,
                step=True,
                potentialEnergy=True,
            )
        )

        logger.info(
            f"Running {steps_to_run} steps. log_step: {self.log_step}, cpt_step: {self.cpt_step}"
        )

        max_step = self.cpt_step + steps_to_run

        miner_energies: np.ndarray = self.log_file[
            (self.log_file['#"Step"'] > self.cpt_step)
            & (self.log_file['#"Step"'] <= max_step)
        ]["Potential Energy (kJ/mole)"].values

        self.simulation.step(steps_to_run)

        check_log_file = pd.read_csv(
            os.path.join(self.miner_data_directory, f"check_{self.current_state}.log")
        )

        check_energies: np.ndarray = check_log_file["Potential Energy (kJ/mole)"].values

        if len(np.unique(check_energies)) == 1:
            logger.warning(
                "All energy values in reproduced simulation are the same. Skipping!"
            )
            return False, [], [], "energies_non_unique"

        if not self.check_gradient(check_energies=check_energies):
            logger.warning(
                f"hotkey {self.hotkey_alias} failed gradient check for {self.pdb_id}, ... Skipping!"
            )
            return False, [], [], "gradient"

        # calculating absolute percent difference per step
        percent_diff = abs(((check_energies - miner_energies) / miner_energies) * 100)
        median_percent_diff = np.median(percent_diff)

        # Plot and save miner/validator energy curves for logging
        plot_miner_validator_curves(
            miner_energies=miner_energies,
            check_energies=check_energies,
            percent_diff=percent_diff,
            miner_data_directory=self.miner_data_directory,
            pdb_id=self.pdb_id,
            current_state=self.current_state,
            cpt_step=self.cpt_step,
        )

        if median_percent_diff > ANOMALY_THRESHOLD:
            return False, check_energies.tolist(), miner_energies.tolist(), "anomaly"

        # Save the folded pdb file if the run is valid
        self.save_pdb(
            output_path=os.path.join(
                self.miner_data_directory, f"{self.pdb_id}_folded.pdb"
            )
        )

        return True, check_energies.tolist(), miner_energies.tolist(), ""

    def get_ns_computed(self):
        """Calculate the number of nanoseconds computed by the miner."""

        return (self.cpt_step * self.system_config.time_step_size) / 1e3

    def save_pdb(self, output_path: str):
        """Save the pdb file to the output path."""
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        topology = self.simulation.topology
        with open(output_path, "w") as f:
            app.PDBFile.writeFile(topology, positions, f)

    def get_energy(self):
        state = self.simulation.context.getState(getEnergy=True)

        return state.getPotentialEnergy() / unit.kilojoules_per_mole

    def get_rmsd(self, output_path: str = None, xvg_command: str = "-xvg none"):
        """TODO: Implement the RMSD calculation"""
        return -1

    def _calculate_epsilon(self):
        # TODO: Make this a better relationship?
        return self.epsilon

    def extract(self, filepath: str, names=["step", "default-name"]):
        return pd.read_csv(filepath, sep="\s+", header=None, names=names)

    def remove_pdb_directory(self):
        """Method to remove the pdb directory after the simulation is complete.
        Temp. method before we know what we want to keep.
        """
        shutil.rmtree(self.pdb_directory)

    def calc_init_energy(self) -> float:
        """Calculate the potential energy from an edr file using gmx energy.
        Args:
            output_dir (str): directory containing the edr file
            edr_name (str): name of the edr file
            xvg_name (str): name of the xvg file

        Returns:
            float: potential energy
        """

        return (
            self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            / unit.kilojoules_per_mole
        )
