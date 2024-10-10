import os
import time
import glob
import base64
import random
import shutil
from pathlib import Path

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
import plotly.express as px

from openmm import app, unit
from pdbfixer import PDBFixer

from folding.store import Job
from folding.base.simulation import OpenMMSimulation
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.ops import (
    OpenMMException,
    ValidationError,
    check_and_download_pdbs,
    check_if_directory_exists,
    write_pkl,
    load_and_sample_random_pdb_ids,
)

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
        epsilon: float = 0.01,
    ) -> None:
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

    @staticmethod
    def from_job(job: Job, config: Dict):
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
            bt.logging.error(
                f"from_job failed for {protein.pdb_id} with Exception {E}."
            )
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

    @staticmethod
    def save_pdb(output_path: str, simulation: app.Simulation):
        """Save the pdb file to the output path."""
        positions = simulation.context.getState(getPositions=True).getPositions()
        topology = simulation.topology
        with open(output_path, "w") as f:
            app.PDBFile.writeFile(topology, positions, f)

    def gather_pdb_id(self):
        if self.pdb_id is None:
            self.pdb_id = load_and_sample_random_pdb_ids(
                root_dir=ROOT_DIR, filename="pdb_ids.pkl"
            )  # TODO: This should be a class variable via config
            bt.logging.debug(f"Selected random pdb id: {self.pdb_id!r}")

    def setup_pdb_directory(self):
        # if directory doesn't exist, download the pdb file and save it to the directory
        if not os.path.exists(self.pdb_directory):
            os.makedirs(self.pdb_directory)

        if not os.path.exists(os.path.join(self.pdb_directory, self.pdb_file)):
            if not check_and_download_pdbs(
                pdb_directory=self.pdb_directory,
                pdb_id=self.pdb_file,
                input_source=self.config.input_source,
                download=True,
                force=self.config.force_use_pdb,
            ):
                raise Exception(
                    f"Failed to download {self.pdb_file} to {self.pdb_directory}"
                )
            self.fix_pdb_file()
        else:
            bt.logging.info(
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

    def setup_simulation(self):
        """forward method defines the following:
        1. gather the pdb_id and setup the namings.
        2. setup the pdb directory and download the pdb file if it doesn't exist.
        3. check for missing files and generate the input files if they are missing.
        4. edit the necessary config files and add them to the synapse object self.md_inputs[file] = content
        4. save the files to the validator directory for record keeping.
        """
        bt.logging.info(
            f"Launching {self.pdb_id} Protein Job with the following configuration\nff : {self.ff}\nbox : {self.box}\nwater : {self.water}"
        )

        ## Setup the protein directory and sample a random pdb_id if not provided
        self.gather_pdb_id()
        self.setup_pdb_directory()

        self.generate_input_files()

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

    def fix_pdb_file(self):
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

    # Function to generate the OpenMM simulation state.
    @OpenMMSimulation.timeit
    def generate_input_files(self):
        """Generate_input_files method defines the following:
        1. Load the pdb file and create the simulation object.
        2. Minimize the energy of the system.
        3. Save the checkpoint file.
        4. Save the system config to the validator directory.
        5. Move all files except the pdb file to the validator directory.
        """
        bt.logging.info(f"Changing path to {self.pdb_directory}")
        os.chdir(self.pdb_directory)

        bt.logging.info(
            f"pdb file is set to: {self.pdb_file}, and it is located at {self.pdb_location}"
        )

        self.simulation, self.system_config = self.create_simulation(
            pdb=self.load_pdb_file(pdb_file=self.pdb_file),
            system_config=self.system_config.get_config(),
            state="em",
        )

        bt.logging.info(f"Minimizing energy for pdb: {self.pdb_id} ...")

        start_time = time.time()
        self.simulation.minimizeEnergy(
            maxIterations=100
        )  # TODO: figure out the right number for this
        bt.logging.warning(f"Minimization took {time.time() - start_time:.4f} seconds")
        self.simulation.step(1000)

        self.simulation.saveCheckpoint("em.cpt")

        # This is only for the validators, as they need to open the right config later.
        # Only save the config if the simulation was successful.
        write_pkl(data=self.system_config, path=self.simulation_pkl, write_mode="wb")

        # Here we are going to change the path to a validator folder, and move ALL the files except the pdb file
        check_if_directory_exists(output_directory=self.validator_directory)
        # Move all files
        cmd = f'find . -maxdepth 1 -type f ! -name "*.pdb" -exec mv {{}} {self.validator_directory}/ \;'
        bt.logging.debug(f"Moving all files except pdb to {self.validator_directory}")
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
        bt.logging.info(f"⏰ Saving files to {output_directory}...")
        check_if_directory_exists(output_directory=output_directory)

        filetypes = {}
        for filename, content in files.items():
            filetypes[filename.split(".")[-1]] = filename

            bt.logging.info(f"Saving file {filename} to {output_directory}")
            if "em.cpt" in filename:
                filename = "em_binary.cpt"

            # loop over all of the output files and save to local disk
            with open(os.path.join(output_directory, filename), write_mode) as f:
                f.write(content)

        return filetypes

    def delete_files(self, directory: str):
        bt.logging.info(f"Deleting files in {directory}")
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
        # os.rmdir(output_directory)

    def get_miner_data_directory(self, hotkey: str):
        return os.path.join(self.validator_directory, hotkey[:8])

    def process_md_output(
        self, md_output: dict, seed: int, state: str, hotkey: str
    ) -> Tuple[bool, Dict]:
        MIN_LOGGING_ENTRIES = 500
        MIN_SIMULATION_STEPS = 5000

        required_files_extensions = ["cpt", "log"]
        hotkey_alias = hotkey[:8]

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {
            k.split(".")[-1]: k for k, v in md_output.items() if len(v) > 0
        }

        if len(md_output.keys()) == 0:
            bt.logging.warning(
                f"Miner {hotkey_alias} returned empty md_output... Skipping!"
            )
            return False, None

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                bt.logging.error(f"Missing file with extension {ext} in md_output")
                return False, None

        miner_data_directory = self.get_miner_data_directory(hotkey=hotkey)

        # Save files so we can check the hash later.
        self.save_files(
            files=md_output,
            output_directory=miner_data_directory,
        )

        try:
            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.

            bt.logging.info(
                f"Recreating miner {hotkey_alias} simulation in state: {state}"
            )
            simulation, local_system_config = self.create_simulation(
                pdb=self.load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=seed,
                state=state,
            )

            checkpoint_path = os.path.join(miner_data_directory, f"{state}.cpt")

            log_file_path = os.path.join(
                miner_data_directory, self.md_outputs_exts["log"]
            )

            simulation.loadCheckpoint(checkpoint_path)
            log_file = pd.read_csv(log_file_path)
            log_step = log_file['#"Step"'].iloc[-1]

            # Checks to see if we have enough steps in the log file to start validation
            if len(log_file) < MIN_LOGGING_ENTRIES:
                raise ValidationError(
                    f"Miner {hotkey_alias} did not run enough steps in the simulation... Skipping!"
                )

            # Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
            # Checks if log_file is MIN_STEPS steps ahead of checkpoint
            if (log_step - simulation.currentStep) < MIN_SIMULATION_STEPS:
                # If the miner did not run enough steps, we will load the old checkpoint
                checkpoint_path = os.path.join(miner_data_directory, f"{state}_old.cpt")
                if os.path.exists(checkpoint_path):
                    bt.logging.warning(
                        f"Miner {hotkey_alias} did not run enough steps since last checkpoint... Loading old checkpoint"
                    )
                    simulation.loadCheckpoint(checkpoint_path)
                    # Checking to see if the old checkpoint has enough steps to validate
                    if (log_step - simulation.currentStep) < MIN_SIMULATION_STEPS:
                        raise ValidationError(
                            f"Miner {hotkey_alias} did not run enough steps in the simulation... Skipping!"
                        )
                else:
                    raise ValidationError(
                        f"Miner {hotkey_alias} did not run enough steps and no old checkpoint found... Skipping!"
                    )

            # Save the system config to the miner data directory
            system_config_path = os.path.join(
                miner_data_directory, f"miner_system_config_{seed}.pkl"
            )
            if not os.path.exists(system_config_path):
                write_pkl(
                    data=local_system_config,
                    path=system_config_path,
                    write_mode="wb",
                )

        except ValidationError as E:
            bt.logging.warning(f"{E}")
            return False, None

        except Exception as e:
            bt.logging.error(f"Failed to recreate simulation: {e}")
            return False, None

        return True, {
            "simulation": simulation,
            "log_file": log_file,
            "log_step": log_step,
        }

    def is_run_valid(
        self,
        simulation: app.Simulation,
        state: str,
        hotkey: str,
        log_file: pd.DataFrame,
        log_step: int,
    ) -> Tuple[bool, list, list]:
        """
        Checks if the run is valid by comparing the potential energy values
        between the current simulation and a reference log file.

        Returns:
            Tuple[bool, list, list]: True if the run is valid, False otherwise.
                The two lists contain the potential energy values from the current simulation and the reference log file.
        """

        # The percentage that we allow the energy to differ from the miner to the validator.
        ANOMALY_THRESHOLD = 0.5
        miner_data_directory = self.get_miner_data_directory(hotkey=hotkey)

        # Check to see if we have a logging resolution of 10 or better, if not the run is not valid
        if (log_file['#"Step"'][1] - log_file['#"Step"'][0]) > 10:
            return False, [], []

        # Run the simulation at most 3000 steps
        steps_to_run = min(3000, log_step - self.simulation.currentStep)

        simulation.reporters.append(
            app.StateDataReporter(
                os.path.join(miner_data_directory, f"check_{state}.log"),
                10,
                step=True,
                potentialEnergy=True,
            )
        )

        bt.logging.info(
            f"Running {steps_to_run} steps. log_step: {log_step}, cpt_step: {simulation.currentStep}"
        )

        simulation.step(steps_to_run)

        check_log_file = pd.read_csv(
            os.path.join(miner_data_directory, f"check_{state}.log")
        )

        max_step = simulation.currentStep + steps_to_run

        check_energies: np.ndarray = check_log_file["Potential Energy (kJ/mole)"].values
        miner_energies: np.ndarray = log_file[
            (log_file['#"Step"'] > simulation.currentStep)
            & (log_file['#"Step"'] <= max_step)
        ]["Potential Energy (kJ/mole)"].values

        # calculating absolute percent difference per step
        percent_diff = abs(((check_energies - miner_energies) / miner_energies) * 100)

        # This is some debugging information for plotting the information from the miner.
        df = pd.DataFrame([check_energies, miner_energies]).T
        df.columns = ["validator", "miner"]

        fig = px.scatter(
            df,
            title=f"Energy: {self.pdb_id} for state {state} starting at checkpoint step: {simulation.currentStep}",
            labels={"index": "Step", "value": "Energy (kJ/mole)"},
            height=600,
            width=1400,
        )
        filename = f"{self.pdb_id}_cpt_step_{simulation.currentStep}_state_{state}"
        fig.write_image(os.path.join(miner_data_directory, filename + "_energy.png"))

        fig = px.scatter(
            percent_diff,
            title=f"Percent Diff: {self.pdb_id} for state {state} starting at checkpoint step: {simulation.currentStep}",
            labels={"index": "Step", "value": "Percent Diff"},
            height=600,
            width=1400,
        )
        fig.write_image(
            os.path.join(miner_data_directory, filename + "_percent_diff.png")
        )

        median_percent_diff = np.median(percent_diff)

        # We want to save all the information to the local filesystem so we can index them later.

        if median_percent_diff > ANOMALY_THRESHOLD:
            return False, check_energies.tolist(), miner_energies.tolist()

        Protein.save_pdb(
            output_path=os.path.join(miner_data_directory, f"{self.pdb_id}_folded.pdb")
        )

        return True, check_energies.tolist(), miner_energies.tolist()

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
