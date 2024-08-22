import os
import glob
import re
import random
import shutil
from enum import Enum
from typing import List, Dict, Literal, Optional
from pathlib import Path
from collections import defaultdict
import pickle

import bittensor as bt
import pandas as pd
from dataclasses import dataclass
from pydantic import BaseModel
from folding.utils.new_config import SimulationConfig

import openmm as mm
from openmm import app
from openmm import unit

from folding.utils.ops import (
    FF_WATER_PAIRS,
    run_cmd_commands,
    check_if_directory_exists,
    gro_hash,
    load_pdb_ids,
    calc_potential_from_edr,
    select_random_pdb_id,
    check_and_download_pdbs,
    get_last_step_time,
)
from folding.store import Job

# root level directory for the project (I HATE THIS)
ROOT_DIR = Path(__file__).resolve().parents[2]

@dataclass
class Protein:
    PDB_RECORDS = ("ATOM", "ANISOU", "REMARK", "HETATM", "CONECT")

    @property
    def name(self):
        return self.protein_pdb.split(".")[0]

    def __init__(
        self,
        ff: str,
        box: str,
        config: Dict,
        pdb_id: str = None,
        water: str = None,
        load_md_inputs: bool = False,
        epsilon: float = 5e3,
    ):
        self.base_directory = os.path.join(str(ROOT_DIR), "data")

        self.pdb_id = pdb_id.lower()
        self.setup_filepaths()

        self.ff = ff
        self.box = box
        self.water = water

        self.system_config = SimulationConfig(
            ff=self.ff, water=self.water, box=self.box
        )

        self.config = config

        self.simulation: app.Simulation = None

        self.input_files = ["config.json", "em.cpt"]

        self.md_inputs = (
            self.read_and_return_files(filenames=self.input_files)
            if load_md_inputs
            else {}
        )
        with open(
            os.path.join(self.base_directory, "upper_bounds_interpolated.pkl"), "rb"
        ) as f:
            self.upper_bounds = pickle.load(f)

        # set to an arbitrarily high number to ensure that the first miner is always accepted.
        self.init_energy = 0
        self.pdb_complexity = defaultdict(int)
        self.epsilon = epsilon

    def setup_filepaths(self):
        self.pdb_file = f"{self.pdb_id}.pdb"
        self.pdb_directory = os.path.join(self.base_directory, self.pdb_id)
        self.pdb_location = os.path.join(self.pdb_directory, self.pdb_file)

        self.validator_directory = os.path.join(self.pdb_directory, "validator")

    @staticmethod
    def from_job(job: Job, config: Dict):
        # TODO: This must be called after the protein has already been downloaded etc.
        bt.logging.warning(f"sampling pdb job {job.pdb}")
        # Load_md_inputs is set to True to ensure that miners get files every query.
        protein = Protein(
            pdb_id=job.pdb,
            ff=job.ff,
            box=job.box,
            water=job.water,
            config=SimulationConfig(**config), 
            load_md_inputs=True,
            epsilon=job.epsilon,
        )

        try:
            protein.pdb_complexity = Protein._get_pdb_complexity(protein.pdb_location)
            protein.create_simulation(protein.gen_seed(), "em")
            protein.init_energy = protein.calc_init_energy()
            protein._calculate_epsilon()
        except Exception as E:
            bt.logging.error(
                f"pdb_complexity or init_energy failed for {protein.pdb_id} with Exception {E}."
            )
        finally:
            return protein

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

    def gather_pdb_id(self):
        if self.pdb_id is None:
            PDB_IDS = load_pdb_ids(
                root_dir=ROOT_DIR, filename="pdb_ids.pkl"
            )  # TODO: This should be a class variable via config
            self.pdb_id = select_random_pdb_id(PDB_IDS=PDB_IDS)
            bt.logging.debug(f"Selected random pdb id: {self.pdb_id!r}")

        self.pdb_file_tmp = f"{self.pdb_id}_protein_tmp.pdb"
        self.pdb_file_cleaned = f"{self.pdb_id}_protein.pdb"

    def setup_pdb_directory(self):
        # if directory doesn't exist, download the pdb file and save it to the directory
        if not os.path.exists(self.pdb_directory):
            os.makedirs(self.pdb_directory)

        if not os.path.exists(os.path.join(self.pdb_directory, self.pdb_file)):
            if not check_and_download_pdbs(
                pdb_directory=self.pdb_directory,
                pdb_id=self.pdb_file,
                download=True,
                force=self.config.force_use_pdb,
            ):
                raise Exception(
                    f"Failed to download {self.pdb_file} to {self.pdb_directory}"
                )

        else:
            bt.logging.info(
                f"PDB file {self.pdb_file} already exists in path {self.pdb_directory!r}."
            )

    def check_for_missing_files(self, required_files: List[str]):
        missing_files = [
            filename
            for filename in required_files
            if not os.path.exists(os.path.join(self.pdb_directory, filename))
        ]

        if len(missing_files) > 0:
            return missing_files
        return None

    def read_and_return_files(self, filenames: List) -> Dict:
        """Read the files and return them as a dictionary."""
        files_to_return = {}
        for file in filenames:
            for f in glob.glob(os.path.join(self.validator_directory, file)):
                try:
                    files_to_return[f.split("/")[-1]] = open(f, "r").read()
                except Exception as E:
                    # bt.logging.warning(
                    #     f"Attempted to put file {file} in md_inputs.\nError: {E}"
                    # )
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

        # TODO: Enable this to send checkpoints rather than only the initial set of files

        missing_files = self.check_for_missing_files(required_files=self.input_files)

        if missing_files is not None:
            self.generate_input_files()

        # Create a validator directory to store the files
        check_if_directory_exists(output_directory=self.validator_directory)

        # Read the files that should exist now based on generate_input_files.
        self.md_inputs = self.read_and_return_files(filenames=self.other_files)

        self.save_files(
            files=self.md_inputs,
            output_directory=self.validator_directory,
            write_mode="w",
        )

        self.pdb_complexity = Protein._get_pdb_complexity(self.pdb_location)
        self.init_energy = self.calc_init_energy()
        self._calculate_epsilon()

    def __str__(self):
        return f"Protein(pdb_id={self.pdb_id}, ff={self.ff}, box={self.box}, water={self.water})"

    def __repr__(self):
        return self.__str__()

    def calculate_params_save_interval(self):
        # TODO Define what this function should do. Placeholder for now.
        return self.config.save_interval

    def check_configuration_file_commands(self) -> List[str]:
        """
        There are a set of configuration files that are used to setup the simulation
        environment. These are typically denoted by the force field name. If they do
        not exist, then we default to the same names without the force field included.
        """
        # strip away trailing number in forcefield name e.g charmm27 -> charmm, and
        ff_base = "".join([c for c in self.ff if not c.isdigit()])

        filepaths = [
            f"{self.base_directory}/nvt_{ff_base}.mdp",
            f"{self.base_directory}/npt_{ff_base}.mdp",
            f"{self.base_directory}/md_{ff_base}.mdp",
            f"{self.base_directory}/emin_{ff_base}.mdp",
        ]

        commands = []

        for file_path in filepaths:
            match = re.search(
                r"/([^/]+)_" + re.escape(ff_base) + r"\.mdp", file_path
            )  # extract the nvt, npt...
            config_name = match.group(1)

            if os.path.exists(file_path):
                commands.append(
                    f"cp {file_path} {self.pdb_directory}/{config_name}.mdp"
                )
            else:
                # If the file with the _ff suffix doesn't exist, remove it from the base filepath.
                path = re.sub(r"_" + re.escape(ff_base), "", file_path)
                commands.append(f"cp {path} {self.pdb_directory}/{config_name}.mdp")

        return commands

    # Function to generate GROMACS input files
    def generate_input_files(self):
        bt.logging.info(f"Changing path to {self.pdb_directory}")
        os.chdir(self.pdb_directory)

        bt.logging.info(
            f"pdb file is set to: {self.pdb_file}, and it is located at {self.pdb_location}"
        )
        with open("config.json", "w") as f:
            f.write(self.system_config.json())

        self.simulation = self.create_simulation(self.gen_seed(), "em")
        self.simulation.minimizeEnergy(
            maxIterations=1000
        )  # TODO: figure out the right number for this
        self.simulation.saveCheckpoint("em.cpt")

        # Here we are going to change the path to a validator folder, and move ALL the files except the pdb file
        check_if_directory_exists(output_directory=self.validator_directory)
        # Move all files
        cmd = f'find . -maxdepth 1 -type f ! -name "*.pdb" -exec mv {{}} {self.validator_directory}/ \;'
        bt.logging.debug(f"Moving all files except pdb to {self.validator_directory}")
        os.system(cmd)

        # We want to catch any errors that occur in the above steps and then return the error to the user
        return True

    def gen_seed(self):
        """Generate a random seed"""
        return random.randint(1000, 999999)

    def edit_files(self, mdp_files: List, params_to_change: List, seed: int = None):
        """Edit the files that are needed for simulation and attach them to md_inputs.

        Args:
            mdp_files (List): Files that contain parameters to change.
            params_to_change (List): List of parameters to change in the desired file(s)
            seed (int): Seed to use for the simulation. Defaults to None.
        """

        seed = self.gen_seed() if seed is None else seed

        def mapper(content: str, param_name: str, value: int) -> str:
            """change the parameter value to the desired value in the content of the file."""
            content = re.sub(
                f"{param_name}\\s+=\\s+-?\\d+", f"{param_name} = {value}", content
            )
            return content

        for file in mdp_files:
            filepath = os.path.join(self.validator_directory, file)
            content = open(filepath, "r").read()

            # Set the value of nsteps based on the config, orders of magnitude smaller than max_steps.
            # Set the value of gen_seed, ld-seed based on the current instance.
            if file == "emin.mdp":
                params_values = {"ld-seed": seed}
            elif file == "nvt.mdp":
                params_values = {
                    "nsteps": (
                        self.config.nvt_steps
                        if self.config.nvt_steps is not None
                        else self.config.max_steps // 100
                    ),
                    "gen_seed": seed,
                    "ld-seed": seed,
                }
            elif file == "npt.mdp":
                params_values = {
                    "nsteps": (
                        self.config.npt_steps
                        if self.config.npt_steps is not None
                        else self.config.max_steps // 10
                    ),
                    "ld-seed": seed,
                }
            elif file == "md.mdp":
                params_values = {"nsteps": self.config.max_steps, "ld-seed": seed}

            # Specific implementation for each file
            for param_name, value in params_values.items():
                content = mapper(content=content, param_name=param_name, value=value)

            save_interval = self.calculate_params_save_interval()
            for param in params_to_change:
                if param in content:
                    bt.logging.debug(
                        f"Changing {param} in {file} to {save_interval}..."
                    )
                    content = mapper(
                        content=content, param_name=param, value=save_interval
                    )

            self.md_inputs[file] = content

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
        self.miner_data_directory = os.path.join(self.validator_directory, hotkey[:8])

    def compute_intermediate_gro(
        self,
        output_directory: str,
        md_outputs_exts: Dict,  # mapping from file extension to filename in md_output
        simulation_step_time: float,  # A step (frame) of the simulation that you want to compute the gro file on.
    ) -> str:
        """
        Compute the intermediate gro file from the xtc and tpr file from the miner.
        We do this because we need to ensure that the miners are running the correct protein.

        Args:
            md_output (Dict): dictionary of information from the miner.
        """

        # Make times to 1 decimal place for gromacs stability.
        simulation_step_time = round(simulation_step_time, 1)

        gro_file_location = os.path.join(output_directory, "intermediate.gro")
        tpr_file = os.path.join(output_directory, md_outputs_exts["tpr"])
        xtc_file = os.path.join(output_directory, md_outputs_exts["xtc"])

        command = [
            f"echo System | gmx trjconv -s {tpr_file} -f {xtc_file} -o {gro_file_location} -nobackup -b {simulation_step_time} -e {simulation_step_time}"
        ]

        bt.logging.warning(f"Computing an intermediate gro...")
        run_cmd_commands(
            commands=command,
            suppress_cmd_output=self.config.suppress_cmd_output,
            verbose=self.config.verbose,
        )

        return gro_file_location

    def rerun(self, output_directory: str, gro_file_location: str):
        """Rerun method is to rerun a step of a miner simulation to ensure that
        the miner is running the correct protein.

        Args:
            output_directory: The directory where the files are stored and where the rerun files will be written to.
                output location will be the miner directory held on the validator side, from get_miner_data_directory
            gro_file_name: The name of the .gro file that will be rerun. This is calculated by the *validator* beforehand.
        """
        rerun_mdp = os.path.join(
            self.base_directory, "rerun.mdp"
        )  # rerun file is a base config that we never change.
        topol_path = os.path.join(
            self.validator_directory, "topol.top"
        )  # all miners get the same topol file.
        tpr_path = os.path.join(output_directory, "rerun.tpr")

        commands = [
            f"gmx grompp -f {rerun_mdp} -c {gro_file_location} -p {topol_path} -o {tpr_path}",
            f"gmx mdrun -s {tpr_path} -rerun {gro_file_location} -deffnm {output_directory}/rerun_energy",  # -s specifies the file.
        ]
        run_cmd_commands(
            commands=commands,
            suppress_cmd_output=self.config.suppress_cmd_output,
            verbose=self.config.verbose,
        )

    def process_md_output(
        self, md_output: dict, seed: int, state: str, hotkey: str
    ) -> bool:
        """
        1. Check md_output for the required files, if unsuccessful return False
        2. Save files if above is valid
        3. Check the hash of the .gro file to ensure miners are running the correct protein.
        4.
        """

        required_files_extensions = ["cpt", "log"]

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {
            k.split(".")[-1]: k for k, v in md_output.items() if len(v) > 0
        }

        if len(md_output.keys()) == 0:
            bt.logging.warning(
                f"Miner {hotkey[:8]} returned empty md_output... Skipping!"
            )
            return False

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                bt.logging.error(f"Missing file with extension {ext} in md_output")
                return False

        self.get_miner_data_directory(hotkey=hotkey)

        # Save files so we can check the hash later.
        self.save_files(
            files=md_output,
            output_directory=self.miner_data_directory,
        )
        try:
            self.simulation = self.create_simulation(
                seed,
                state,
            )
            self.simulation.loadCheckpoint(
                f"{self.miner_data_directory}/{self.md_outputs_exts['cpt']}"
            )
        except Exception as e:
            bt.logging.error(f"Failed to recreate simulation: {e}")
            return False

        cpt_step = self.simulation.currentStep
        log_file = pd.read_csv(
            f"{self.miner_data_directory}/{self.md_outputs_exts['log']}"
        )
        log_step = log_file['#"Step"'].iloc[-1]

        ## Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
        if (log_step - cpt_step) < 5000:
            bt.logging.error("Miner did not run enough steps since last checkpoint")
            return False

        return True

    def create_simulation(self, seed: str, state: str) -> app.Simulation:
        """Recreates a simulation object based on the provided parameters.

        This method takes in a seed, state, and checkpoint file path to recreate a simulation object.
        Args:
            seed (str): The seed for the random number generator.
            state (str): The state of the simulation.
            cpt_file (str): The path to the checkpoint file.

        Returns:
            app.Simulation: The recreated simulation object.
        """
        pdb = app.PDBFile(self.pdb_file)
        forcefield = app.ForceField(self.system_config.ff, self.system_config.water)

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.deleteWater()

        modeller.addSolvent(
            forcefield,
            padding=self.system_config.box_padding * unit.nanometer,
            boxShape=self.system_config.box,
            model=self.system_config.water,
        )

        # Create the system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=self.system_config.nonbonded_method,
            nonbondedCutoff=self.system_config.cutoff * mm.unit.nanometers,
            constraints=self.system_config.constraints,
        )

        # Integrator settings
        integrator = mm.LangevinIntegrator(
            self.system_config.temperature * unit.kelvin,
            self.system_config.friction / unit.picosecond,
            self.system_config.time_step_size * unit.picoseconds,
        )
        integrator.setRandomNumberSeed(seed)
        # Periodic boundary conditions
        pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

        if state != "nvt":
            system.addForce(
                mm.MonteCarloBarostat(
                    self.system_config.pressure * unit.bar,
                    self.system_config.temperature * unit.kelvin,
                )
            )
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"DeterministicForces": "true", "Precision": "double"}
        simulation = mm.app.Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        # Create the simulation object
        return simulation

    def is_run_valid(self):
        """
        Checks if the run is valid by comparing the potential energy values
        between the current simulation and a reference log file.

        Returns:
            bool: True if the run is valid, False otherwise.
        """
        log_file = pd.read_csv(
            f"{self.miner_data_directory}/{self.md_outputs_exts['log']}"
        )
        log_step = log_file['#"Step"'].iloc[-1]
        cpt_step = self.simulation.currentStep

        steps_to_run = log_step - cpt_step

        self.simulation.reporters.append(
            app.StateDataReporter(
                f"{self.miner_data_directory}/check.log",
                10,
                step=True,
                potentialEnergy=True,
                temperature=True,
            )
        )
        self.simulation.step(steps_to_run)

        check_log_file = pd.read_csv(f"{self.miner_data_directory}/check.log")

        check_energies = check_log_file["Potential Energy (kJ/mole)"][cpt_step:]
        miner_energies = log_file["Potential Energy (kJ/mole)"][cpt_step:]

        # calculating absolute percent difference per step
        percent_diff = abs((check_energies - miner_energies) / miner_energies * 100)
        min_length = min(len(percent_diff), len(self.upper_bounds))

        # Compare the entries up to the length of the shorter array
        comparison_result = percent_diff[:min_length] > self.upper_bounds[:min_length]

        # Calculate the percentage of True values
        percent_true = (sum(comparison_result) / len(comparison_result)) * 100

        if percent_true > 20:
            return False
        return True

    def get_energy(self):
        state = self.simulation.context.getState(getEnergy=True)

        return state.getPotentialEnergy() / unit.kilojoules_per_mole

    def get_rmsd(self, output_path: str = None, xvg_command: str = "-xvg none"):
        """TODO: Implement the RMSD calculation"""
        return -1

    def _calculate_epsilon(self):
        if "ATOM" in self.pdb_complexity.keys():
            num_atoms = self.pdb_complexity["ATOM"]

            if num_atoms > 100:
                self.epsilon = 7.14819473 * num_atoms + 1.68442317e04

    def extract(self, filepath: str, names=["step", "default-name"]):
        return pd.read_csv(filepath, sep="\s+", header=None, names=names)

    def remove_pdb_directory(self):
        """Method to remove the pdb directory after the simulation is complete.
        Temp. method before we know what we want to keep.
        """
        shutil.rmtree(self.pdb_directory)

    def calc_init_energy(self):
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
