import time
import glob
import os
import pickle
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal
from log_event_funct import log_event

import bittensor as bt
import openmm as mm
import pandas as pd
from openmm import app, unit
from pdbfixer import PDBFixer
from folding.store import Job
from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.ops import (
    check_and_download_pdbs,
    check_if_directory_exists,
    load_pdb_ids,
    select_random_pdb_id,
)
from folding.store import Job
from folding.base.simulation import OpenMMSimulation

# root level directory for the project (I HATE THIS)
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Protein(OpenMMSimulation):
    PDB_RECORDS = ("ATOM", "ANISOU", "REMARK", "HETATM", "CONECT")

    @property
    def name(self):
        return self.protein_pdb.split(".")[0]

    def __init__(
        self,
        ff: str,
        box: Literal["cube", "dodecahedron", "octahedron"],
        config: Dict,
        pdb_id: str = None,
        water: str = None,
        load_md_inputs: bool = False,
        epsilon: float = 5e3,
        source: str = None,
    ) -> None:
        if source == "old":
            self.base_directory = os.path.join(str(ROOT_DIR), "data/old_source")

        if source == "cif":
            self.base_directory = os.path.join(str(ROOT_DIR), "data/cif_to_pdb")

        if source == "new-cif":
            self.base_directory = os.path.join(str(ROOT_DIR), "data/cifs_with_pbc")

        self.pdb_id: str = pdb_id.lower()
        self.setup_filepaths()

        self.ff: str = ff
        self.box: Literal["cube", "dodecahedron", "octahedron"] = box
        self.water: str = water

        self.system_config = SimulationConfig(
            ff=self.ff, water=self.water, box=self.box, seed=self.gen_seed()
        )

        self.config = config
        self.simulation: app.Simulation = None

        self.simulation_pkl = f"config_{self.pdb_id}.pkl"
        self.simulation_cpt = "em.cpt"
        self.input_files = [self.simulation_cpt]

        self.md_inputs = (
            self.read_and_return_files(filenames=self.input_files)
            if load_md_inputs
            else {}
        )
        self.source = source

        # # Historic data that specifies the upper bounds of the energy as a function of steps.
        # with open(
        #     os.path.join(self.base_directory, "upper_bounds_interpolated.pkl"), "rb"
        # ) as f:
        #     self.upper_bounds : List = pickle.load(f)

        self.upper_bounds = [0, 1, 2, 3]

        # set to an arbitrarily high number to ensure that the first miner is always accepted.
        self.init_energy = 0
        self.pdb_complexity = defaultdict(int)
        self.epsilon = epsilon

    def setup_filepaths(self):
        self.pdb_file = f"{self.pdb_id}.pdb"
        self.pdb_directory = os.path.join(self.base_directory, self.pdb_id)
        self.pdb_location = os.path.join(self.pdb_directory, self.pdb_file)
        self.cif_file = f"{self.pdb_id}.cif.gz"
        self.cif_directory = os.path.join(
            self.base_directory, self.pdb_id
        )  # use the pdb_id to name the directory
        self.cif_location = os.path.join(self.cif_directory, self.cif_file)

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
            config=config,
            load_md_inputs=True,
            epsilon=job.epsilon,
        )

        try:
            protein.pdb_complexity = Protein._get_pdb_complexity(protein.pdb_location)
            protein.pdb_obj = protein.load_pdb_file(pdb_file=protein.pdb_location)
            protein.simulation = protein.create_simulation(
                pdb=protein.pdb_obj,
                system_config=protein.system_config,
                seed=protein.system_config.seed,
                state="em",
            )
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
                root_dir=ROOT_DIR, filename="cif_ids_new_parent_dir.pkl"
            )  # TODO: This should be a class variable via config
            self.pdb_id = select_random_pdb_id(PDB_IDS=PDB_IDS)
            bt.logging.debug(f"Selected random pdb id: {self.pdb_id!r}")

        self.pdb_file_tmp = f"{self.pdb_id}_protein_tmp.pdb"
        self.pdb_file_cleaned = f"{self.pdb_id}_protein.pdb"

    def setup_pdb_directory(self, source: str):
        # if directory doesn't exist, download the pdb file and save it to the directory
        # if not os.path.exists(self.cif_file):
        #     raise FileNotFoundError(f"{self.cif_file} does not exist")
        self.source = source

        if self.source == "new-cif":
            if not os.path.exists(self.cif_directory):
                os.makedirs(self.cif_directory)
                check_and_download_pdbs(
                    pdb_directory=self.cif_directory,
                    pdb_id=self.pdb_id,
                    download=False,
                    # force=True,
                    source=self.source,
                )

            self.convert_cif_to_pdb(cif_file=self.cif_location, pdb_file=self.pdb_file)
            pdb_fixer_dict = self.fix_pdb_file(filename=self.pdb_location)
            log_event(pdb_fixer_dict)

        # if self.source == "old":
        #     if not os.path.exists(self.pdb_location):
        #         if not check_and_download_pdbs(
        #             pdb_directory=self.pdb_directory,
        #             pdb_id=self.pdb_id,
        #             download=True,
        #             # force=True,
        #             source=self.source,
        #         ):
        #             raise Exception(
        #                 f"Failed to download {self.pdb_file} to {self.pdb_directory}"
        #             )

        #     self.convert_cif_to_pdb(cif_file=self.cif_location, pdb_file=self.pdb_file) # convert cif to pdb if the source is cif

        #     pdb_fixer_dict = self.fix_pdb_file()
        #     log_event(pdb_fixer_dict)

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
                    # A bit of a hack to load in the data correctly depending on the file ext
                    name = f.split("/")[-1]
                    if name.split(".")[-1] == "cpt":
                        readmode = "rb"
                    else:
                        readmode = "r"

                    files_to_return[name] = open(f, readmode).read()
                except Exception as E:
                    continue
        return files_to_return

    def setup_simulation(self, source: str):
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
        self.setup_pdb_directory(source=source)

        missing_files = self.check_for_missing_files(required_files=self.input_files)

        if missing_files is not None:
            self.generate_input_files()

        # Create a validator directory to store the files
        check_if_directory_exists(output_directory=self.validator_directory)

        # Read the files that should exist now based on generate_input_files.
        self.md_inputs = self.read_and_return_files(filenames=self.input_files)

        self.save_files(
            files=self.md_inputs,
            output_directory=self.validator_directory,
            write_mode="wb",
        )

        self.pdb_complexity = Protein._get_pdb_complexity(self.pdb_location)
        self.init_energy = self.calc_init_energy()
        self._calculate_epsilon()

    def __str__(self):
        return f"Protein(pdb_id={self.pdb_id}, ff={self.ff}, box={self.box}, water={self.water})"

    def __repr__(self):
        return self.__str__()

    def load_pdb_file(self, pdb_file: str) -> app.PDBFile:
        """Method to take in the pdb file and load it into an OpenMM PDBFile object."""
        return app.PDBFile(pdb_file)

    def fix_pdb_file(self, filename):
        """
        Protein Data Bank (PDB or PDBx/mmCIF) files often have a number of problems
        that must be fixed before they can be used in a molecular dynamics simulation.

        Reference docs for the PDBFixer class can be found here:
            https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
        """
        start_time = time.time()

        event_dict = {}
        fixer = PDBFixer(filename=filename)
        event_dict.update({"pdb_fixer_time": time.time() - start_time})

        start_time = time.time()
        fixer.findMissingResidues()
        event_dict.update({"missing_residues_time": time.time() - start_time})

        start_time = time.time()
        fixer.findNonstandardResidues()
        event_dict.update({"nonstandard_residues_time": time.time() - start_time})

        start_time = time.time()
        fixer.replaceNonstandardResidues()
        event_dict.update(
            {"replace_nonstandard_residues_time": time.time() - start_time}
        )
        start_time = time.time()
        fixer.removeHeterogens(True)
        event_dict.update({"remove_heterogens_time": time.time() - start_time})

        start_time = time.time()
        fixer.findMissingAtoms()
        event_dict.update({"missing_atoms_time": time.time() - start_time})

        start_time = time.time()
        fixer.addMissingAtoms()
        event_dict.update({"add_missing_atoms_time": time.time() - start_time})

        start_time = time.time()
        fixer.addMissingHydrogens(pH=7.0)
        event_dict.update({"add_missing_hydrogens_time": time.time() - start_time})

        original_pdb = self.pdb_location.split(".")[0] + "_original.pdb"
        os.rename(self.pdb_location, original_pdb)
        app.PDBFile.writeFile(
            topology=fixer.topology,
            positions=fixer.positions,
            file=open(self.pdb_location, "w"),
        )

        return event_dict

    def convert_cif_to_pdb(self, cif_file: str, pdb_file: str):
        """Convert a CIF file to a PDB file using the `parmed` library."""
        import parmed as pmd

        if not os.path.exists(self.pdb_location):
            try:
                # Load the structure from the CIF file
                structure = pmd.load_file(self.cif_location)
                # Write the structure to a PDB file
                structure.write_pdb(
                    os.path.join(self.cif_directory, f"{self.pdb_id}.pdb")
                )
            except Exception as e:
                print(f"Failed to convert {cif_file} to PDB format. Error: {e}")

    # Function to generate the OpenMM simulation state.
    @OpenMMSimulation.timeit
    def generate_input_files(self):
        bt.logging.info(f"Changing path to {self.pdb_directory}")
        os.chdir(self.pdb_directory)

        bt.logging.info(
            f"pdb file is set to: {self.pdb_file}, and it is located at {self.pdb_location}"
        )
        self.simulation = self.create_simulation(
            pdb=self.load_pdb_file(pdb_file=self.pdb_file),
            system_config=self.system_config.get_config(),
            state="em",
        )

        bt.logging.info(f"Minimizing energy for pdb: {self.pdb_id} ...")

        start_time = time.time()
        self.simulation.minimizeEnergy(
            maxIterations=1000
        )  # TODO: figure out the right number for this
        bt.logging.warning(f"Minimization took {time.time() - start_time:.4f} seconds")

        self.simulation.saveCheckpoint("em.cpt")

        # This is only for the validators, as they need to open the right config later.
        # Only save the config if the simulation was successful.
        with open(self.simulation_pkl, "wb") as f:
            pickle.dump(self.system_config, f)

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
            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.
            self.simulation = self.create_simulation(
                pdb=self.load_pdb_file(pdb_file=self.pdb_file),
                system_config=self.system_config,
                seed=seed,
                state=state,
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
