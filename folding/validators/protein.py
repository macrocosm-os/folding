import os
import re
import hashlib
import requests
from typing import List, Dict
from pathlib import Path

import bittensor as bt
from dataclasses import dataclass

from folding.utils.ops import (
    run_cmd_commands,
    check_if_directory_exists,
    load_pdb_ids,
    select_random_pdb_id,
    FF_WATER_PAIRS,
)


# root level directory for the project (I HATE THIS)
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Protein:
    @property
    def name(self):
        return self.protein_pdb.split(".")[0]

    def __init__(
        self, ff: str, box: str, config: Dict, pdb_id: str = None, water: str = None
    ):
        self.pdb_id = pdb_id
        self.ff = ff
        self.box = box

        if water is not None:
            bt.logging.warning(
                "config.protein.water is not None... Potentially deviating away from recomended GROMACS FF+Water pairing"
            )
            self.water = water
        else:
            self.water = FF_WATER_PAIRS[self.ff]

        self.config = config
        self.md_inputs = {}

    def gather_pdb_id(self):
        if self.pdb_id is None:
            PDB_IDS = load_pdb_ids(
                root_dir=ROOT_DIR, filename="pdb_ids.pkl"
            )  # TODO: This should be a class variable via config
            self.pdb_id = select_random_pdb_id(PDB_IDS=PDB_IDS)
            bt.logging.success(f"Selected random pdb id: {self.pdb_id!r}")

        self.pdb_id = (
            self.pdb_id.lower()
        )  # pdb_id is insensitive to capitalization so we convert to lowercase

        self.pdb_file = f"{self.pdb_id}.pdb"
        self.pdb_file_tmp = f"{self.pdb_id}_protein_tmp.pdb"
        self.pdb_file_cleaned = f"{self.pdb_id}_protein.pdb"

    def setup_pdb_directory(self):
        bt.logging.info(f"\nChanging output directory to {self.pdb_directory}")

        # if directory doesn't exist, download the pdb file and save it to the directory
        if not os.path.exists(self.pdb_directory):
            os.makedirs(self.pdb_directory)

        if not os.path.exists(os.path.join(self.pdb_directory, self.pdb_file)):
            bt.logging.info(
                f"\n⏰ {self.pdb_file} does not exist in repository... Downloading"
            )
            self.download_pdb(pdb_directory=self.pdb_directory, pdb_file=self.pdb_file)
            bt.logging.info(f"\n💬 {self.pdb_file} Downloaded!")
        else:
            bt.logging.success(
                f"PDB file {self.pdb_file} already exists in path {self.pdb_directory!r}."
            )

    # Function to download PDB file
    def download_pdb(self, pdb_directory: str, pdb_file: str):
        url = f"https://files.rcsb.org/download/{pdb_file}"
        path = os.path.join(pdb_directory, f"{pdb_file}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(path, "w") as file:
                file.write(r.text)
            bt.logging.info(
                f"PDB file {pdb_file} downloaded successfully from {url} to path {path!r}."
            )
        else:
            bt.logging.error(
                f"Failed to download PDB file with ID {self.pdb_file} from {url}"
            )
            raise Exception(f"Failed to download PDB file with ID {self.pdb_file}.")

    def check_for_missing_files(self, required_files: List[str]):
        missing_files = [
            filename
            for filename in required_files
            if not os.path.exists(os.path.join(self.pdb_directory, filename))
        ]

        if len(missing_files) > 0:
            return missing_files
        return None

    def forward(self):
        """forward method defines the following:
        1. gather the pdb_id and setup the namings.
        2. setup the pdb directory and download the pdb file if it doesn't exist.
        3. check for missing files and generate the input files if they are missing.
        4. edit the necessary config files and add them to the synapse object self.md_inputs[file] = content
        4. save the files to the validator directory for record keeping.
        """
        ## Setup the protein directory and sample a random pdb_id if not provided
        self.gather_pdb_id()

        self.base_directory = os.path.join(str(ROOT_DIR), "data")
        self.pdb_directory = os.path.join(self.base_directory, self.pdb_id)
        self.pdb_location = os.path.join(self.pdb_directory, self.pdb_file)

        self.setup_pdb_directory()

        self.validator_directory = os.path.join(self.pdb_directory, "validator")
        self.gro_path = os.path.join(self.validator_directory, "em.gro")
        self.topol_path = os.path.join(self.validator_directory, "topol.top")

        mdp_files = ["nvt.mdp", "npt.mdp", "md.mdp"]
        other_files = [
            "em.gro",
            "topol.top",
            "posre.itp",
            "posre_Protein_chain_A.itp",
            "posre_Protein_chain_L.itp",
        ]

        required_files = mdp_files + other_files
        missing_files = self.check_for_missing_files(required_files=required_files)

        if missing_files is not None:
            bt.logging.warning(
                f"Essential files are missing from path {self.pdb_directory!r}: {missing_files!r}... Generating!"
            )
            self.generate_input_files()

        # Create a validator directory to store the files
        check_if_directory_exists(output_directory=self.validator_directory)

        for file in other_files:
            try:
                self.md_inputs[file] = open(
                    os.path.join(self.validator_directory, file), "r"
                ).read()
            except Exception as E:
                bt.logging.warning(
                    f"Attempted to put file {file} in md_inputs.\nError: {E}"
                )
                continue

        params_to_change = [
            "nstvout",  # Save velocities every 0 steps
            "nstfout",  # Save forces every 0 steps
            "nstxout-compressed",  # Save coordinates to trajectory every 50,000 steps
            "nstenergy",  # Save energies every 50,000 steps
            "nstlog",  # Update log file every 50,000 steps
        ]

        # Check if the files need to be changed based on the config, and then save.
        self.edit_files(mdp_files=mdp_files, params_to_change=params_to_change)
        self.save_files(
            files=self.md_inputs,
            output_directory=self.validator_directory,
            write_mode="w",
        )

        self.remaining_steps = []

    def __str__(self):
        return f"Protein(pdb_id={self.pdb_id}, ff={self.ff}, box={self.box}"

    def __repr__(self):
        return self.__str__()

    def calculate_params_save_interval(self, num_steps_to_save: int = 5) -> float:
        """determining the save_frequency to step

        Args:
            num_steps_to_save (int, optional): How many steps to save during the simulation.
                This is to reduce the amount of data that gets saved during simulation.
        """

        save_interval = self.config.max_steps // num_steps_to_save

        bt.logging.success(
            f"Setting save_interval to {save_interval}, from max_steps = {self.config.max_steps}"
        )
        if save_interval == 0:  # only happens when max_steps is < num_steps
            return 1
        return save_interval

    # Function to generate GROMACS input files
    def generate_input_files(self):
        bt.logging.info(f"Changing path to {self.pdb_directory}")
        os.chdir(self.pdb_directory)

        bt.logging.info(
            f"pdb file is set to: {self.pdb_file}, and it is located at {self.pdb_location}"
        )

        # strip away trailing number in forcefield name e.g charmm27 -> charmm, and
        # Copy mdp template files to protein directory
        ff_base = "".join([c for c in self.ff if not c.isdigit()])
        commands = [
            f"cp {self.base_directory}/nvt.mdp {self.pdb_directory}/nvt.mdp",
            f"cp {self.base_directory}/npt.mdp {self.pdb_directory}/npt.mdp",
            f"cp {self.base_directory}/md.mdp  {self.pdb_directory}/md.mdp ",
            f"cp {self.base_directory}/emin.mdp  {self.pdb_directory}/emin.mdp ",
            f"cp {self.base_directory}/minim.mdp  {self.pdb_directory}/minim.mdp",
        ]

        # Commands to generate GROMACS input files
        commands += [
            f"grep -v HETATM {self.pdb_file} > {self.pdb_file_tmp}",  # remove lines with HETATM
            f"grep -v CONECT {self.pdb_file_tmp} > {self.pdb_file_cleaned}",  # remove lines with CONECT
            f"gmx pdb2gmx -f {self.pdb_file_cleaned} -ff {self.ff} -o processed.gro -water {self.water}",  # Input the file into GROMACS and get three output files: topology, position restraint, and a post-processed structure file
            f"gmx editconf -f processed.gro -o newbox.gro -c -d 1.0 -bt {self.box}",  # Build the "box" to run our simulation of one protein molecule
            "gmx solvate -cp newbox.gro -cs spc216.gro -o solvated.gro -p topol.top",
            "touch ions.mdp",  # Create a file to add ions to the system
            "gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr",
            'echo "SOL" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral',
        ]
        # Run the first step of the simulation
        bt.logging.info("Run the first step of the simulation")
        # TODO: Move this to the miner side, but make sure that this runs!
        commands += [
            f"gmx grompp -f {self.pdb_directory}/emin.mdp -c solv_ions.gro -p topol.top -o em.tpr",
            "gmx mdrun -v -deffnm em",  # Run energy minimization
        ]

        run_cmd_commands(
            commands=commands, suppress_cmd_output=self.config.suppress_cmd_output
        )

        # Here we are going to change the path to a validator folder, and move ALL the files except the pdb file
        output_directory = os.path.join(self.pdb_directory, "validator")
        check_if_directory_exists(output_directory=output_directory)

        # Move all files
        cmd = f'find . -maxdepth 1 -type f ! -name "*.pdb" -exec mv {{}} {output_directory}/ \;'
        bt.logging.info(f"Moving all files except pdb to {output_directory}")
        os.system(cmd)

        # We want to catch any errors that occur in the above steps and then return the error to the user
        return True

    def edit_files(self, mdp_files: List, params_to_change: List):
        """Edit the files that are needed for simulation.

        Args:
            mdp_files (List): Files that contain parameters to change.
            params_to_change (List): List of parameters to change in the desired file(s)
        """
        bt.logging.info("Editing file content...")

        # TODO: save_frequency = 0.10 needs to be replaced with config.save_frequency
        save_interval = self.calculate_params_save_interval(num_steps_to_save=100)

        for file in mdp_files:
            filepath = os.path.join(self.validator_directory, file)
            bt.logging.info(f"Processing file {filepath}")
            content = open(filepath, "r").read()
            if self.config.max_steps is not None:
                content = re.sub(
                    "nsteps\\s+=\\s+\\d+", f"nsteps = {self.config.max_steps}", content
                )

            for param in params_to_change:
                if param in content:
                    bt.logging.info(f"Changing {param} in {file} to {save_interval}...")
                    content = re.sub(
                        f"{param}\\s+=\\s+\\d+",
                        f"{param} = {save_interval}",
                        content,
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

    def gro_hash(self, gro_path: str):
        """Generates the hash for a specific gro file.
        Enables validators to ensure that miners are running the correct
        protein, and not generating fake data.

        Args:
            gro_path (str): location to the gro file
        """
        bt.logging.info(f"Calculating hash for path {gro_path!r}")
        pattern = re.compile(r"\s*(\d+\w+)\s+(\w+\d*\s*\d+)\s+(\-?\d+\.\d+)+")

        with open(gro_path, "rb") as f:
            name, length, *lines, _ = f.readlines()
            length = int(length)
            bt.logging.info(f"{name=}, {length=}, {len(lines)=}")

        buf = ""
        for line in lines:
            line = line.decode().strip()
            match = pattern.match(line)
            if not match:
                raise Exception(f"Error parsing line in {gro_path!r}: {line!r}")
            buf += match.group(1) + match.group(2).replace(" ", "")

        return hashlib.md5(name + buf.encode()).hexdigest()

    def delete_files(self, directory: str):
        bt.logging.info(f"Deleting files in {directory}")
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))
        # os.rmdir(output_directory)

    def get_miner_data_directory(self, hotkey: str):
        return os.path.join(self.validator_directory, hotkey[:8])

    def process_md_output(self, md_output: Dict, hotkey: str) -> bool:
        """
        1. Check md_output for the required files, if unsuccessful return False
        2. Save files if above is valid
        3. Check the hash of the .gro file to ensure miners are running the correct protein.
        4.
        """

        required_files_extensions = ["edr", "gro"]

        # This is just mapper from the file extension to the name of the file stores in the dict.
        md_outputs_exts = {
            k.split(".")[-1]: k for k, v in md_output.items() if len(v) > 0
        }

        bt.logging.warning(
            f"Files that have been sent back by the miner: {md_output.keys()}"
        )
        bt.logging.warning(f"non zero length files: {md_outputs_exts.values()}")

        for ext in required_files_extensions:
            if ext not in md_outputs_exts:
                bt.logging.error(f"Missing file with extension {ext} in md_output")
                return False

        output_directory = self.get_miner_data_directory(hotkey=hotkey)

        # Save files so we can check the hash later.
        self.save_files(
            files=md_output,
            output_directory=output_directory,
        )

        # Check that the md_output contains the right protein through gro_hash
        gro_path = os.path.join(output_directory, md_outputs_exts["gro"])
        if self.gro_hash(self.gro_path) != self.gro_hash(gro_path):
            bt.logging.warning(
                f"The hash for .gro file from hotkey {hotkey} is incorrect, so reward is zero!"
            )
            self.delete_files(directory=output_directory)
            return False

        bt.logging.success(f"The hash for .gro file is correct!")
        return True
