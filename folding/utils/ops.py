import hashlib
import os
import pickle as pkl
import random
import re
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List

import bittensor as bt
import requests
import tqdm

from folding.protocol import JobSubmissionSynapse

# Recommended force field-water pairs, retrieved from gromacs-2024.1/share/top
FF_WATER_PAIRS = {
    "amber03": "tip3p",  # AMBER force fields
    "amber94": "tip3p",
    "amber96": "tip3p",
    "amber99": "tip3p",
    "amber99sb-ildn": "tip3p",
    "amber99sb": "tip3p",
    "amberGS": "tip3p",
    "charmm27": "tip3p",  # CHARMM all-atom force field
    "gromos43a1": "spc",  # GROMOS force fields
    "gromos43a2": "spc",
    "gromos45a3": "spc",
    "gromos53a5": "spc",
    "gromos53a6": "spc",
    "gromos54a7": "spc",
    "oplsaa": "tip4p",  # OPLS all-atom force field
}


class GromacsException(Exception):
    """Exception raised for errors in the versioning."""

    def __init__(self, message="Version error occurred"):
        self.message = message
        super().__init__(self.message)


def delete_directory(directory: str):
    """We create a lot of files in the process of tracking pdb files.
    Therefore, we want to delete the directory after we are done with the tests.
    """
    shutil.rmtree(directory)


def load_pdb_ids(root_dir: str, filename: str = "pdb_ids.pkl") -> Dict[str, List[str]]:
    """If you want to randomly sample pdb_ids, you need to load in
    the data that was computed via the gather_pdbs.py script.

    Args:
        root_dir (str): location of the file that contains all the names of pdb_ids
        filename (str, optional): name of the pdb_id file. Defaults to "pdb_ids.pkl".
    """
    PDB_PATH = os.path.join(root_dir, filename)

    if not os.path.exists(PDB_PATH):
        raise ValueError(
            f"Required Pdb file {PDB_PATH!r} was not found. Run `python scripts/gather_pdbs.py` first."
        )

    with open(PDB_PATH, "rb") as f:
        PDB_IDS = pkl.load(f)
    return PDB_IDS


def select_random_pdb_id(PDB_IDS: Dict, exclude: list = None) -> str:
    """This function is really important as its where you select the protein you want to fold"""
    while True:
        family = random.choice(list(PDB_IDS.keys()))
        choices = PDB_IDS[family]
        if not len(choices):
            continue
        selected_pdb_id = random.choice(choices)
        if exclude is not None and selected_pdb_id not in exclude:
            return selected_pdb_id


def gro_hash(gro_path: str):
    """Generates the hash for a specific gro file.
    Enables validators to ensure that miners are running the correct
    protein, and not generating fake data.

    Connects the (residue name, atom name, and residue number) from each line
    together into a single string. This way, we can ensure that the protein is the same.

    Example:
    10LYS  N  1
    10LYS  H1 2

    Output: 10LYSN1LYSH12

    Args:
        gro_path (str): location to the gro file
    """
    bt.logging.info(f"Calculating hash for path {gro_path!r}")
    pattern = re.compile(r"\s*(-?\d+\w+)\s+(\w+'?\d*\s*\d+)\s+(\-?\d+\.\d+)")

    with open(gro_path, "rb") as f:
        name, length, *lines, _ = f.readlines()
        name = (
            name.decode().split(" t=")[0].strip("\n").encode()
        )  # if we are rerunning the gro file using trajectory, we need to include this
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


def calc_potential_from_edr(
    output_dir: str = None, edr_name: str = "em.edr", xvg_name: str = "_tmp.xvg"
):
    """Calculate the potential energy from an edr file using gmx energy.
    Args:
        output_dir (str): directory containing the edr file
        edr_name (str): name of the edr file
        xvg_name (str): name of the xvg file

    Returns:
        float: potential energy
    """
    edr_file = os.path.join(output_dir, edr_name)
    xvg_file = os.path.join(output_dir, xvg_name)
    command = [f"echo 'Potential' | gmx energy -f {edr_file} -o {xvg_file} -nobackup"]

    run_cmd_commands(command, verbose=True)

    # Just take the last line of the 2 column xvg file (step, energy) and report the energy
    with open(xvg_file, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]
        return float(lines[-1].split()[-1])


def check_if_directory_exists(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        bt.logging.debug(f"Created directory {output_directory!r}")


def get_tracebacks():
    """A generic traceback function obtain the traceback details of an exception."""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    formatted_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)

    bt.logging.error(" ---------------- Traceback details ---------------- ")
    bt.logging.warning("".join(formatted_traceback))
    bt.logging.warning(" ---------------- End of Traceback ----------------\n")


def run_cmd_commands(
    commands: List[str], suppress_cmd_output: bool = True, verbose: bool = False
):
    for cmd in tqdm.tqdm(commands):
        bt.logging.debug(f"Running command: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if not suppress_cmd_output:
                bt.logging.info(result.stdout.decode())

        except subprocess.CalledProcessError as e:
            bt.logging.error(f"❌ Failed to run command ❌: {cmd}")
            if verbose:
                bt.logging.error(f"Output: {e.stdout.decode()}")
                bt.logging.error(f"Error: {e.stderr.decode()}")
                get_tracebacks()
            raise


def check_and_download_pdbs(
    pdb_directory: str, pdb_id: str, download: bool = True, force: bool = False
) -> bool:
    """Check the status and optionally download a PDB file from the RCSB PDB database.

    Args:
        pdb_directory (str): Directory to save the downloaded PDB file.
        pdb_id (str): PDB file ID to download.

    Returns:
        bool: True if the PDB file is downloaded successfully and doesn't contain missing values, False otherwise.

    Raises:
        Exception: If download fails.

    """
    url = f"https://files.rcsb.org/download/{pdb_id}"
    path = os.path.join(pdb_directory, f"{pdb_id}")

    r = requests.get(url)
    if r.status_code == 200:
        is_complete = is_pdb_complete(r.text)
        if is_complete or force:
            if download:
                check_if_directory_exists(output_directory=pdb_directory)
                with open(path, "w") as file:
                    file.write(r.text)

            message = " but contains missing values." if not is_complete else ""
            bt.logging.success(f"PDB file {pdb_id} downloaded" + message)

            return True
        else:
            bt.logging.warning(
                f"🚫 PDB file {pdb_id} downloaded successfully but contains missing values. 🚫"
            )
            return False
    else:
        bt.logging.error(f"Failed to download PDB file with ID {pdb_id} from {url}")
        raise Exception(f"Failed to download PDB file with ID {pdb_id}.")


def is_pdb_complete(pdb_text: str) -> bool:
    """Check if the downloaded PDB file is complete.

    Returns:
        bool: True if the PDB file is complete, False otherwise.

    """
    missing_values = {"missing heteroatom", "missing residues", "missing atom"}
    pdb_text_lower = pdb_text.lower()
    for value in missing_values:
        if value in pdb_text_lower:
            return False
    return True


def get_response_info(responses: List[JobSubmissionSynapse]) -> Dict:
    """Gather all desired response information from the set of miners."""

    response_times = []
    response_status_messages = []
    response_status_codes = []
    response_returned_files = []
    response_returned_files_sizes = []
    response_miners_serving = []

    for resp in responses:
        if resp.dendrite.process_time != None:
            response_times.append(resp.dendrite.process_time)
        else:
            response_times.append(0)

        response_status_messages.append(str(resp.dendrite.status_message))
        response_status_codes.append(str(resp.dendrite.status_code))
        response_returned_files.append(list(resp.md_output.keys()))
        response_returned_files_sizes.append(list(map(len, resp.md_output.values())))
        response_miners_serving.append(resp.miner_serving)

    return {
        "response_times": response_times,
        "response_status_messages": response_status_messages,
        "response_status_codes": response_status_codes,
        "response_returned_files": response_returned_files,
        "response_returned_files_sizes": response_returned_files_sizes,
        "response_miners_serving": response_miners_serving,
    }


def get_last_step_time(log_file: str) -> float:
    """Validators need to know where miners are in the simulation procedure to ensure that
    the gro file that is computed is done on the most recent step of the simulation. The easiest
    way to do this is by checking a log file and parsing it such that it finds the Step Time header.

    args:
        log_file (str): location of the log file that contains the step time header
    """
    step_pattern = re.compile(r"^\s*Step\s+Time$")
    step_value_pattern = re.compile(r"^\s*(\d+)\s+([\d.]+)$")

    num_matches = 0
    last_step_time = 0  # default incase we don't have more than 1 log.

    # Open and read the log file
    with open(log_file, "r") as file:
        lines = file.readlines()

    # Reverse iterate over the lines for efficiency
    for i, line in enumerate(reversed(lines)):
        if step_pattern.match(line.strip()):  # Check for "Step Time" header
            # Check the previous line in the original order for step value
            value_line = lines[-1 + (-i + 1)]
            match = step_value_pattern.match(value_line.strip())
            if match:
                num_matches += 1
                if num_matches > 1:  # get second last line. Most stable.
                    last_step_time = float(
                        match.group(2)
                    )  # group looks like:   191   0.3200
                    break

    return last_step_time
