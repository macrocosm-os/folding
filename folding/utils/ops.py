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
import gzip
import parmed as pmd

import bittensor as bt
import requests
import tqdm

from folding.protocol import JobSubmissionSynapse


class OpenMMException(Exception):
    """Exception raised for errors in the versioning."""

    def __init__(self, message="Version error occurred"):
        self.message = message
        super().__init__(self.message)


def delete_directory(directory: str):
    """We create a lot of files in the process of tracking pdb files.
    Therefore, we want to delete the directory after we are done with the tests.
    """
    shutil.rmtree(directory)


def write_pkl(data, path: str, write_mode="wb"):
    with open(path, write_mode) as f:
        pkl.dump(data, f)


def load_pkl(path: str, read_mode="rb"):
    with open(path, read_mode) as f:
        data = pkl.load(f)
    return data


def load_pdb_ids(
    root_dir: str, filename: str = "pdb_ids.pkl", input_source: str = None
) -> List[str]:
    """load pdb ids from the specified source, or from all sources if None is provided.

    Args:
        root_dir (str): location of the pdb_ids.pkl file
        filename (str, optional): Defaults to "pdb_ids.pkl".
        input_source (str, optional): A valid input source name. Defaults to None.

    Raises:
        ValueError: if the pdb file is not found
        ValueError: if the input source is not valid when not None

    Returns:
        List[str]: list of pdb ids
    """
    VALID_SOURCES = ["rcsb", "pdbe"]
    PDB_PATH = os.path.join(root_dir, filename)

    if not os.path.exists(PDB_PATH):
        raise ValueError(
            f"Required pdb file {PDB_PATH!r} was not found. Run `python scripts/gather_pdbs.py` first."
        )

    with open(PDB_PATH, "rb") as f:
        file = pkl.load(f)

    if input_source is not None:
        if input_source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid input source: {input_source}. Valid sources are {VALID_SOURCES}"
            )

        ids = file[input_source]["pdbs"]  # get the pdb_ids from the specified source
    else:  # None case
        ids = []
        for source in VALID_SOURCES:
            ids.extend(file[source]["pdbs"])

    return ids


def select_random_pdb_id(PDB_IDS: List[str], exclude: List[str] = None) -> str:
    """Select a random protein PDB ID to fold from a specified source."""
    while True:
        if not len(PDB_IDS):
            continue
        selected_pdb_id = random.choice(PDB_IDS)
        if exclude is None or selected_pdb_id not in exclude:
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
    pdb_directory: str,
    pdb_id: str,
    input_source: str,
    download: bool = True,
    force: bool = False,
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
    if input_source == None:
        choices = ["pdbe", "rcsb"]
        input_source = random.choice(choices)
        bt.logging.info(f"No input source specified. Randomly selected: {input_source}")

    path = os.path.join(pdb_directory, f"{pdb_id}")

    if input_source == "rcsb":
        url = f"https://files.rcsb.org/download/{pdb_id}"
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

    elif input_source == "pdbe":
        # strip the string of the extension
        id = pdb_id[0:4]
        substring = pdb_id[1:3]

        unzip_command = ["gunzip", f"{pdb_directory}/{id}.cif.gz"]

        rsync_command = [
            "rsync",
            "-rlpt",
            "-v",
            "-z",
            f"rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/{substring}/{id}.cif.gz",
            f"{pdb_directory}/",
        ]

        try:
            subprocess.run(rsync_command, check=True)
            subprocess.run(unzip_command, check=True)
            bt.logging.success(f"PDB file {pdb_id} downloaded successfully from PDBe.")

            convert_cif_to_pdb(
                cif_file=f"{pdb_directory}/{id}.cif",
                pdb_file=f"{pdb_directory}/{id}.pdb",
            )
            return True
        except subprocess.CalledProcessError as e:
            bt.logging.error(
                f"Failed to download PDB file with ID {pdb_id} using rsync: {e}"
            )
            raise Exception(
                f"Failed to download PDB file with ID {pdb_id} using rsync."
            )
    else:
        bt.logging.error(f"Unknown input source: {input_source}")
        raise ValueError(f"Unknown input source: {input_source}")


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


def convert_cif_to_pdb(cif_file: str, pdb_file: str):
    """Convert a CIF file to a PDB file using the `parmed` library."""
    try:
        structure = pmd.load_file(cif_file)
        structure.write_pdb(pdb_file)  # Write the structure to a PDB file
        bt.logging.debug(f"Successfully converted {cif_file} to {pdb_file}")

    except Exception as e:
        bt.logging.error(f"Failed to convert {cif_file} to PDB format. Error: {e}")
