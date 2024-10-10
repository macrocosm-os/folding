import re
import os
import sys
import tqdm
import random
import shutil
import requests
import traceback
import subprocess
import pickle as pkl
from typing import Dict, List

import numpy as np
import pandas as pd
import parmed as pmd
import bittensor as bt
import plotly.express as px

from folding.protocol import JobSubmissionSynapse


class OpenMMException(Exception):
    """Exception raised for errors in the versioning."""

    def __init__(self, message="Version error occurred"):
        self.message = message
        super().__init__(self.message)


class ValidationError(Exception):
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


def load_and_sample_random_pdb_ids(
    root_dir: str,
    filename: str = "pdb_ids.pkl",
    input_source: str = None,
    exclude: List = None,
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

        pdb_ids = file[input_source][
            "pdbs"
        ]  # get the pdb_ids from the specified source
        pdb_id = select_random_pdb_id(PDB_IDS=pdb_ids, exclude=exclude)

    else:  # randomly sample all pdbs from all sources
        ids = {}
        for source in VALID_SOURCES:
            for pdb_id in file[source]["pdbs"]:
                ids[pdb_id] = source

        pdb_ids = list(ids.keys())
        pdb_id = select_random_pdb_id(PDB_IDS=pdb_ids, exclude=exclude)
        input_source = ids[pdb_id]

    return pdb_id, input_source


def select_random_pdb_id(PDB_IDS: List[str], exclude: List[str] = None) -> str:
    """Select a random protein PDB ID to fold from a specified source."""
    while True:
        if not len(PDB_IDS):
            continue
        selected_pdb_id = random.choice(PDB_IDS)
        if exclude is None or selected_pdb_id not in exclude:
            return selected_pdb_id


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

    if input_source not in ["rcsb", "pdbe"]:
        raise ValueError(f"Unknown input source for pdb sampling: {input_source}")

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
            raise Exception(
                f"Failed to download PDB file with ID {pdb_id} using rsync."
            )
    else:
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


def plot_miner_validator_curves(
    check_energies: np.ndarray,
    miner_energies: np.ndarray,
    percent_diff: np.ndarray,
    miner_data_directory: str,
    pdb_id: str,
    current_state: str,
    cpt_step: int,
):
    """plotting utility function for the Protein class. Used for validator-side checking of plots.
    Saves the data to the miner_data_directory.
    """
    df = pd.DataFrame([check_energies, miner_energies]).T
    df.columns = ["validator", "miner"]

    fig = px.scatter(
        df,
        title=f"Energy: {pdb_id} for state {current_state} starting at checkpoint step: {cpt_step}",
        labels={"index": "Step", "value": "Energy (kJ/mole)"},
        height=600,
        width=1400,
    )
    filename = f"{pdb_id}_cpt_step_{cpt_step}_state_{current_state}"
    fig.write_image(os.path.join(miner_data_directory, filename + "_energy.png"))

    fig = px.scatter(
        percent_diff,
        title=f"Percent Diff: {pdb_id} for state {current_state} starting at checkpoint step: {cpt_step}",
        labels={"index": "Step", "value": "Percent Diff"},
        height=600,
        width=1400,
    )
    fig.write_image(os.path.join(miner_data_directory, filename + "_percent_diff.png"))
