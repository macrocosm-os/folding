import os
import sys
import signal
import random
import shutil
import requests
import functools
import traceback
import subprocess
import pickle as pkl
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import parmed as pmd
from openmm import app, unit
import plotly.express as px

from folding.protocol import JobSubmissionSynapse
from folding.utils.logger import logger


class TimeoutException(Exception):
    pass


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


class RsyncException(Exception):
    def __init__(self, message="Rsync error occurred"):
        self.message = message
        super().__init__(self.message)


def timeout_handler(seconds, func_name):
    raise TimeoutException(f"Function '{func_name}' timed out after {seconds} seconds")


# Decorator to apply the timeout
def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)  # Retain original function metadata
        def wrapper(*args, **kwargs):
            # Set the signal alarm with the function name
            signal.signal(
                signal.SIGALRM,
                lambda signum, frame: timeout_handler(seconds, func.__name__),
            )
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def print_on_retry(retry_state):
    function_name = retry_state.fn.__name__
    max_retries = retry_state.retry_object.stop.max_attempt_number
    logger.warning(
        f"Retrying {function_name}: retry #{retry_state.attempt_number} out of {max_retries}"
    )


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
        logger.debug(f"Created directory {output_directory!r}")


def get_tracebacks():
    """A generic traceback function obtain the traceback details of an exception."""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    formatted_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)

    logger.error(" ---------------- Traceback details ---------------- ")
    logger.warning("".join(formatted_traceback))
    logger.warning(" ---------------- End of Traceback ----------------\n")


async def check_and_download_pdbs(
    pdb_directory: str,
    pdb_id: str,
    input_source: str,
    download: bool = True,
    force: bool = False,
) -> bool:
    """Check the status and optionally download a PDB file from the specified input source.

    Args:
        pdb_directory (str): Directory to save the downloaded PDB file.
        pdb_id (str): PDB file ID to download.

    Returns:
        bool: True if the PDB file is downloaded successfully and doesn't contain missing values, False otherwise.

    Raises:
        Exception: If download fails for rcsb source
        RsyncException: If download fails for pdbe input source
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
                logger.success(f"PDB file {pdb_id} downloaded" + message)

                return True
            else:
                logger.warning(
                    f"🚫 PDB file {pdb_id} downloaded successfully but contains missing values. 🚫"
                )
                return False
        else:
            logger.error(f"Failed to download PDB file with ID {pdb_id} from {url}")
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
            logger.success(f"PDB file {pdb_id} downloaded successfully from PDBe.")

            convert_cif_to_pdb(
                cif_file=f"{pdb_directory}/{id}.cif",
                pdb_file=f"{pdb_directory}/{id}.pdb",
            )
            return True
        except subprocess.CalledProcessError as e:
            raise RsyncException(
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

    for resp in responses:
        if resp.dendrite.process_time != None:
            response_times.append(resp.dendrite.process_time)
        else:
            response_times.append(0)

        response_status_messages.append(str(resp.dendrite.status_message))
        response_status_codes.append(str(resp.dendrite.status_code))
        response_returned_files.append(list(resp.md_output.keys()))
        response_returned_files_sizes.append(list(map(len, resp.md_output.values())))

    return {
        "response_times": response_times,
        "response_status_messages": response_status_messages,
        "response_status_codes": response_status_codes,
        "response_returned_files": response_returned_files,
        "response_returned_files_sizes": response_returned_files_sizes,
    }


def convert_cif_to_pdb(cif_file: str, pdb_file: str):
    """Convert a CIF file to a PDB file using the `parmed` library."""
    try:
        structure = pmd.load_file(cif_file)
        structure.write_pdb(pdb_file)  # Write the structure to a PDB file
        logger.debug(f"Successfully converted {cif_file} to {pdb_file}")

    except Exception as e:
        logger.error(f"Failed to convert {cif_file} to PDB format. Error: {e}")


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


def load_pdb_file(pdb_file: str) -> app.PDBFile:
    """Method to take in the pdb file and load it into an OpenMM PDBFile object."""
    return app.PDBFile(pdb_file)


def save_files(files: Dict, output_directory: str, write_mode: str = "wb") -> Dict:
    """Save files generated on the validator side to a desired output directory.

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


def save_pdb(positions, topology, output_path: str):
    """Save the pdb file to the output path."""
    with open(output_path, "w") as f:
        app.PDBFile.writeFile(topology, positions, f)


def create_velm(simulation: app.Simulation) -> Dict[str, Any]:
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
