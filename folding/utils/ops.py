import os
import tqdm
import random
import subprocess
import concurrent.futures
import pickle as pkl
from typing import List, Dict
import requests

import bittensor as bt


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
        PDB_IDS = pickle.load(f)
    return PDB_IDS


def select_random_pdb_id(PDB_IDS: Dict) -> str:
    """This function is really important as its where you select the protein you want to fold"""
    while True:
        family = random.choice(list(PDB_IDS.keys()))
        choices = PDB_IDS[family]
        if len(choices):
            return random.choice(choices)


def check_if_directory_exists(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        bt.logging.debug(f"Created directory {output_directory!r}")


def run_cmd_commands(commands: List[str], suppress_cmd_output: bool = True):
    for cmd in tqdm.tqdm(commands):
        bt.logging.info(f"Running command: {cmd}")

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
            bt.logging.error(f"Output: {e.stdout.decode()}")
            bt.logging.error(f"Error: {e.stderr.decode()}")
            raise


def download_pdb(pdb_directory: str, pdb_id: str) -> bool:
    """Download a PDB file from the RCSB PDB database.

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
        if is_pdb_complete(r.text):
            with open(path, "w") as file:
                file.write(r.text)
            bt.logging.info(
                f"PDB file {pdb_id} downloaded successfully from {url} to path {path!r}."
            )
            return True
        else:
            bt.logging.error(
                f"PDB file {pdb_id} downloaded successfully but contains missing values."
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


def classify_pdb_batch(data, verbose=False):
    """Downloads PDB files from a batch of PDB IDs and classifies them into complete, incomplete, and not downloadable lists. Saves the results to pickle files.

    Args:
        data (defaultdict[List]): A batch of PDB IDs, as returned by scripts/gather_pdbs.py.
        verbose (bool, optional): If True, print the time required by the analysis and the percentages + frequencies of each list. Defaults to False.

    Returns:
        None
    """
    number_of_pdb_ids = sum([len(v) for v in data.values()])

    complete = []
    incomplete = []
    not_downloadable = []
    count = 0
    complete_file = "scripts/pdb_ids_complete.pkl"
    incomplete_file = "scripts/pdb_ids_incomplete.pkl"
    not_downloadable_file = "scripts/pdb_ids_not_downloadable.pkl"

    for v in tqdm(data.values()):
        for pdb_id in v:
            count += 1

            try:
                result = download_pdb("./complete_pdbs/", pdb_id + ".pdb")
                if result:  # PDB was correctly downloaded and is complete
                    complete.append(pdb_id)
                else:  # PDB was correctly downloaded but is incomplete
                    incomplete.append(pdb_id)
            except Exception:  # Unable to download PDB
                not_downloadable.append(pdb_id)
                continue

            if count % 10 == 0:  # Saving progress for safety
                with open(complete_file, "wb") as f:
                    pkl.dump(complete, f)
                with open(incomplete_file, "wb") as f:
                    pkl.dump(incomplete, f)
                with open(not_downloadable_file, "wb") as f:
                    pkl.dump(not_downloadable, f)

    with open(complete_file, "wb") as f:
        pkl.dump(complete, f)
    with open(incomplete_file, "wb") as f:
        pkl.dump(incomplete, f)
    with open(not_downloadable_file, "wb") as f:
        pkl.dump(not_downloadable, f)

    if verbose:
        complete_percentage = len(complete) / number_of_pdb_ids * 100
        incomplete_percentage = len(incomplete) / number_of_pdb_ids * 100
        not_downloadable_percentage = len(not_downloadable) / number_of_pdb_ids * 100

        print("=====================================")
        print("Analysis Summary:")
        print(f"Total number of PDB IDs: {number_of_pdb_ids}")
        print(f"Complete: {len(complete)} ({complete_percentage:.2f}%)")
        print(f"Incomplete: {len(incomplete)} ({incomplete_percentage:.2f}%)")
        print(
            f"Not Downloadable: {len(not_downloadable)} ({not_downloadable_percentage:.2f}%)"
        )

    print(
        f"Analysis done!\nFiles saved at {complete_file}, {incomplete_file}, and {not_downloadable_file}"
    )
    print("=====================================")


def parallel_classify_pdb_batch(data, verbose=False):
    """
    Classifies PDB IDs in parallel and saves the results to pickle files.

    Args:
        data (dict): A dictionary containing PDB IDs to be classified into complete, incomplete and not_downloadable.
        verbose (bool, optional): If True, prints an analysis summary. Defaults to False.

    Returns:
        None

    Raises:
        None

    Example:
        data = {
            'group1': ['pdb1', 'pdb2', 'pdb3'],
            'group2': ['pdb4', 'pdb5']
        }
        parallel_classify_pdb_batch(data, verbose=True)

    """
    number_of_pdb_ids = sum([len(v) for v in data.values()])
    complete = []
    incomplete = []
    not_downloadable = []
    complete_file = "scripts/pdb_ids_complete.pkl"
    incomplete_file = "scripts/pdb_ids_incomplete.pkl"
    not_downloadable_file = "scripts/pdb_ids_not_downloadable.pkl"

    def process_pdb(pdb_id):
        nonlocal complete, incomplete, not_downloadable
        try:
            result = download_pdb("./complete_pdbs/", pdb_id + ".pdb")
            if result:
                complete.append(pdb_id)
            else:
                incomplete.append(pdb_id)
        except Exception:
            not_downloadable.append(pdb_id)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_pdb, pdb_id) for v in data.values() for pdb_id in v
        ]
        concurrent.futures.wait(futures)

    with open(complete_file, "wb") as f:
        pkl.dump(complete, f)
    with open(incomplete_file, "wb") as f:
        pkl.dump(incomplete, f)
    with open(not_downloadable_file, "wb") as f:
        pkl.dump(not_downloadable, f)
    if verbose:

        complete_percentage = len(complete) / number_of_pdb_ids * 100
        incomplete_percentage = len(incomplete) / number_of_pdb_ids * 100
        not_downloadable_percentage = len(not_downloadable) / number_of_pdb_ids * 100

        print("=====================================")
        print("Analysis Summary:")
        print(f"Total number of PDB IDs: {number_of_pdb_ids}")
        print(f"Complete: {len(complete)} ({complete_percentage:.2f}%)")
        print(f"Incomplete: {len(incomplete)} ({incomplete_percentage:.2f}%)")
        print(
            f"Not Downloadable: {len(not_downloadable)} ({not_downloadable_percentage:.2f}%)"
        )

    print(
        f"Analysis done!\nFiles saved at {complete_file}, {incomplete_file}, and {not_downloadable_file}"
    )
    print("=====================================")
