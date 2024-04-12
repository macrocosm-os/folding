import os
import tqdm
import random
import subprocess

import pickle
from typing import List, Dict

import bittensor as bt


def load_pdb_ids(root_dir: str, filename: str = "pdb_ids.pkl") -> Dict[str]:
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
