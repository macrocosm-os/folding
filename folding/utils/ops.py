import os
import tqdm
from typing import List, Dict
import bittensor as bt
import random


def select_random_pdb_id(PDB_IDS: Dict):
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

        if suppress_cmd_output:
            cmd += " > /dev/null 2>&1"

        if os.system(cmd) != 0:
            raise Exception(f"❌ Failed to run GROMACS command ❌: {cmd}")
