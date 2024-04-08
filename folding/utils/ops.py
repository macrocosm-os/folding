import os
from typing import List
from tqdm import tqdm
import bittensor as bt


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
