import os
import tqdm
from typing import List
import bittensor as bt
import subprocess

def check_if_directory_exists(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        bt.logging.debug(f"Created directory {output_directory!r}")




def run_cmd_commands(commands: List[str], suppress_cmd_output: bool = True):
    for cmd in tqdm.tqdm(commands):
        bt.logging.info(f"Running command: {cmd}")

        try:
            result = subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not suppress_cmd_output:
                bt.logging.info(result.stdout.decode())
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"❌ Failed to run command ❌: {cmd}")
            bt.logging.error(f"Output: {e.stdout.decode()}")
            bt.logging.error(f"Error: {e.stderr.decode()}")
            raise
