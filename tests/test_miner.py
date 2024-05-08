import pytest

import os
import time
import random, string
from typing import Dict
from pathlib import Path
import bittensor as bt

from folding.protocol import FoldingSynapse
from folding.miners.folding_miner import FoldingMiner
from folding.utils.config import add_miner_args, add_args
from folding.utils.ops import delete_directory

from tests.fixtures.gro_files.default_config import get_test_config

ROOT_PATH = Path(__file__).parent
OUTPUT_PATH = os.path.join(ROOT_PATH, "mock_data", "test_miner")


# create a testfoldingminer to not run anything gromacs, just some waits.
class TestFoldingMiner(FoldingMiner):
    """A class that closely follows the true logic of the FoldingMiner class with some
    non-simulation specific tests.
    """

    def __init__(self, config=None, base_data_path=OUTPUT_PATH):
        # Need to the make the blacklist methods None.
        self.blacklist = None
        self.priority = None

        super().__init__(config=config, base_data_path=base_data_path)

    def configure_commands(self, mdrun_args) -> Dict:
        """The set of commands that will run for each state of the simulation.

        Args:
            mdrun_args: Empty, but needed since it is referred in base class
        """
        sleep_time = 2
        commands = [
            'echo "command 0"',
            f"sleep {sleep_time}",
            'echo "command 1"',
            f"sleep {sleep_time}",
            'echo "command 2"',
            f"sleep {sleep_time}",
            'echo "command 3"',
        ]

        state_commands = {
            "nvt": commands[:2],
            "npt": commands[2:4],
            "md_0_1": commands[4:],
        }

        return state_commands


def create_miner(config=None):
    return TestFoldingMiner(config=config)


def _make_pdb():
    return "test_" + "".join(
        random.choices(string.digits + string.ascii_lowercase, k=4)
    )


@pytest.mark.parametrize("num_requests", [2])
def test_miner(num_requests: int):
    test_config = get_test_config()

    if "timeout" not in test_config:
        test_config["timeout"] = 10

    test_config.protein.pdb_id = _make_pdb()

    miner = create_miner(config=test_config)

    # the creation of N jobs
    for ii in range(num_requests):
        test_synapse = FoldingSynapse(
            pdb_id=test_config.protein.pdb_id + f"_{ii}", md_inputs={}
        )

        miner.forward(synapse=test_synapse)

    time.sleep(5)  # give some time before the miner starts.

    # This will ask for information from the last synapse created.
    returned_synapse = miner.forward(synapse=test_synapse)
    bt.logging.info(f"Returned synapse: {returned_synapse}")
    assert (
        len(miner.simulations.keys()) == num_requests
    ), f"The number of simulations {len(miner.simulations.keys())} should be equal to the number of requests {num_requests}"

    delete_directory(directory = OUTPUT_PATH)
