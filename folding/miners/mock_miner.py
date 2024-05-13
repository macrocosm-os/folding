import os
from typing import Dict
from folding.miners.folding_miner import FoldingMiner
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")


class MockFoldingMiner(FoldingMiner):
    """A class that closely follows the true logic of the FoldingMiner class with some
    non-simulation specific tests.
    """

    def __init__(self, base_data_path: str = None, config=None):
        # Need to the make the blacklist methods None.
        self.mock = True
        super().__init__(config=config, base_data_path=base_data_path)

    def configure_commands(self, mdrun_args: str) -> Dict:
        """The set of commands that will run for each state of the simulation.

        Args:
            mdrun_args: Empty, but needed since it is referred in base class
        """
        sleep_time = int(random.uniform(2, 3))

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
