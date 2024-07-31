import os
import random
import string
import random
import string
import time
from collections import defaultdict
from pathlib import Path

import pytest
from pathlib import Path

import pytest

from folding.miners.mock_miner import MockFoldingMiner
from folding.protocol import JobSubmissionSynapse
from folding.utils.ops import delete_directory
from tests.fixtures.gro_files.default_config import get_test_config

ROOT_PATH = Path(__file__).parent
OUTPUT_PATH = os.path.join(ROOT_PATH, "mock_data", "test_miner")


def create_miner(config=None):
    return MockFoldingMiner(base_data_path=OUTPUT_PATH, config=config)


def _make_pdb():
    return "test_" + "".join(
        random.choices(string.digits + string.ascii_lowercase, k=4)
    )


@pytest.mark.parametrize("num_requests", [2])
def test_miner(num_requests: int):
    test_config = get_test_config()
    test_config.protein.pdb_id = _make_pdb()

    miner = create_miner(config=test_config)

    # the creation of N jobs
    synapses = []
    for ii in range(num_requests):
        test_synapse = JobSubmissionSynapse(
            pdb_id=test_config.protein.pdb_id + f"_{ii}",
            md_inputs={},  # label ids with _ii
        )
        synapses.append(test_synapse)

        miner.forward(synapse=test_synapse)

    time.sleep(5)  # give some time before the miner starts.

    num_miners_finished = 0
    simulations = defaultdict(list)

    while num_miners_finished < num_requests:
        num_miners_finished = 0
        for synapse in synapses:
            # pdb.set_trace(header = 'check synapse')
            returned_synapse = miner.forward(synapse=synapse)

            if len(returned_synapse.md_output) == 0:
                continue

            state = list(returned_synapse.md_output.keys())[0].split(".")[0]
            simulations[f"{state}"].append(returned_synapse.md_output)
            if state == "md_0_1":
                num_miners_finished += 1

            if num_miners_finished == num_requests:
                break

            time.sleep(2)

    num_files_returned_when_in_last_state = [len(e) for e in simulations["md_0_1"]]
    assert (
        num_files_returned_when_in_last_state.count(4) == num_requests
    ), f"Each simulation should return 4 files, but got {num_files_returned_when_in_last_state}"
    delete_directory(directory=OUTPUT_PATH)
