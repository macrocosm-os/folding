import pytest

import os
import random
import string
from folding.utils.logger import logger

from pathlib import Path
import concurrent.futures
from collections import defaultdict
from folding.miners.folding_miner import MockSimulationManager

ROOT_PATH = Path(__file__).parent
OUTPUT_PATH = os.path.join(ROOT_PATH, "mock_data", "test_miner")

TOTAL_WAIT_TIME = 3  # seconds


def _make_pdb():
    return "".join(random.choices(string.digits + string.ascii_lowercase, k=4))


def delete_output_dir():
    """We create a lot of files in the process of tracking pdb files.
    Therefore, we want to delete the output directory after we are done with the tests.
    """
    if os.path.exists(OUTPUT_PATH):
        for file in os.listdir(OUTPUT_PATH):
            os.remove(os.path.join(OUTPUT_PATH, file))
        os.rmdir(OUTPUT_PATH)


def test_gromacs_executor_simple():
    executor = MockSimulationManager(pdb_id=_make_pdb(), output_dir=OUTPUT_PATH)
    executor.run(total_wait_time=1)
    state = executor.get_state()
    assert state is not None  # Add more specific checks as needed
    assert state == "finished"  # Add more specific checks as needed

    delete_output_dir()


def check_file_exists(file_path: str):
    return os.path.exists(file_path)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_simulation_manager(max_workers: int):
    def nested_dict():
        return defaultdict(
            lambda: None
        )  # allows us to set the desired attribute to anything.

    EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    simulations = defaultdict(
        nested_dict
    )  # Maps pdb_ids to the current state of the simulation

    for _ in range(max_workers):
        pdb_id = _make_pdb()
        simulation_manager = MockSimulationManager(
            pdb_id=pdb_id, output_dir=OUTPUT_PATH
        )
        assert simulation_manager is not None, "Executor should not be None"
        assert simulation_manager.state is None, "State should be None"

        future = EXECUTOR.submit(
            simulation_manager.run, total_wait_time=TOTAL_WAIT_TIME
        )

        assert future.done() is False, "Future should not be done"

        simulations[pdb_id]["future"] = future
        simulations[pdb_id]["executor"] = simulation_manager

    # Now we want to continually check the state of the simulation
    while True:
        for pdb_id, sim in simulations.items():
            if not sim["future"].done():
                # check if output path has any .txt files
                # This seems to happen because of some IO delay..
                if os.path.exists(OUTPUT_PATH):
                    if len(os.listdir(OUTPUT_PATH)) == 0:
                        logger.warning(f"OUTPUT_PATH IS EMPTY... continue")
                        continue
                else:
                    logger.warning(f"OUTPUT_PATH DOES NOT EXIST... continue")
                    continue

                current_state = sim["executor"].get_state()

                if current_state not in simulations[pdb_id]["recorded_states"]:
                    simulations[pdb_id]["recorded_states"].append(current_state)

            else:  # therefore the simulation must finish
                assert (
                    sim["executor"].get_state() == "finished"
                ), "Simulation should be finished"

                simulations[pdb_id]["recorded_states"].append(
                    sim["executor"].get_state()
                )

                assert sim["executor"].required_values.issubset(
                    set(simulations[pdb_id]["recorded_states"])
                ), f"Required values, {sim['executor'].required_values}, should be a subset of recorded states, {simulations[pdb_id]['recorded_states']}"

        delete_output_dir()
        return None
