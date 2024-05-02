import pytest

import os
import time
from pathlib import Path
import concurrent.futures
from collections import defaultdict
import bittensor as bt
from folding.miners.folding_miner import MockGromacsExecutor

ROOT_PATH = Path(__file__).parent
OUTPUT_PATH = os.path.join(ROOT_PATH, "mock_data", "test_miner")

EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=2)
TOTAL_WAIT_TIME = 2


def test_gromacs_executor_simple():
    executor = MockGromacsExecutor(output_dir=OUTPUT_PATH)
    executor.run(total_wait_time=1)
    state = executor.get_state()
    assert state is not None  # Add more specific checks as needed
    assert state == "finished"  # Add more specific checks as needed


def test_gromacs_executor():
    def nested_dict():
        return defaultdict(
            lambda: None
        )  # allows us to set the desired attribute to anything.

    simulations = defaultdict(
        nested_dict
    )  # Maps pdb_ids to the current state of the simulation

    gromax_executor = MockGromacsExecutor(output_dir=OUTPUT_PATH)
    assert gromax_executor is not None, "Executor should not be None"
    assert gromax_executor.state is None, "State should be None"

    future = EXECUTOR.submit(gromax_executor.run, total_wait_time=TOTAL_WAIT_TIME)

    assert future.done() is False, "Future should not be done"

    simulations["test_pdb"]["future"] = future
    simulations["test_pdb"]["executor"] = gromax_executor

    recorded_states = []
    all_states = []

    # Now we want to continually check the state of the simulation
    while True:
        for _, sim in simulations.items():
            if not sim["future"].done():
                current_state = sim["executor"].get_state()
                bt.logging.info(
                    f"Inside of the while loop. Current state is {current_state}"
                )
                all_states.append(current_state)
                if current_state not in recorded_states:
                    recorded_states.append(current_state)

            else:  # therefore the simulation must finish
                assert (
                    sim["executor"].get_state() == "finished"
                ), "Simulation should be finished"
                recorded_states.append(sim["executor"].get_state())

                print(f"all states: {all_states}")
                assert sim["executor"].required_values.issubset(
                    set(recorded_states)
                ), f"Required values, {sim['executor'].required_values}, should be a subset of recorded states, {recorded_states}"

                return None
