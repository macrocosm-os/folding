import pytest

import os
import bittensor as bt
from pathlib import Path
from folding.store import PandasJobStore, MockJob

ROOT_PATH = Path(__file__).parent


def test_job_store():
    bt.logging.warning(f"WHERE ROOT PATH IS: {ROOT_PATH}")

    job = MockJob()
    store = PandasJobStore(
        db_path=os.path.join(ROOT_PATH, "mock_data"), force_create=True
    )

    assert store._db.empty == True, "store should be empty on initialization"

    store.insert(pdb=job.pdb, hotkeys=job.hotkeys)

    assert store._db.empty == False, "store should not be empty after inserting a job"
    assert (
        store.get_queue(ready=False).qsize() == 1
    ), f"queue should have one job, currently has {store.get_queue(ready=False).qsize()}"
