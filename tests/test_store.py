
import os
import time
import pytest
import pandas as pd
from pathlib import Path
from folding.store import PandasJobStore, MockJob, Job

# TODO: cleanup files after tests

ROOT_PATH = Path(__file__).parent
DB_PATH = os.path.join(ROOT_PATH, "mock_data")

@pytest.mark.parametrize('mock', [True, False])
def test_create_job(mock):

    if mock:
        job = MockJob()
    else:
        job = Job(pdb='ab12', hotkeys=['a','b','c','d'])

@pytest.mark.parametrize('loss, commit_hash, gro_hash', [
    (0.1, '1234', '5678'),
    (0.2, '1234', '5678'),
    (0.3, '1234', '5678'),
    (0.4, '1234', '5678'),
])
def test_update_job(loss, commit_hash, gro_hash):

        job = MockJob()
        prev_loss = job.best_loss

        hotkey = job.hotkeys[0]

        job.update(loss=loss, hotkey=hotkey, commit_hash=commit_hash, gro_hash=gro_hash)

        assert job.updated_count == 1, f"updated count should be 1, currently is {job.updated_count}"
        assert job.updated_at > job.created_at, f"updated at should not be the same as created at"

        if loss >= prev_loss:
            return

        assert job.active == True, f"job should be active, currently is {job.active}"
        assert job.best_loss == loss, f"best loss should be {loss}, currently is {job.best_loss}"
        assert job.best_hotkey == hotkey, f"best hotkey should be {hotkey}, currently is {job.best_hotkey}"
        assert job.commit_hash == commit_hash, f"commit hash should be {commit_hash}, currently is {job.commit_hash}"
        assert job.gro_hash == gro_hash, f"gro hash should be {gro_hash}, currently is {job.gro_hash}"

def test_init_store():

    store = PandasJobStore(db_path=DB_PATH, force_create=True)

    assert store._db.empty == True, "store should be empty on initialization"

@pytest.mark.parametrize('mock', [True, False])
@pytest.mark.parametrize('to_dict', [True, False])
def test_insert_single_job_into_store(mock, to_dict):

    store = PandasJobStore(db_path=DB_PATH, force_create=True)

    info = {'pdb': 'ab12', 'hotkeys': ['a','b','c','d']}
    if mock:
        job = MockJob()
    else:
        job = Job(**info)

    if to_dict:
        store.insert(**job.to_dict())
    else:
        store.insert(pdb=job.pdb, hotkeys=job.hotkeys)

    assert store._db.empty == False, "store should not be empty after inserting a job"
    assert (
        store.get_queue(ready=False).qsize() == 1
    ), f"queue should have one job, currently has {store.get_queue(ready=False).qsize()}"

def test_repeat_insert_same_pdb_fails():

    store = PandasJobStore(db_path=DB_PATH, force_create=True)

    job = MockJob()

    store.insert(**job.to_dict())

    # This fails because the pdb is already in the store
    with pytest.raises(ValueError):
        store.insert(**job.to_dict())

@pytest.mark.parametrize('n', [0, 1, 10, 100])
def test_save_then_load_store(n):

    store = PandasJobStore(db_path=DB_PATH, force_create=True)

    for i in range(n):
        job = MockJob()
        store.insert(**job.to_dict())

    frame_before = store._db.copy()
    store = PandasJobStore(db_path=DB_PATH, force_create=False)

    frame_after = store._db.copy()
    pd.testing.assert_frame_equal(frame_before, frame_after)


@pytest.mark.parametrize('ready', [True, False])
@pytest.mark.parametrize('update_seconds', [0, 10])
def test_queue_contains_jobs(ready, update_seconds):
    # Check that queue contains active jobs when ready is False and contains ready jobs when ready is True

    store = PandasJobStore(db_path=DB_PATH, force_create=True)

    t0 = time.time()


    for i in range(10):
        job = MockJob(update_seconds=update_seconds)
        store.insert(**job.to_dict())

    elapsed = time.time() - t0
    queue = store.get_queue(ready=ready)
    if ready and elapsed < update_seconds:
        assert (
            queue.qsize() == 0
        ), f"queue should have 0 jobs, currently has {queue.qsize()}"
    assert (
        store.get_queue(ready=False).qsize() == 10
    ), f"queue should have 10 jobs, currently has {store.get_queue(ready=False).qsize()}"


def test_queue_is_empty_when_all_jobs_are_complete():

    store = PandasJobStore(db_path=DB_PATH, force_create=True)

    for i in range(10):
        job = MockJob()
        job.active = False
        store.insert(**job.to_dict())

    assert (
        store.get_queue(ready=False).qsize() == 0
    ), f"queue should be empty, currently has {store.get_queue(ready=False).qsize()}"

