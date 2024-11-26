import os
import time
import shutil
import pytest
import sqlite3
import pandas as pd
import asyncio
from pathlib import Path
from folding.store import SQLiteJobStore, MockJob, Job


ROOT_PATH = Path(__file__).parent
DB_PATH = os.path.join(ROOT_PATH, "mock_data")
PDB = "ab12"
FF = "charmm27"
BOX = "cubic"
WATER = "tip3p"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


@pytest.mark.parametrize("mock", [True, False])
def test_create_job(mock):
    if mock:
        MockJob()
    else:
        Job(
            pdb=PDB,
            ff=FF,
            box=BOX,
            water=WATER,
            hotkeys=["a", "b", "c", "d"],
            created_at=time.time(),
            updated_at=time.time(),
        )


@pytest.mark.parametrize(
    "loss, commit_hash, gro_hash",
    [
        (-0.1, "1234", "5678"),
        (-0.2, "1234", "5678"),
        (-0.3, "1234", "5678"),
        (-0.4, "1234", "5678"),
    ],
)
@pytest.mark.asyncio()
async def test_update_job(loss, commit_hash, gro_hash):
    job = MockJob()
    prev_loss = job.best_loss
    hotkey = job.hotkeys[0]

    await job.update(loss=loss, hotkey=hotkey)

    assert (
        job.updated_count == 1
    ), f"updated count should be 1, currently is {job.updated_count}"
    assert (
        job.updated_at > job.created_at
    ), f"updated at should not be the same as created at"

    if loss >= prev_loss:
        return
    print(job)
    assert job.active == True, f"job should be active, currently is {job.active}"
    assert (
        job.best_loss == loss
    ), f"best loss should be {loss}, currently is {job.best_loss}"
    assert (
        job.best_hotkey == hotkey
    ), f"best hotkey should be {hotkey}, currently is {job.best_hotkey}"


def test_init_store():
    store = SQLiteJobStore(db_path=DB_PATH)

    # Check if database is empty
    with sqlite3.connect(store.db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {store.table_name}")
        count = cursor.fetchone()[0]

    assert count == 0, "store should be empty on initialization"


@pytest.mark.parametrize("mock", [True, False])
def test_insert_single_job_into_store(mock):
    store = SQLiteJobStore(db_path=DB_PATH)

    info = {
        "pdb": PDB,
        "ff": FF,
        "box": BOX,
        "water": WATER,
        "hotkeys": ["a", "b", "c", "d"],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    if mock:
        job = MockJob()
    else:
        job = Job(**info)

    store.insert(
        pdb=job.pdb,
        ff=job.ff,
        water=job.water,
        box=job.box,
        hotkeys=job.hotkeys,
        epsilon=job.epsilon,
        system_kwargs=job.system_kwargs,
        event=job.event,
    )
    # Check database is not empty
    with sqlite3.connect(store.db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {store.table_name}")
        count = cursor.fetchone()[0]

    assert count > 0, "store should not be empty after inserting a job"
    assert (
        store.get_queue(ready=False).qsize() == 1
    ), f"queue should have one job, currently has {store.get_queue(ready=False).qsize()}"


def test_repeat_insert_same_pdb_fails():
    store = SQLiteJobStore(db_path=DB_PATH)
    job = MockJob()

    store.insert(
        pdb=job.pdb,
        ff=job.ff,
        water=job.water,
        box=job.box,
        hotkeys=job.hotkeys,
        epsilon=job.epsilon,
        system_kwargs=job.system_kwargs,
        event=job.event,
    )

    # This should fail because the pdb is already in the store
    with pytest.raises(ValueError):
        store.insert(
            pdb=job.pdb,
            ff=job.ff,
            water=job.water,
            box=job.box,
            hotkeys=job.hotkeys,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
            event=job.event,
        )


@pytest.mark.parametrize("n", [0, 1, 10, 100])
def test_save_then_load_store(n):
    # Create first store and insert jobs
    store1 = SQLiteJobStore(db_path=DB_PATH)

    jobs = []
    for i in range(n):
        job = MockJob()
        store1.insert(
            pdb=job.pdb,
            ff=job.ff,
            water=job.water,
            box=job.box,
            hotkeys=job.hotkeys,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
            event=job.event,
        )
        jobs.append(job)

    # Create second store pointing to same database
    store2 = SQLiteJobStore(db_path=DB_PATH)

    # Compare contents
    with sqlite3.connect(store1.db_file) as conn:
        df1 = pd.read_sql_query(f"SELECT * FROM {store1.table_name}", conn)

    with sqlite3.connect(store2.db_file) as conn:
        df2 = pd.read_sql_query(f"SELECT * FROM {store2.table_name}", conn)

    pd.testing.assert_frame_equal(df1, df2)
    assert len(df1) == n, f"store should have {n} jobs, has {len(df1)}"


@pytest.mark.parametrize("ready", [True, False])
@pytest.mark.parametrize("update_seconds", [0, 10])
def test_queue_contains_jobs(ready, update_seconds):
    store = SQLiteJobStore(db_path=DB_PATH)

    t0 = time.time()

    for i in range(10):
        job = MockJob(update_seconds=update_seconds)
        store.insert(
            pdb=job.pdb,
            ff=job.ff,
            water=job.water,
            box=job.box,
            hotkeys=job.hotkeys,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
            event=job.event,
        )

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
    store = SQLiteJobStore(db_path=DB_PATH)

    for i in range(10):
        job = MockJob()
        job.active = False
        store.insert(
            pdb=job.pdb,
            ff=job.ff,
            water=job.water,
            box=job.box,
            hotkeys=job.hotkeys,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
            event=job.event,
            active=job.active,
        )

    assert (
        store.get_queue(ready=False).qsize() == 0
    ), f"queue should be empty, currently has {store.get_queue(ready=False).qsize()}"


@pytest.mark.asyncio()
async def test_job_update_in_store():
    store = SQLiteJobStore(db_path=DB_PATH)
    job = MockJob()

    # Insert initial job
    store.insert(
        pdb=job.pdb,
        ff=job.ff,
        water=job.water,
        box=job.box,
        hotkeys=job.hotkeys,
        epsilon=job.epsilon,
        system_kwargs=job.system_kwargs,
        event=job.event,
    )

    # Update job
    new_loss = -0.1
    await job.update(loss=new_loss, hotkey=job.hotkeys[0])
    store.update(job)

    # Retrieve and check updated job
    with sqlite3.connect(store.db_file) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT best_loss FROM {store.table_name} WHERE pdb = ?", (job.pdb,)
        )
        stored_loss = cursor.fetchone()["best_loss"]

    assert (
        stored_loss == new_loss
    ), f"stored loss should be {new_loss}, got {stored_loss}"


def test_store_handles_complex_data_types():
    store = SQLiteJobStore(db_path=DB_PATH)

    # Create job with complex data types
    job = MockJob()
    job.event = {"type": "test", "data": [1, 2, 3]}
    job.system_kwargs = {"param1": "value1", "param2": [4, 5, 6]}

    # Insert and retrieve
    store.insert(
        pdb=job.pdb,
        ff=job.ff,
        water=job.water,
        box=job.box,
        hotkeys=job.hotkeys,
        epsilon=job.epsilon,
        system_kwargs=job.system_kwargs,
        event=job.event,
    )

    queue = store.get_queue(ready=False)
    retrieved_job = queue.get()

    assert retrieved_job.event == job.event, "Complex event data not preserved"
    assert (
        retrieved_job.system_kwargs == job.system_kwargs
    ), "Complex system_kwargs not preserved"


def test_get_all_pdbs():
    store = SQLiteJobStore(db_path=DB_PATH)

    # Insert some mock jobs
    expected_pdbs = []
    for _ in range(5):
        job = MockJob()
        expected_pdbs.append(job.pdb)
        store.insert(
            pdb=job.pdb,
            ff=job.ff,
            water=job.water,
            box=job.box,
            hotkeys=job.hotkeys,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
            event=job.event,
        )
    # Get all PDBs
    pdbs = store.get_all_pdbs()

    # Check results
    assert len(pdbs) == 5, f"Expected 5 PDBs, got {len(pdbs)}"
    assert set(pdbs) == set(expected_pdbs), "Retrieved PDBs don't match expected ones"


@pytest.mark.asyncio()
async def insert_jobs(store, jobs):
    for job in jobs:
        store.insert(
            pdb=job.pdb,
            ff=job.ff,
            water=job.water,
            box=job.box,
            hotkeys=job.hotkeys,
            epsilon=job.epsilon,
            system_kwargs=job.system_kwargs,
            event=job.event,
        )


@pytest.mark.asyncio()
async def test_simul_write():
    store = SQLiteJobStore(db_path=DB_PATH)
    jobs = [MockJob() for _ in range(100)]
    jobs2 = [MockJob() for _ in range(100)]

    task1 = asyncio.create_task(insert_jobs(store, jobs))
    task2 = asyncio.create_task(insert_jobs(store, jobs2))
    await asyncio.gather(task1, task2)

    assert (
        store.get_queue(ready=False).qsize() == 200
    ), f"queue should have 200 jobs, currently has {store.get_queue(ready=False).qsize()}"
    for job, job2 in zip(jobs, jobs2):
        assert job.pdb in store.get_all_pdbs(), f"job {job.pdb} not found in store"
        assert job2.pdb in store.get_all_pdbs(), f"job {job2.pdb} not found in store"
