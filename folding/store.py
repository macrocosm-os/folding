import os
import random
import string
import json
import sqlite3
from typing import List
from queue import Queue
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(__file__), "db")


class SQLiteJobStore:
    """SQLite-based job store replacing the CSV-based implementation."""

    def __init__(self, db_path=DB_DIR, table_name="protein_jobs"):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(self.db_path, exist_ok=True)
        self.db_file = os.path.join(self.db_path, "jobs.db")

        self.init_db()

    def init_db(self):
        """Initialize the SQLite database with the required schema."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    pdb TEXT PRIMARY KEY,
                    active INTEGER,
                    hotkeys TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    best_loss REAL,
                    best_loss_at TIMESTAMP,
                    best_hotkey TEXT,
                    commit_hash TEXT,
                    gro_hash TEXT,
                    update_interval INTEGER,
                    updated_count INTEGER,
                    max_time_no_improvement INTEGER,
                    event TEXT,
                    ff TEXT,
                    box TEXT,
                    water TEXT,
                    epsilon REAL,
                    system_kwargs TEXT,
                    min_updates INTEGER
                )
            """
            )

    def _row_to_job(self, row) -> "Job":
        """Convert a database row to a Job object."""
        if not row:
            return None

        data = dict(row)
        # Convert stored JSON strings back to Python objects
        data["hotkeys"] = json.loads(data["hotkeys"])
        data["event"] = json.loads(data["event"]) if data["event"] else None
        data["system_kwargs"] = (
            json.loads(data["system_kwargs"]) if data["system_kwargs"] else None
        )

        # Convert timestamps
        for field in ["created_at", "updated_at", "best_loss_at"]:
            if data[field]:
                data[field] = pd.Timestamp(data[field])
            else:
                data[field] = pd.NaT

        # Convert intervals
        data["update_interval"] = pd.Timedelta(seconds=data["update_interval"])
        data["max_time_no_improvement"] = pd.Timedelta(
            seconds=data["max_time_no_improvement"]
        )

        # Convert boolean
        data["active"] = bool(data["active"])

        return Job(**data)

    def _job_to_dict(self, job: "Job") -> dict:
        """Convert a Job object to a dictionary for database storage."""
        data = job.to_dict()

        # Convert Python objects to JSON strings
        data["hotkeys"] = json.dumps(data["hotkeys"])
        data["event"] = json.dumps(data["event"]) if data["event"] else None
        data["system_kwargs"] = (
            json.dumps(data["system_kwargs"]) if data["system_kwargs"] else None
        )

        # Convert timestamps to strings
        for field in ["created_at", "updated_at", "best_loss_at"]:
            if isinstance(data[field], pd.Timestamp):
                data[field] = data[field].isoformat()
            elif pd.isna(data[field]):
                data[field] = None

        # Convert intervals to seconds
        data["update_interval"] = int(data["update_interval"].total_seconds())
        data["max_time_no_improvement"] = int(
            data["max_time_no_improvement"].total_seconds()
        )

        # Convert boolean to integer
        data["active"] = int(data["active"])

        return data

    def get_queue(self, ready=True) -> Queue:
        """Get active jobs as a queue."""
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            if ready:
                # Calculate the threshold time for ready jobs
                now = datetime.utcnow().isoformat()
                query = f"""
                    SELECT * FROM {self.table_name}
                    WHERE active = 1
                    AND datetime(updated_at, '+' || update_interval || ' seconds') <= datetime(?)
                """
                cur.execute(query, (now,))
            else:
                cur.execute(f"SELECT * FROM {self.table_name} WHERE active = 1")

            rows = cur.fetchall()

        queue = Queue()
        for row in rows:
            job = self._row_to_job(row)
            queue.put(job)

        return queue

    def insert(
        self,
        pdb: str,
        ff: str,
        box: str,
        water: str,
        hotkeys: List[str],
        epsilon: float,
        system_kwargs: dict,
        **kwargs,
    ):
        """Insert a new job into the database."""
        with sqlite3.connect(self.db_file) as conn:
            cur = conn.cursor()

            # Check if job already exists
            cur.execute(f"SELECT 1 FROM {self.table_name} WHERE pdb = ?", (pdb,))
            if cur.fetchone():
                raise ValueError(f"pdb {pdb!r} is already in the store")

            # Create and convert job
            job = Job(
                pdb=pdb,
                ff=ff,
                box=box,
                water=water,
                hotkeys=hotkeys,
                created_at=pd.Timestamp.now().floor("s"),
                updated_at=pd.Timestamp.now().floor("s"),
                epsilon=epsilon,
                system_kwargs=system_kwargs,
                **kwargs,
            )

            data = self._job_to_dict(job)

            # Build the insert query
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"

            cur.execute(query, list(data.values()))

    def update(self, job: "Job"):
        """Update an existing job in the database."""
        with sqlite3.connect(self.db_file) as conn:
            cur = conn.cursor()

            data = self._job_to_dict(job)
            pdb = data.pop("pdb")  # Remove pdb from update data as it's the primary key

            # Build the update query
            set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
            query = f"UPDATE {self.table_name} SET {set_clause} WHERE pdb = ?"

            cur.execute(query, list(data.values()) + [pdb])

    def get_all_pdbs(self) -> list:
        """
        Retrieve all PDB IDs from the job store.

        Returns:
            list: List of PDB IDs as strings
        """
        with sqlite3.connect(self.db_file) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT pdb FROM {self.table_name}")
            # Flatten the list of tuples into a list of strings
            return [row[0] for row in cur.fetchall()]

    def __repr__(self):
        """Show current state of the database."""
        with sqlite3.connect(self.db_file) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        return f"{self.__class__.__name__}\n{df.__repr__()}"


# Keep the Job and MockJob classes as they are, they work well with both implementations
@dataclass
class Job:
    pdb: str
    ff: str
    box: str
    water: str
    hotkeys: list
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    active: bool = True
    best_loss: float = np.inf
    best_loss_at: pd.Timestamp = pd.NaT
    best_hotkey: str = None
    commit_hash: str = None
    gro_hash: str = None
    update_interval: pd.Timedelta = pd.Timedelta(minutes=10)
    updated_count: int = 0
    min_updates: int = 5
    max_time_no_improvement: pd.Timedelta = pd.Timedelta(minutes=25)
    epsilon: float = 0.05  # percentage.
    event: dict = None
    system_kwargs: dict = None

    def to_dict(self):
        return asdict(self)

    async def update(self, loss: float, hotkey: str, hotkeys: List[str] = None):
        """Updates the status of a job in the database. If the loss improves, the best loss, hotkey and hashes are updated."""
        if hotkeys is not None:
            assert len(hotkeys) > 0, "hotkeys must be a non-empty list"
            self.hotkeys = hotkeys

        if hotkey not in self.hotkeys:
            raise ValueError(f"Hotkey {hotkey!r} is not a valid choice")

        percent_improvement = (
            (loss - self.best_loss) / self.best_loss
            if not np.isinf(self.best_loss) and not self.best_loss == 0
            else np.nan
        )
        self.updated_at = pd.Timestamp.now().floor("s")
        self.updated_count += 1

        never_updated_better_loss = (
            np.isnan(percent_improvement) and loss < self.best_loss
        )
        better_loss = percent_improvement >= self.epsilon

        if never_updated_better_loss or better_loss:
            self.best_loss = loss
            self.best_loss_at = pd.Timestamp.now().floor("s")
            self.best_hotkey = hotkey
        elif (
            pd.Timestamp.now().floor("s") - self.best_loss_at
            > self.max_time_no_improvement
            and self.updated_count >= self.min_updates
        ):
            self.active = False
        elif (
            isinstance(self.best_loss_at, pd._libs.tslibs.nattype.NaTType)
            and pd.Timestamp.now().floor("s") - self.created_at
            > self.max_time_no_improvement
        ):
            self.active = False

    def check_for_available_hotkeys(self, hotkeys: List[str]) -> bool:
        """Checks the job's hotkeys to only include those that are still valid."""
        self.hotkeys = list(set(self.hotkeys) & set(hotkeys))
        if not self.hotkeys:
            self.active = False
            return False
        return True


class MockJob(Job):
    def __init__(self, n_hotkeys=5, update_seconds=5, stop_after_seconds=3600 * 24):
        self.pdb = self._make_pdb()
        self.ff = "charmm27"
        self.box = "cube"
        self.water = "tip3p"
        self.hotkeys = self._make_hotkeys(n_hotkeys)
        self.created_at = (
            pd.Timestamp.now().floor("s")
            - pd.Timedelta(seconds=random.randint(0, 3600 * 24))
        ).floor("s")
        self.updated_at = self.created_at
        self.best_loss = 0
        self.best_hotkey = random.choice(self.hotkeys)
        self.commit_hash = self._make_commit_hash()
        self.gro_hash = self._make_commit_hash()
        self.update_interval = pd.Timedelta(seconds=update_seconds)
        self.min_updates = 1
        self.max_time_no_improvement = pd.Timedelta(seconds=stop_after_seconds)

    @staticmethod
    def _make_pdb():
        return "".join(random.choices(string.digits + string.ascii_lowercase, k=4))

    @staticmethod
    def _make_hotkeys(n=10, length=8):
        return [MockJob._make_hotkey(length) for _ in range(n)]

    @staticmethod
    def _make_hotkey(length=8):
        return "".join(random.choices(string.digits + string.ascii_letters, k=length))

    @staticmethod
    def _make_commit_hash(k=40):
        return "".join(random.choices(string.digits + string.ascii_letters, k=k))
