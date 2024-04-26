import os
import random
import sqlite3
import string
import queue

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

class SQLiteJobStore:
    columns = {
                'id': 'INT AUTO_INCREMENT PRIMARY KEY',
                'pdb': 'TEXT',
                'active': 'BOOL',
                'hotkeys': 'VARCHAR(64) NOT NULL',
                'created_at': 'TIMESTAMP NOT NULL',
                'updated_at': 'TIMESTAMP NOT NULL',
                'best_loss': 'REAL',
                'best_loss_at': 'TEXT',
                'best_uid': 'TEXT',
                'commit_hash': 'TEXT',
                'gro_hash': 'TEXT',
                'num_updates': 'INT',
            }

    def __init__(self, db_path, table_name='protein_jobs'):
        self.db_path = db_path
        self.table_name = table_name

    def create_table(self):
        """Creates a table in the database to store jobs."""
        # Use context manager to handle the database connection
        with sqlite3.connect(self.db_path) as conn:
            # Use another context manager to handle the cursor
            column_definitions = ', '.join([f'{name} {datatype}' for name, datatype in self.columns.items()])
            with conn.cursor() as cursor:
                # Create table
                cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {column_definitions}
                )
                ''')
                # Commit the changes
                conn.commit()

    def get_active_jobs(self):
        """Checks DB for all jobs with active status and returns them as a DataFrame."""
        # Use context manager to handle the database connection
        with sqlite3.connect(self.db_path) as conn:
            # Use another context manager to handle the cursor
            with conn.cursor() as cursor:
                cursor.execute(f'''
                SELECT * FROM {self.table_name} WHERE status = 'active'
                ''')
                jobs = cursor.fetchall()

                # Return a pandas DataFrame containing the jobs
                return pd.DataFrame(jobs, columns=self.columns.keys())

    def insert_job(self, job):
        """Adds a list of jobs to the database."""
        # Use context manager to handle the database connection
        with sqlite3.connect(self.db_path) as conn:
            # Use another context manager to handle the cursor
            with conn.cursor() as cursor:
                # Create a list of values to insert
                values = ', '.join([job.get(column) for column in self.columns.keys()])

                # Insert the values into the table
                cursor.execute(f'''
                INSERT INTO {self.table_name} ({', '.join(self.columns.keys())})
                VALUES ({values})
                ''')

                # Commit the changes
                conn.commit()

class PandasJobStore:
    """Basic csv-based job store using pandas."""

    columns = {
        'pdb': 'str',
        'active': 'bool',
        'hotkeys': 'object',
        'created_at': 'datetime64[s]',
        'updated_at': 'datetime64[s]',
        'best_loss': 'float64',
        'best_loss_at': 'datetime64[s]',
        'best_hotkey': 'str',
        'commit_hash': 'str',
        'gro_hash': 'str',
        'update_interval': 'timedelta64[s]',
        'updated_count': 'int',
    }

    def __init__(self, db_path, table_name='protein_jobs', force_create=False):
        self.db_path = db_path
        self.table_name = table_name
        self.file_path = os.path.join(self.db_path, f'{self.table_name}.csv')

        self._db = self.load_table(force_create=force_create)

    def __repr__(self):
        """Just shows the underlying DataFrame."""
        return f'{self.__class__.__name__}\n{self._db.__repr__()}'

    def write(self):
        self._db.to_csv(self.file_path, index=False)

    def load_table(self, force_create=False):
        """Creates a table in the database to store jobs."""
        # Use context manager to handle the database connection
        if not os.path.exists(self.file_path) or force_create:
            os.makedirs(self.db_path, exist_ok=True)

            self._db = pd.DataFrame(columns=self.columns.keys())
            self.write()

        return pd.read_csv(self.file_path).astype(self.columns).set_index('pdb')

    def get_queue(self):
        """Checks DB for all jobs with active status and returns them as a DataFrame."""
        active = self._db['active'] == True
        pending = (pd.Timestamp.now() - self._db['updated_at']) >= self._db['update_interval']

        ready_jobs = self._db.loc[active & pending]

        jobs = queue.Queue()
        for pdb, row in ready_jobs.iterrows():
            jobs.put(Job(pdb=pdb, **row.to_dict()))

        return jobs

    def insert(self, pdb, hotkeys, **kwargs):
        """Adds a new job to the database."""

        if pdb in self._db.index.tolist():
            raise ValueError(f'pdb {pdb!r} is already in the store')

        job = Job(pdb=pdb, hotkeys=hotkeys, **kwargs).to_frame()

        if len(self._db) == 0:
            self._db = job#.astype(self.columns)
        else:
            self._db = pd.concat([self._db, job], ignore_index=False, axis=0)

        self.write()

    def update(self, job):
        """Updates the status of a job in the database."""

        self._db.update(job.to_frame())
        self.write()

@dataclass
class Job:
    # TODO: inherit from pydantic BaseModel which should take care of dtypes and mutation

    pdb: str
    hotkeys: str
    active: bool = True
    created_at: pd.Timestamp = pd.Timestamp.now()
    updated_at: pd.Timestamp = pd.Timestamp.now()
    best_loss: float = np.inf
    best_loss_at: pd.Timestamp = pd.NaT
    best_hotkey: str = None
    commit_hash: str = None
    gro_hash: str = None
    update_interval: pd.Timedelta = pd.Timedelta(minutes=10)
    updated_count: int = 0
    max_time_no_improvement: pd.Timedelta = pd.Timedelta(hours=6)
    min_updates: int = 10

    def to_dict(self):
        return asdict(self)

    def to_series(self):
        data = asdict(self)
        name = data.pop('pdb')
        return pd.Series(data, name=name)

    def to_frame(self):
        return pd.DataFrame([self.to_series()])

    def update(self, loss, hotkey, commit_hash, gro_hash):
        self.updated_at = pd.Timestamp.now()
        if hotkey not in self.hotkeys:
            raise ValueError(f'Hotkey {hotkey!r} is not a valid choice')

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_at = pd.Timestamp.now()
            self.best_hotkey = hotkey
            self.commit_hash = commit_hash
            self.gro_hash = gro_hash
            self.updated_count += 1
        elif (
            pd.Timestamp.now() - self.best_loss_at > self.max_time_no_improvement and self.updated_count >= self.min_updates
        ):
            self.active = False



class MockJob(Job):

    def __init__(self, n_hotkeys=5, update_seconds=5, stop_after_seconds=10):
        self.pdb = self._make_pdb()
        self.hotkeys = self._make_hotkeys(n_hotkeys)
        self.created_at = pd.Timestamp.now() - pd.Timedelta(seconds=random.randint(0, 3600*24))
        self.best_loss = random.random()
        self.best_hotkey = random.choice(self.hotkeys)
        self.commit_hash = self._make_commit_hash()
        self.gro_hash = self._make_commit_hash()
        self.update_interval = pd.Timedelta(seconds=update_seconds)
        self.min_updates = 1
        self.max_time_no_improvement = pd.Timedelta(seconds=stop_after_seconds)


    @staticmethod
    def _make_pdb():
        return ''.join(random.choices(string.digits + string.ascii_lowercase, k=4))

    @staticmethod
    def _make_hotkeys(n=10, length=8):
        return [MockJob._make_hotkey(length) for _ in range(n)]

    @staticmethod
    def _make_hotkey(length=8):
        return ''.join(random.choices(string.digits + string.ascii_letters, k=length))

    @staticmethod
    def _make_commit_hash(k=40):
        return ''.join(random.choices(string.digits + string.ascii_letters, k=k))

