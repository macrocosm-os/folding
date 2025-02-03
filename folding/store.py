import os
import json
import string
import random
import sqlite3
import requests
from queue import Queue
from typing import Dict, List

from datetime import datetime

import numpy as np
import pandas as pd

from atom.epistula.epistula import Epistula
from dotenv import load_dotenv
from gjp_models.models import JobBase, SystemConfig, SystemKwargs

load_dotenv()

rqlite_data_dir = os.getenv("RQLITE_DATA_DIR")
rqlite_ip = os.getenv("JOIN_ADDR").split(":")[0]
local_db_addr = os.getenv("RQLITE_HTTP_ADDR")

if rqlite_data_dir is None:
    raise ValueError(
        "RQLITE_DATA_DIR environment variable is not set inside the .env file"
    )
DB_DIR = os.path.abspath(rqlite_data_dir)


class SQLiteJobStore:
    """SQLite-based job store replacing the CSV-based implementation."""

    def __init__(self, db_path=DB_DIR, table_name="jobs"):
        self.db_path = db_path
        self.table_name = table_name
        self.db_file = os.path.join(self.db_path, "db.sqlite")
        self.epistula = Epistula()

    def _row_to_job(self, row) -> "Job":
        """Convert a database row to a Job object."""
        if not row:
            return None

        data = dict(row)
        # Convert stored JSON strings back to Python objects
        data["hotkeys"] = json.loads(data["hotkeys"])
        data["system_config"] = (
            json.loads(data["system_config"]) if data["system_config"] else None
        )
        data["s3_links"] = json.loads(data["s3_links"]) if data["s3_links"] else None
        data["best_cpt_links"] = (
            json.loads(data["best_cpt_links"]) if data["best_cpt_links"] else None
        )
        data["event"] = json.loads(data["event"]) if data["event"] else None

        # Convert timestamps
        for field in ["created_at", "updated_at", "best_loss_at"]:
            if data[field]:
                data[field] = pd.Timestamp(data[field])
            else:
                data[field] = pd.NaT

        # Convert boolean
        data["active"] = bool(data["active"])

        return Job(**data)

    def get_queue(self, validator_hotkey: str, ready=True) -> Queue:
        """Get active jobs as a queue."""
        print(f"Getting jobs for {validator_hotkey}")

        if ready:
            # Calculate the threshold time for ready jobs
            now = datetime.utcnow().isoformat()
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE active = 1
                AND datetime(updated_at, '+' || update_interval || ' seconds') <= datetime('{now}')
                AND validator_hotkey = '{validator_hotkey}'
            """
            print(f"query: {query}, args: {now}, {validator_hotkey}")
            response = requests.get(
                f"http://{local_db_addr}/db/query",
                params={"q": query, "level": "strong"},
            )
        else:
            response = requests.get(
                f"http://{local_db_addr}/db/query",
                params={
                    "q": f"SELECT * FROM {self.table_name} WHERE active = 1 AND validator_hotkey = '{validator_hotkey}'",
                    "level": "strong",
                },
            )
        print(f"Response: {response.json()}")

        if response.status_code != 200:
            raise ValueError(f"Failed to get jobs: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get jobs: {response['error']}")
        elif "values" not in response.keys():
            return Queue()

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]

        queue = Queue()
        for row in rows:
            job = self._row_to_job(row)
            queue.put(job)

        return queue

    def get_inactive_queue(self, validator_hotkey: str) -> Queue:
        """Get inactive jobs as a queue."""

        # TODO: Implement a way to filter it based on time. We should keep track of the last time
        # we read the db?
        response = requests.get(
            f"http://{local_db_addr}/db/query",
            params={
                "q": f"SELECT * FROM {self.table_name} WHERE active = 0 AND validator_hotkey != {validator_hotkey}",
                "consistency": "strong",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get jobs: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get jobs: {response['error']}")
        elif "values" not in response.keys():
            return Queue()

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]

        queue = Queue()
        for row in rows:
            job = self._row_to_job(row)
            queue.put(job)

        return queue

    def update_gjp_job(self, job: "Job", gjp_address: str, keypair, job_id: str):
        """
        Updates a GJP job with the given parameters.
        Args:
            job (Job): The job object containing job details.
            gjp_address (str): The address of the GJP server.
            keypair (Keypair): The keypair for authentication.
            job_id (str): The ID of the job to be updated.
        Raises:
            ValueError: If the job update fails (response status code is not 200).
        Returns:
            str: The ID of the updated job.
        """

        body = job.model_dump()

        body_bytes = self.epistula.create_message_body(body)
        headers = self.epistula.generate_header(hotkey=keypair, body=body_bytes)

        response = requests.post(
            f"http://{gjp_address}/jobs/update/{job_id}",
            headers=headers,
            data=body_bytes,
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to upload job: {response.text}")
        return response.json()["job_id"]

    # TODO: Find a different way to implement this method
    def get_all_pdbs(self) -> list:
        """
        Retrieve all PDB IDs from the job store.

        Returns:
            list: List of PDB IDs as strings
        """
        response = requests.get(
            f"http://{local_db_addr}/db/query",
            params={
                "q": f"SELECT pdb_id FROM {self.table_name}",
                "consistency": "strong",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get all PDBs: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get all PDBs: {response['error']}")
        elif "values" not in response.keys():
            return []

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]

        return [row["pdb_id"] for row in rows]

    def check_for_available_hotkeys(
        self, job: "Job", hotkeys: List[str]
    ) -> (bool, "Job"):
        """Checks the job's hotkeys to only include those that are still valid."""
        job.hotkeys = list(set(job.hotkeys) & set(hotkeys))
        if not job.hotkeys:
            job.active = False
            return False, job
        return True, job

    def __repr__(self):
        """Show current state of the database."""
        with sqlite3.connect(self.db_file) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        return f"{self.__class__.__name__}\n{df.__repr__()}"

    def upload_job(
        self,
        pdb: str,
        ff: str,
        box: str,
        water: str,
        hotkeys: list,
        job_type: str,
        system_kwargs: dict,
        keypair,
        gjp_address: str,
        epsilon: int,
        s3_links: Dict[str, str],
        **kwargs,
    ):
        """
        Upload a job to the global job pool database.

        Args:
            pdb (str): The PDB ID of the job.
            ff (str): The force field configuration.
            box (str): The box configuration.
            water (str): The water configuration.
            hotkeys (list): A list of hotkeys.
            system_kwargs (dict): Additional system configuration arguments.
            keypair (Keypair): The keypair for generating headers.
            gjp_address (str): The address of the api server.
            event (dict): Additional event data.

        Returns:
            str: The job ID of the uploaded job.

        Raises:
            ValueError: If the job upload fails.
        """
        job = Job(
            pdb_id=pdb,
            system_config=SystemConfig(
                ff=ff, box=box, water=water, system_kwargs=SystemKwargs(**system_kwargs)
            ),
            hotkeys=hotkeys,
            job_type=job_type,
            created_at=pd.Timestamp.now().floor("s"),
            updated_at=pd.Timestamp.now().floor("s"),
            epsilon=epsilon,
            s3_links=s3_links,
            priority=1,
            **kwargs,
        )

        body = job.model_dump()

        body_bytes = self.epistula.create_message_body(body)
        headers = self.epistula.generate_header(hotkey=keypair, body=body_bytes)

        response = requests.post(
            f"http://{gjp_address}/jobs", headers=headers, data=body_bytes
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to upload job: {response.text}")
        job.job_id = response.json()["job_id"]
        return job

    async def confirm_upload(self, job: "Job"):
        """
        Confirm the upload of a job to the global job pool by trying to read in the uploaded job.

        Args:
            job (Job): The job object to be confirmed.

        Returns:
            str: The job ID of the confirmed job.
        """

        response = requests.get(
            f"http://{rqlite_ip}:4001/db/query",
            params={"q": f"SELECT * FROM jobs WHERE job_id = '{job.job_id}'"},
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to confirm job: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get all PDBs: {response['error']}")
        elif "values" not in response.keys():
            return []

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]
        return rows[0]["job_id"]


# Keep the Job and MockJob classes as they are, they work well with both implementations
class Job(JobBase):
    """Job class for storing job information."""

    async def update(self, loss: float, hotkey: str, hotkeys: List[str] = None):
        """Updates the status of a job in the database. If the loss improves, the best loss, hotkey and hashes are updated."""
        if hotkeys is not None:
            assert len(hotkeys) > 0, "hotkeys must be a non-empty list"
            self.hotkeys = hotkeys

        if hotkey not in self.hotkeys:
            raise ValueError(f"Hotkey {hotkey!r} is not a valid choice")

        percent_improvement = (
            100 * (loss - self.best_loss) / self.best_loss
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
        ).total_seconds() > self.max_time_no_improvement and self.updated_count >= self.min_updates:
            self.active = False
        elif (
            isinstance(self.best_loss_at, pd._libs.tslibs.nattype.NaTType)
            and (pd.Timestamp.now().floor("s") - self.created_at).total_seconds()
            > self.max_time_no_improvement
        ):
            self.active = False


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
