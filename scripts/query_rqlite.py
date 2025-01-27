import sqlite3
import requests
import os
import json
from typing import Dict, List, Optional
from folding.utils.logger import logger


def fetch_job_details(
    db_path: str, max_workers: int, columns: List[str], pdb_id: Optional[str] = None
) -> Dict:
    """
    Fetches job records from GJP database based on priority and specified fields.
    Optionally filters by a specific pdb_id if provided.

    Parameters:
        db_path (str): The file path to the SQLite database.
        max_workers (int): The maximum number of job records to fetch, sorted by priority in descending order.
        columns (List[str]): The list of columns to fetch from the database.
        pdb_id (Optional[str]): Specific pdb_id to filter the jobs by. If None, fetches jobs without filtering.

    Returns:
        Dict: A dictionary mapping each job 'id' to its details as specified in the columns list.
    """
    logger.info("Fetching job details from the database")
    columns_to_select = ", ".join(columns)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if pdb_id:
            query = f"SELECT id, {columns_to_select} FROM jobs WHERE pdb_id = ? ORDER BY priority DESC LIMIT 1"
            cursor.execute(query, (pdb_id,))
        else:
            query = f"SELECT id, {columns_to_select} FROM jobs ORDER BY priority DESC LIMIT ?"
            cursor.execute(query, (max_workers,))

        selected_columns = ["id"] + [desc[0] for desc in cursor.description[1:]]
        jobs = cursor.fetchall()
        if not jobs:
            logger.info("No jobs found.")
            return {}

        jobs_dict = {}
        for job in jobs:
            job_details = dict(zip(selected_columns, job))
            job_id = job_details.pop("id")
            jobs_dict[job_id] = job_details
        return jobs_dict


def download_files(job_details: Dict, output_dir: str = "./local-gjp"):
    """
    Downloads files based on links contained in the job details dictionary. The function handles
    two types of links: `s3_links` which are expected to be dictionaries containing file keys and URLs,
    and `best_cpt_links` which are expected to be lists of URLs.

    Parameters:
        job_details (Dict): A dictionary where each key is a job_id and each value is another dictionary
                            containing job details including 'pdb_id', 's3_links', and 'best_cpt_links'.
        output_dir (str): The root directory where downloaded files will be organized and stored. Each set of files
                          corresponding to a job will be placed in a subdirectory named after its `pdb_id`.
    Note:
        This function expects the `s3_links` to be a JSON string that can be decoded into a dictionary and `best_cpt_links`
        to be a JSON string that can be decoded into a list. Error handling is implemented for JSON decoding issues.
    """
    for job_id, details in job_details.items():
        pdb_id = details.get("pdb_id")
        if not pdb_id:
            logger.error(f"Missing pdb_id for job_id {job_id}")
            continue

        dir_path = os.path.join(output_dir, pdb_id)
        os.makedirs(dir_path, exist_ok=True)

        # Handle s3_links as dict
        s3_links_str = details.get("s3_links")
        if s3_links_str:
            try:
                s3_links = json.loads(s3_links_str)
                if isinstance(s3_links, dict):
                    for key, url in s3_links.items():
                        download_file(pdb_id, key, url, dir_path)
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON for s3_links for pdb_id {pdb_id}: {s3_links_str}"
                )

        # Handle best_cpt_links as a list
        best_cpt_links = details.get("best_cpt_links")
        if best_cpt_links:
            try:
                best_cpt_links = json.loads(best_cpt_links)
                if isinstance(best_cpt_links, list):
                    for url in best_cpt_links:
                        key = url.split("/")[-1]
                        download_file(pdb_id, key, url, dir_path)
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON for best_cpt_links for pdb_id {pdb_id}: {best_cpt_links}"
                )


def download_file(pdb_id, key, url, dir_path):
    """
    Handles the downloading of a single file from a specified URL into a specified directory path. This function
    is called by 'download_files' to manage individual file downloads.

    Parameters:
        pdb_id (str): The PDB ID associated with the job, used for logging purposes.
        key (str): A key or filename identifier for the file being downloaded.
        url (str): The URL from which the file will be downloaded.
        dir_path (str): The directory path where the file will be saved. This path should already exist.

    Behavior:
        - Attempts to download the file from the provided URL.
        - If successful, saves the file to the specified directory with a filename based on the 'key' and 'pdb_id'.
        - Logs the outcome of the download attempt, noting successful downloads and detailing errors for failed attempts.

    Note:
        - The function assumes HTTP(S) URLs and will handle HTTP errors. It does not perform retries and will
          raise an exception if the download fails.
    """

    file_name = f"{key}-{pdb_id}{os.path.splitext(url)[1]}"
    file_path = os.path.join(dir_path, file_name)
    logger.info(f"Attempting to download from {url} to {file_path}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Successfully downloaded {key} file for {pdb_id} to {file_path}")
    except requests.exceptions.RequestException as e:
        logger("ERROR", f"Failed to download file from {url}: {e}")


if __name__ == "__main__":
    db_path = "db/db.sqlite"
    max_workers = 2
    columns = ["job_id", "pdb_id", "best_cpt_links"]
    job_details = fetch_job_details(db_path, max_workers, columns, pdb_id="1zeg")
    logger.info(f"Job details fetched: {job_details}")
    download_files(job_details)
