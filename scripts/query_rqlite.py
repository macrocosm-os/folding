import sqlite3
import requests
import os
import json 
from folding.utils.logger import logger
from typing import Dict, List, Optional



def fetch_job_details(db_path: str, max_workers: int, columns: List[str], pdb_id: Optional[str] = None) -> Dict:
    """
    Fetches job records from an SQLite database based on priority and specified fields.
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
    columns_to_select = ', '.join(columns)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            if pdb_id:
                query = f"SELECT id, {columns_to_select} FROM jobs WHERE pdb_id = ? ORDER BY priority DESC LIMIT ?"
                cursor.execute(query, (pdb_id, max_workers,))
            else:
                query = f"SELECT id, {columns_to_select} FROM jobs ORDER BY priority DESC LIMIT ?"
                cursor.execute(query, (max_workers,))

            selected_columns = ['id'] + [desc[0] for desc in cursor.description[1:]]  
            jobs = cursor.fetchall()
            if not jobs:
                logger.info("No jobs found.")
                return {}

            jobs_dict = {}
            for job in jobs:
                job_details = dict(zip(selected_columns, job))
                job_id = job_details.pop('id')
                jobs_dict[job_id] = job_details
            return jobs_dict
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise

def download_files(job_details: Dict, output_dir: str = "./local-gjp"):
    """
    Downloads files based on links in job details, organizing them by source.
    Enhanced error logging to debug issues with downloading from best_cpt_links.
    """
    for job_id, details in job_details.items():
        pdb_id = details.get("pdb_id")
        if not pdb_id:
            logger.error(f"Missing pdb_id for job_id {job_id}")
            continue

        # Directory path for this pdb_id
        dir_path = os.path.join(output_dir, pdb_id)
        os.makedirs(dir_path, exist_ok=True)

        # Process each type of link
        for link_type in ['s3_links', 'best_cpt_links']:
            links_str = details.get(link_type)
            if links_str:
                try:
                    links = json.loads(links_str)
                    if isinstance(links, dict):
                        for key, url in links.items():
                            file_name = f"{key}-{pdb_id}{os.path.splitext(url)[1]}"
                            file_path = os.path.join(dir_path, file_name)
                            logger.info(f"Attempting to download from {url} to {file_path}")
                            try:
                                response = requests.get(url)
                                response.raise_for_status()
                                with open(file_path, 'wb') as f:
                                    f.write(response.content)
                                logger.info(f"Successfully downloaded {key} file for {pdb_id} to {file_path}")
                            except requests.exceptions.RequestException as e:
                                logger.error(f"Failed to download file from {url}: {e}")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {link_type} for {pdb_id}: {links_str}")


def download_s3_files(job_details: dict, output_dir: str = "./local-gjp"):
    """
    Downloads files from s3_links based on details in job details, organizing them by source.
    """
    for job_id, details in job_details.items():
        pdb_id = details.get("pdb_id")
        if not pdb_id:
            logger.error(f"Missing pdb_id for job_id {job_id}")
            continue

        dir_path = os.path.join(output_dir, pdb_id)
        os.makedirs(dir_path, exist_ok=True)

        links_str = details.get('s3_links')
        if links_str:
            try:
                links = json.loads(links_str)
                if isinstance(links, dict):
                    for key, url in links.items():
                        file_name = f"{key}-{pdb_id}{os.path.splitext(url)[1]}"
                        file_path = os.path.join(dir_path, file_name)
                        logger.info(f"Attempting to download from {url} to {file_path}")
                        try:
                            response = requests.get(url)
                            response.raise_for_status()
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Successfully downloaded {key} file for {pdb_id} to {file_path}")
                        except requests.exceptions.RequestException as e:
                            logger.error(f"Failed to download file from {url}: {e}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON for pdb_id {pdb_id}: {links_str}")


if __name__ == "__main__":
    db_path = "db/db.sqlite"
    max_workers = 30
    columns = ['job_id', 'pdb_id', 'best_cpt_links', 's3_links']  
    job_details = fetch_job_details(db_path, max_workers, columns, pdb_id="3a8a")
    logger.info(job_details)
    download_files(job_details)

