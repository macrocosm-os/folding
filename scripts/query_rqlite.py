import sqlite3
import requests
import os
import json 
from folding.utils.logger import logger
from typing import Dict, List

def fetch_job_details(db_path: str, max_workers: int, columns: List[str]) -> Dict:
    """
    Fetches job records from an SQLite database based on priority and specified fields.
    """
    logger.info("Fetching job details from the database")
    columns_to_select = ', '.join(columns)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT id, {columns_to_select} FROM jobs ORDER BY priority DESC LIMIT ?"
            cursor.execute(query, (max_workers,))
            selected_columns = ['id'] + [desc[0] for desc in cursor.description[1:]]  # Include 'id' manually
            jobs = cursor.fetchall()
            if not jobs:
                logger.info("No jobs found.")
                return {}

            jobs_dict = {}
            for job in jobs:
                job_details = {selected_columns[i]: job[i] for i in range(len(job))}
                job_id = job_details.pop('id')
                jobs_dict[job_id] = job_details
            return jobs_dict
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise

def download_files(job_details: Dict, base_directory: str = "./local-gjp"):
    """
    Downloads files based on links in job details, organizing them by source (s3_links or best_cpt_links).
    """
    for job_id, details in job_details.items():
        pdb_id = details.get("pdb_id")
        for link_type in ['s3_links', 'best_cpt_links']:
            links_str = details.get(link_type)
            if pdb_id and links_str:
                try:
                    links = json.loads(links_str)
                    if isinstance(links, dict):  # Assuming the structure contains multiple file types per link_type
                        for key, url in links.items():
                            dir_path = os.path.join(base_directory, pdb_id, link_type)  # Differentiates s3 and cpt by directory
                            os.makedirs(dir_path, exist_ok=True)
                            file_name = f"{key}-{pdb_id}{os.path.splitext(url)[1]}"  # Name files uniquely to avoid overwrites
                            file_path = os.path.join(dir_path, file_name)
                            try:
                                response = requests.get(url)
                                response.raise_for_status()
                                with open(file_path, 'wb') as f:
                                    f.write(response.content)
                                logger.info(f"Downloaded {key} file for {pdb_id} to {file_path}")
                            except requests.exceptions.RequestException as e:
                                logger.error(f"Failed to download file from {url}: {e}")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {link_type} for {pdb_id}")


if __name__ == "__main__":
    db_path = "db/db.sqlite"
    max_workers = 5 
    columns = ['job_id', 'pdb_id', 'best_cpt_links']  
    job_details = fetch_job_details(db_path, max_workers, columns)
    logger.info(job_details)
    download_files(job_details)
