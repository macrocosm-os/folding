import os
import re
import pickle
import requests
from tqdm import tqdm

import bittensor as bt
from bs4 import BeautifulSoup
from collections import defaultdict


def save_data_to_pkl(data, folder_location, filename):
    with open(os.path.join(folder_location, filename), "wb") as f:
        pickle.dump(data, f)
        bt.logging.info(f"Saved data to {folder_location}/{filename}")


def extract_pdb_id(filename: str) -> str:
    result = re.search(r"pdb(.*?)\.", filename)
    if result:
        return result.group(1)
    return "ERROR"


def get_parent_pdb_directories(parent_directory: str) -> list:
    response = requests.get(parent_directory)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all <a> tags
    a_tags = soup.find_all("a")

    # Extract href content
    href_contents = [a.get("href") for a in a_tags]
    pdb_directories = [
        d.split("/")[0] for d in href_contents if len(d) == 3
    ]  # Filter out non-PDB IDs
    return pdb_directories


def main(
    parent_directory="https://files.rcsb.org/pub/pdb/data/structures/divided/pdb/",
    save_location=".",
) -> defaultdict:
    """Main function to generate a dictionary of PDB IDs for each parent directory.

    Args:
        parent_directory (str, optional): Defaults to "https://files.rcsb.org/pub/pdb/data/structures/divided/pdb/".

    Returns:
        defaultdict: parent_directory -> list of PDB IDs
    """
    pdbs = defaultdict(list)
    pdb_directories = get_parent_pdb_directories(parent_directory=parent_directory)

    count = 0
    for pdb_directory in tqdm(pdb_directories):
        response = requests.get(parent_directory + pdb_directory)
        soup = BeautifulSoup(response.content, "html.parser")
        a_tags = soup.find_all("a")
        href_contents = [a.get("href") for a in a_tags]
        pdb_files = [f for f in href_contents if f.endswith(".ent.gz")]

        for pdb_file in set(
            pdb_files
        ):  # sometimes there are duplicate entries, so loop over the set.
            pdb_id = extract_pdb_id(pdb_file)
            pdbs[pdb_directory].append(pdb_id)

        count += 1
        if count % 10 == 0:  # save every 10 iterations for safety.
            save_data_to_pkl(
                pdbs, folder_location=save_location, filename="pdb_ids.pkl"
            )


if __name__ == "__main__":
    main()
