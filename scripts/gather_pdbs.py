import os
import re
import pickle
import requests
from tqdm import tqdm

import bittensor as bt

import itertools
import pandas as pd

from typing import List
from bs4 import BeautifulSoup
from collections import defaultdict
from folding.utils.logger import logger


def save_data_to_pkl(data, folder_location, filename):
    with open(os.path.join(folder_location, filename), "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Saved data to {folder_location}/{filename}")


def save_data_as_df(data: defaultdict[List]):
    """Save the data as a dataframe with two columns: parent_folder and pdb."""
    all_pdbs = list(itertools.chain.from_iterable(data.values()))

    keys_list = [[key] * len(data[key]) for key in data.keys()]
    keys_list = list(itertools.chain.from_iterable(keys_list))

    # Create DataFrame with two columns
    df = pd.DataFrame({"parent_folder": keys_list, "pdb": all_pdbs})
    df.to_csv("./pdb_ids.csv", index=False)


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
    return set(pdb_directories)  # remove duplicates from a.get("href")


def get_pdb_files(parent_directory: str, pdb_directory: str):
    """iterate over the possible pdb files in the file location."""
    response = requests.get(parent_directory + pdb_directory)
    soup = BeautifulSoup(response.content, "html.parser")
    a_tags = soup.find_all("a")

    href_contents = [a.get("href") for a in a_tags]
    pdb_files = [f for f in href_contents if f.endswith(".ent.gz")]
    return set(pdb_files)  # remove duplicates from a.get("href")


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
        pdb_files = get_pdb_files(
            parent_directory=parent_directory, pdb_directory=pdb_directory
        )

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

    # save the final packet
    save_data_to_pkl(pdbs, folder_location=save_location, filename="pdb_ids.pkl")


if __name__ == "__main__":
    main()
