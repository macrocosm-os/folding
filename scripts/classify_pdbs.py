import tqdm
import pickle as pkl
import argparse
from typing import Dict, List
import concurrent.futures
from folding.utils.ops import download_pdb

COMPLETE_IDs_FILE = "./pdb_ids_complete.pkl"
INCOMPLETE_IDs_FILE = "./pdb_ids_incomplete.pkl"
NOT_DOWNLOADABLE_IDs_FILE = "./pdb_ids_not_downloadable.pkl"

COMPLETE_PDB_FILES = "./complete_pdbs/"


def save_pkl(file, data):
    with open(file, "wb") as f:
        pkl.dump(data, f)


def verbose_analysis(complete, incomplete, not_downloadable, number_of_pdb_ids):
    complete_percentage = len(complete) / number_of_pdb_ids * 100
    incomplete_percentage = len(incomplete) / number_of_pdb_ids * 100
    not_downloadable_percentage = len(not_downloadable) / number_of_pdb_ids * 100

    print("=====================================")
    print("Analysis Summary:")
    print(f"Total number of PDB IDs: {number_of_pdb_ids}")
    print(f"Complete: {len(complete)} ({complete_percentage:.2f}%)")
    print(f"Incomplete: {len(incomplete)} ({incomplete_percentage:.2f}%)")
    print(
        f"Not Downloadable: {len(not_downloadable)} ({not_downloadable_percentage:.2f}%)"
    )
    print("=====================================")


def classify_pdb_batch(data, verbose=False):
    """Downloads PDB files from a batch of PDB IDs and classifies them into complete, incomplete, and not downloadable lists. Saves the results to pickle files.

    Args:
        data (defaultdict[List]): A batch of PDB IDs, as returned by scripts/gather_pdbs.py.
        verbose (bool, optional): If True, print the time required by the analysis and the percentages + frequencies of each list. Defaults to False.

    Returns:
        None
    """
    number_of_pdb_ids = sum([len(v) for v in data.values()])

    complete = []
    incomplete = []
    not_downloadable = []
    count = 0

    for v in tqdm(data.values()):
        for pdb_id in v:
            count += 1

            try:
                result = download_pdb(COMPLETE_PDB_FILES, pdb_id + ".pdb")
                if result:  # PDB was correctly downloaded and is complete
                    complete.append(pdb_id)
                else:  # PDB was correctly downloaded but is incomplete
                    incomplete.append(pdb_id)
            except Exception:  # Unable to download PDB
                not_downloadable.append(pdb_id)
                continue

            if count % 10 == 0:  # Saving progress for safety
                save_pkl(file=COMPLETE_IDs_FILE, data=complete)
                save_pkl(file=INCOMPLETE_IDs_FILE, data=incomplete)
                save_pkl(file=NOT_DOWNLOADABLE_IDs_FILE, data=not_downloadable)

    # Once the entire process is finished, we save all the data.
    save_pkl(file=COMPLETE_IDs_FILE, data=complete)
    save_pkl(file=INCOMPLETE_IDs_FILE, data=incomplete)
    save_pkl(file=NOT_DOWNLOADABLE_IDs_FILE, data=not_downloadable)

    if verbose:
        verbose_analysis(complete, incomplete, not_downloadable, number_of_pdb_ids)
    print(
        f"Analysis done!\nPDB ID files saved at {COMPLETE_IDs_FILE}, {INCOMPLETE_IDs_FILE}, and {NOT_DOWNLOADABLE_IDs_FILE}\nPDB files saved at {COMPLETE_PDB_FILES}"
    )
    print("=====================================")


def parallel_classify_pdb_batch(data, verbose=False):
    """
    Classifies PDB IDs in parallel and saves the results to pickle files.

    Args:
        data (dict): A dictionary containing PDB IDs to be classified into complete, incomplete and not_downloadable.
        verbose (bool, optional): If True, prints an analysis summary. Defaults to False.

    Returns:
        None

    Raises:
        None

    Example:
        data = {
            'group1': ['pdb1', 'pdb2', 'pdb3'],
            'group2': ['pdb4', 'pdb5']
        }
        parallel_classify_pdb_batch(data, verbose=True)

    """
    number_of_pdb_ids = sum([len(v) for v in data.values()])
    complete = []
    incomplete = []
    not_downloadable = []

    def process_pdb(pdb_id):
        nonlocal complete, incomplete, not_downloadable
        try:
            result = download_pdb(COMPLETE_PDB_FILES, pdb_id + ".pdb")
            if result:
                complete.append(pdb_id)
            else:
                incomplete.append(pdb_id)
        except Exception:
            not_downloadable.append(pdb_id)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_pdb, pdb_id) for v in data.values() for pdb_id in v
        ]
        concurrent.futures.wait(futures)

    save_pkl(file=COMPLETE_IDs_FILE, data=complete)
    save_pkl(file=INCOMPLETE_IDs_FILE, data=incomplete)
    save_pkl(file=NOT_DOWNLOADABLE_IDs_FILE, data=not_downloadable)

    if verbose:
        verbose_analysis(complete, incomplete, not_downloadable, number_of_pdb_ids)

    print(
        f"Analysis done!\nPDB ID files saved at {COMPLETE_IDs_FILE}, {INCOMPLETE_IDs_FILE}, and {NOT_DOWNLOADABLE_IDs_FILE}\nPDB files saved at {COMPLETE_PDB_FILES}"
    )
    print("=====================================")


def main(
    classification_type: str = "parallel",
    verbose=False,
    pdb_id_path: str = "pdb_ids.pkl",
):
    # Load the PDB IDs
    with open(pdb_id_path, "rb") as f:
        PDB_IDS = pkl.load(f)

    if classification_type == "parallel":
        parallel_classify_pdb_batch(PDB_IDS, verbose=verbose)
    else:
        classify_pdb_batch(PDB_IDS, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification_type",
        type=str,
        default="parallel",
        help="Type of classification (parallel or sequential)",
    )
    parser.add_argument(
        "--pdb_id_path",
        type=str,
        default="pdb_ids.pkl",
        help="Path to the PDB ID file",
    )
    parser.add_argument("--verbose", action="store_true", help="Print analysis summary")
    args = parser.parse_args()

    main(
        classification_type=args.classification_type,
        verbose=args.verbose,
        pdb_id_path=args.pdb_id_path,
    )
