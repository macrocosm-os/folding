import time
from typing import List
from itertools import chain
from collections import defaultdict 

import numpy as np

from folding.protocol import JobSubmissionSynapse
from folding.validators.protein import Protein
from folding.utils.logger import logger

def check_if_identical(event):
    """ method to check if any of the submissions are idential. If they are, 0 reward for any uids involved.
    This should only happen for uids that are submitting the same results from a single owner. 

    We look at the simulated energy via the validator to avoid tampering. Anything that is not fully reprod will be 
    caught inside of the protein.is_run_valid method.
    """

    groups = defaultdict(list)
    for idx, energy_list in enumerate(event["checked_energy"]):
        if energy_list == 0:
            continue

        groups[tuple(energy_list)].append(idx)

    # Display identical groups
    identical_groups = [indices for indices in groups.values() if len(indices) > 1]
    flattened_list = list(chain.from_iterable(identical_groups))

    if len(flattened_list) > 0:
        logger.warning(f"Setting {len(flattened_list)} / {len(event['checked_energy'])} uids to 0 reward due to identical submissions.")
        for idx in flattened_list:
            event["is_valid"][idx] = False
            event["reason"][idx] = "Identical submission to another hotkey in the group"

    return event 

def get_energies(protein: Protein, responses: List[JobSubmissionSynapse], uids: List[int]):
    """Takes all the data from reponse synapses, checks if the data is valid, and returns the energies.

    Args:
        protein (Protein): instance of the Protein class
        responses (List[JobSubmissionSynapse]): list of JobSubmissionSynapse objects
        uids (List[int]): list of uids

    Returns:
        Tuple: Tuple containing the energies and the event dictionary
    """
    event = {}
    event["is_valid"] = [False] * len(uids)
    event["checked_energy"] = [0] * len(uids)
    event["reported_energy"] = [0] * len(uids)
    event["miner_energy"] = [0] * len(uids)
    event["rmsds"] = [0] * len(uids)
    event["process_md_output_time"] = [0] * len(uids)
    event["is_run_valid_time"] = [0] * len(uids)
    event["ns_computed"] = [0] * len(uids)
    event["reason"] = [""] * len(uids)
    event["best_cpt"] = [""] * len(uids)
    event["seed"] = []

    energies = np.zeros(len(uids))

    for i, (uid, resp) in enumerate(zip(uids, responses)):
        # Ensures that the md_outputs from the miners are parsed correctly
        try:
            start_time = time.time()
            can_process = protein.process_md_output(
                md_output=resp.md_output,
                hotkey=resp.axon.hotkey,
                state=resp.miner_state,
                seed=resp.miner_seed,
            )
            event['seed'].append(resp.miner_seed)
            event["process_md_output_time"][i] = time.time() - start_time
            event["best_cpt"][i] = protein.checkpoint_path if hasattr(protein, "checkpoint_path") else ""

            if not can_process:
                continue

            if resp.dendrite.status_code != 200:
                logger.info(f"uid {uid} responded with status code {resp.dendrite.status_code}")
                continue

            ns_computed = protein.get_ns_computed()
            energy = protein.get_energy()
            rmsd = protein.get_rmsd()

            if energy == 0:
                continue

            start_time = time.time()
            is_valid, checked_energy, miner_energy, reason = protein.is_run_valid()
            event["is_run_valid_time"][i] = time.time() - start_time
            event["reason"][i] = reason

            event["checked_energy"][i] = checked_energy
            event["miner_energy"][i] = miner_energy
            event["is_valid"][i] = is_valid
            event["reported_energy"][i] = float(energy)
            event["rmsds"][i] = float(rmsd)
            event["ns_computed"][i] = float(ns_computed)

        except Exception as E:
            # If any of the above methods have an error, we will catch here.
            logger.error(f"Failed to parse miner data for uid {uid} with error: {E}")
            continue

    # Check if the miners return identical energy results. 
    event = check_if_identical(event)

    for idx, is_valid in enumerate(event["is_valid"]):
        if is_valid: 
            energies[idx] = np.median(event["checked_energy"][idx][-10:]) #energy that we computed... 
        else:
            energies[idx] = 0

    return energies, event
