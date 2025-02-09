import time
from typing import List
from itertools import chain
from collections import defaultdict

import numpy as np

from folding.utils.logger import logger
from folding.utils import constants as c
from folding.validators.protein import Protein
from folding.protocol import JobSubmissionSynapse
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


def check_if_identical(event):
    """method to check if any of the submissions are idential. If they are, 0 reward for any uids involved.
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
        logger.warning(
            f"Setting {len(flattened_list)} / {len(event['checked_energy'])} uids to 0 reward due to identical submissions."
        )
        for idx in flattened_list:
            event["is_valid"][idx] = False
            if event["reason"] == "":
                event["reason"][
                    idx
                ] = "Identical submission to another hotkey in the group"

    return event


def evaluate(
    protein: Protein,
    responses: List[JobSubmissionSynapse],
    uids: List[int],
    job_type: str,
    event: dict,
):
    reported_energies = np.zeros(len(uids))
    evaluators = [None] * len(uids)
    seed = []
    best_cpt = [""] * len(uids)
    process_md_output_time = [0.0] * len(uids)

    for i, (uid, resp) in enumerate(zip(uids, responses)):
        try:
            if resp.dendrite.status_code != 200:
                logger.info(
                    f"uid {uid} responded with status code {resp.dendrite.status_code}"
                )
                continue

            start_time = time.time()
            evaluator = EVALUATION_REGISTRY[job_type](
                pdb_id=protein.pdb_id,
                pdb_location=protein.pdb_location,
                hotkey=resp.axon.hotkey,
                state=resp.miner_state,
                seed=resp.miner_seed,
                md_output=resp.md_output,
                basepath=protein.pdb_directory,
                system_config=protein.system_config,
                velm_array_pkl_path=protein.velm_array_pkl,
            )

            can_process = evaluator.evaluate()
            if not can_process:
                continue
            seed.append(resp.miner_seed)
            best_cpt[i] = (
                evaluator.checkpoint_path
                if hasattr(evaluator, "checkpoint_path")
                else ""
            )

            reported_energies[i] = evaluator.get_reported_energy()
            process_md_output_time[i] = time.time() - start_time
            evaluators[i] = evaluator

        except Exception as E:
            # If any of the above methods have an error, we will catch here.
            logger.error(f"Failed to parse miner data for uid {uid} with error: {E}")
            continue
    return reported_energies, evaluators, seed, best_cpt, process_md_output_time


def get_energies(
    protein: Protein,
    responses: List[JobSubmissionSynapse],
    uids: List[int],
    job_type: str,
):
    """Takes all the data from reponse synapses, checks if the data is valid, and returns the energies.

    Args:
        protein (Protein): instance of the Protein class
        responses (List[JobSubmissionSynapse]): list of JobSubmissionSynapse objects
        uids (List[int]): list of uids

    Returns:
        Tuple: Tuple containing the energies and the event dictionary
    """

    TOP_K = 5

    # Initialize event dictionary with lists matching uids length
    event = {
        "is_valid": [False] * len(uids),
        "checked_energy": [0] * len(uids),
        "reported_energy": [0] * len(uids),
        "miner_energy": [0] * len(uids),
        "rmsds": [0] * len(uids),
        "is_run_valid_time": [0] * len(uids),
        "ns_computed": [0] * len(uids),
        "reason": [""] * len(uids),
        "is_duplicate": [False] * len(uids),  # Initialize is_duplicate field
    }
    energies = np.zeros(len(uids))

    # Get initial evaluations
    reported_energies, evaluators, seed, best_cpt, process_md_output_time = evaluate(
        protein, responses, uids, job_type, event
    )

    # Sort all lists by reported energy but keep original indices
    sorted_data = sorted(
        enumerate(zip(
            reported_energies,
            responses,
            uids,
            evaluators,
            seed,
            best_cpt,
            process_md_output_time,
        )),
        key=lambda x: x[1][0] if x[1][0] != 0 else float("inf")  # Push zeros to the end
    )

    # Extract sorted values and keep track of original indices
    sorted_indices, sorted_tuples = zip(*sorted_data)
    reported_energies_sorted, responses_sorted, uids_sorted, evaluators_sorted, seed_sorted, best_cpt_sorted, process_md_output_time_sorted = zip(*sorted_tuples)

    valid_unique_count = 0
    processed_indices = []
    unique_energies = set()  # Track unique energy values

    # Process responses until we get TOP_K valid non-duplicate ones or run out of responses
    for i, (reported_energy, response, uid, evaluator, s, bc, pmt) in enumerate(
        zip(
            reported_energies_sorted,
            responses_sorted,
            uids_sorted,
            evaluators_sorted,
            seed_sorted,
            best_cpt_sorted,
            process_md_output_time_sorted,
        )
    ):
        try:
            if reported_energy == 0:
                continue

            ns_computed = evaluator.get_ns_computed()

            start_time = time.time()
            median_energy, checked_energy, miner_energy, reason = evaluator.validate()

            if median_energy != 0.0:
                is_valid = True
            else:
                is_valid = False

            # Map back to original index
            original_idx = sorted_indices[i]

            # Update event dictionary for this index using original index
            event["is_run_valid_time"][original_idx] = time.time() - start_time
            event["reason"][original_idx] = reason
            event["checked_energy"][original_idx] = checked_energy
            event["miner_energy"][original_idx] = miner_energy
            event["is_valid"][original_idx] = is_valid
            event["ns_computed"][original_idx] = float(ns_computed)

            processed_indices.append(original_idx)

            if is_valid:

                if not abs(median_energy - reported_energy) < c.DIFFERENCE_THRESHOLD:
                    event["is_valid"][original_idx] = False
                    continue

                is_duplicate = any(
                    abs(median_energy - energy) < c.DIFFERENCE_THRESHOLD
                    for energy in unique_energies
                )
                event["is_duplicate"][original_idx] = is_duplicate

                if not is_duplicate:
                    unique_energies.add(median_energy)
                    valid_unique_count += 1
                    if valid_unique_count == TOP_K:
                        break

        except Exception as E:
            logger.error(f"Failed to parse miner data for uid {uid} with error: {E}")
            continue

    # Now we restore the original ordering before updating event
    event["reported_energy"] = [0] * len(uids)  # Reset to ensure original order
    for idx in processed_indices:
        event["reported_energy"][idx] = reported_energies[idx]

    # Calculate final energies in the original order
    energies = np.zeros(len(uids))
    for idx in range(len(uids)):
        if event["is_valid"][idx] and not event["is_duplicate"][idx]:
            energies[idx] = np.median(
                event["checked_energy"][idx][-c.ENERGY_WINDOW_SIZE :]
            )

    return energies, event