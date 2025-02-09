import time
from typing import List
from collections import defaultdict

import numpy as np

from folding.utils.logger import logger
from folding.utils import constants as c
from folding.validators.protein import Protein
from folding.protocol import JobSubmissionSynapse
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


def evaluate_energies(protein, responses, uids, job_type):
    """Evaluates the energies and returns structured results."""
    results = []

    for uid, resp in zip(uids, responses):
        try:
            if resp.dendrite.status_code != 200:
                logger.info(f"uid {uid} responded with status code {resp.dendrite.status_code}")
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

            if not evaluator.evaluate():
                continue

            reported_energy = evaluator.get_reported_energy()
            process_time = time.time() - start_time

            results.append({
                "uid": uid,
                "reported_energy": reported_energy,
                "evaluator": evaluator,
                "seed": resp.miner_seed,
                "best_cpt": getattr(evaluator, "checkpoint_path", ""),
                "process_time": process_time,
            })

        except Exception as e:
            logger.error(f"Failed to process uid {uid}: {e}")

    return results


def process_valid_energies(results, event, top_k=5):
    """Processes valid energies, filtering out invalid and duplicate values."""
    # Sort by reported energy, keeping original indices
    results.sort(key=lambda x: x["reported_energy"] if x["reported_energy"] != 0 else float("inf"))

    unique_energies = set()
    valid_count = 0

    for res in results:
        uid = res["uid"]
        evaluator = res["evaluator"]
        reported_energy = res["reported_energy"]

        if reported_energy == 0:
            continue

        try:
            ns_computed = evaluator.get_ns_computed()
            start_time = time.time()
            median_energy, checked_energy, miner_energy, reason = evaluator.validate()

            is_valid = median_energy != 0.0
            is_duplicate = any(abs(median_energy - e) < c.DIFFERENCE_THRESHOLD for e in unique_energies)

            # Update event
            event["reported_energy"][uid] = reported_energy
            event["checked_energy"][uid] = checked_energy
            event["miner_energy"][uid] = miner_energy
            event["reason"][uid] = reason
            event["is_valid"][uid] = is_valid and not is_duplicate
            event["is_duplicate"][uid] = is_duplicate
            event["is_run_valid_time"][uid] = time.time() - start_time
            event["ns_computed"][uid] = float(ns_computed)

            if is_valid and not is_duplicate:
                unique_energies.add(median_energy)
                valid_count += 1
                if valid_count == top_k:
                    break

        except Exception as e:
            logger.error(f"Validation failed for uid {uid}: {e}")

    return event


def get_energies(protein: Protein, responses: List[JobSubmissionSynapse], uids: List[int], job_type: str):
    """Main function to process and return energies in original UID order."""
    event = {
        "is_valid": {uid: False for uid in uids},
        "checked_energy": {uid: 0 for uid in uids},
        "reported_energy": {uid: 0 for uid in uids},
        "miner_energy": {uid: 0 for uid in uids},
        "rmsds": {uid: 0 for uid in uids},
        "is_run_valid_time": {uid: 0 for uid in uids},
        "ns_computed": {uid: 0 for uid in uids},
        "reason": {uid: "" for uid in uids},
        "is_duplicate": {uid: False for uid in uids},
    }

    # Step 1: Evaluate all energies
    results = evaluate_energies(protein, responses, uids, job_type)

    # Step 2: Process valid responses
    event = process_valid_energies(results, event)

    # Step 3: Compute final energies in original UID order
    energies = np.zeros(len(uids))
    for i, uid in enumerate(uids):
        if event["is_valid"][uid] and not event["is_duplicate"][uid]:
            energies[i] = np.median(event["checked_energy"][uid][-c.ENERGY_WINDOW_SIZE:])

    return energies, event
