from collections import defaultdict
import time
from typing import List

import numpy as np
import bittensor as bt
from folding.utils.logger import logger
from folding.utils import constants as c
from folding.validators.protein import Protein
from folding.base.evaluation import BaseEvaluator
from folding.protocol import JobSubmissionSynapse
from folding.registries.miner_registry import MinerRegistry
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


def evaluate(
    protein: Protein,
    responses: List[JobSubmissionSynapse],
    uids: List[int],
    job_type: str,
):
    reported_energies = np.zeros(len(uids))
    evaluators = [None] * len(uids)
    seed = [-1] * len(uids)
    files = [defaultdict(str)] * len(uids)
    process_md_output_time = [0.0] * len(uids)

    for i, (uid, resp) in enumerate(zip(uids, responses)):
        try:
            if resp.dendrite.status_code != 200:
                logger.info(
                    f"uid {uid} responded with status code {resp.dendrite.status_code}"
                )
                continue

            start_time = time.time()
            seed[i] = resp.miner_seed
            evaluator: BaseEvaluator = EVALUATION_REGISTRY[job_type](
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
                logger.info(f"uid {uid} failed to process")
                continue

            files[i]["best_cpt"] = (
                evaluator.checkpoint_path
                if hasattr(evaluator, "checkpoint_path")
                else ""
            )
            files[i]["folded_pdb"] = (
                evaluator.folded_pdb_path
                if hasattr(evaluator, "folded_pdb_path")
                else ""
            )
            files[i]["system_config"] = (
                evaluator.system_config_path
                if hasattr(evaluator, "system_config_path")
                else ""
            )
            files[i]["log_file_path"] = (
                evaluator.log_file_path if hasattr(evaluator, "log_file_path") else ""
            )
            files[i]["state_xml_path"] = (
                evaluator.state_xml_path if hasattr(evaluator, "state_xml_path") else ""
            )

            reported_energies[i] = evaluator.get_reported_energy()
            process_md_output_time[i] = time.time() - start_time
            evaluators[i] = evaluator

        except Exception as e:
            # If any of the above methods have an error, we will catch here.
            logger.error(f"Failed to parse miner data for uid {uid} with error: {e}")
            continue

    return reported_energies, evaluators, seed, files, process_md_output_time


async def get_energies(
    validator: "Validator",
    protein: Protein,
    responses: List[JobSubmissionSynapse],
    uids: List[int],
    miner_registry: MinerRegistry,
    job_type: str,
    job_id: str,
    axons: List[bt.Axon],
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
        "checked_energy_final": [{}] * len(uids),
        "miner_energy_final": [{}] * len(uids),
        "checked_energy_intermediate": [{}] * len(uids),
        "miner_energy_intermediate": [{}] * len(uids),
        "reported_energy": [0] * len(uids),
        "rmsds": [0] * len(uids),
        "is_run_valid_time": [0] * len(uids),
        "ns_computed": [0] * len(uids),
        "reason": [""] * len(uids),
        "is_duplicate": [False] * len(uids),  # Initialize is_duplicate field
    }
    energies = np.zeros(len(uids))

    # Get initial evaluations
    reported_energies, evaluators, seed, files, process_md_output_time = evaluate(
        protein, responses, uids, job_type
    )

    # Sort all lists by reported energy
    sorted_data = sorted(
        zip(
            reported_energies,
            responses,
            uids,
            evaluators,
            seed,
            files,
            process_md_output_time,
            axons,
        ),
        key=lambda x: x[0] if x[0] != 0 else float("inf"),  # Push zeros to the end
    )

    valid_unique_count = 0
    processed_indices = []
    unique_energies = set()  # Track unique energy values

    # Process responses until we get TOP_K valid non-duplicate ones or run out of responses
    checkpoint_files = {}
    for i, (
        reported_energy,
        _,
        uid,
        evaluator,
        seed,
        files,
        process_md_output_time,
        axon,
    ) in enumerate(sorted_data):
        try:
            i = uids.index(uid)
            if reported_energy == 0:
                continue

            ns_computed = evaluator.get_ns_computed()

            # Get the miner's credibility for this task.
            validation_probability = miner_registry.get_validation_probability(
                miner_uid=uid, task=job_type
            )

            # Calculate the probability of validation based on the miner's credibility
            start_time = time.time()
            checked_energies = {}
            miner_energies = {}

            if np.random.rand() < validation_probability:
                (
                    median_energy,
                    checked_energies,
                    miner_energies,
                    reason,
                ) = await evaluator.validate(
                    validator=validator, job_id=job_id, axon=axon
                )
            else:
                (
                    median_energy,
                    checked_energies["final"],
                    miner_energies["final"],
                    reason,
                ) = (
                    reported_energy,
                    evaluator.final_miner_energies,
                    evaluator.final_miner_energies,
                    "skip",
                )

            # Add intermediate checkpoint files to files dictionary
            for (
                checkpoint_num,
                checkpoint_energy,
            ) in evaluator.intermediate_checkpoint_files.items():
                checkpoint_files[f"checkpoint_{checkpoint_num}"] = checkpoint_energy

            is_valid: bool = median_energy != 0.0

            # Update event dictionary for this index
            event["is_run_valid_time"][i] = time.time() - start_time
            event["reason"][i] = reason
            event["checked_energy_final"][i] = checked_energies.pop("final", None)
            event["miner_energy_final"][i] = miner_energies.pop("final", None)
            event["is_valid"][i] = is_valid
            event["ns_computed"][i] = float(ns_computed)
            event["checked_energy_intermediate"][i] = checked_energies
            event["miner_energy_intermediate"][i] = miner_energies

            percent_diff = (
                abs((median_energy - reported_energy) / reported_energy) * 100
            )

            if is_valid:
                if percent_diff > c.ANOMALY_THRESHOLD:
                    event["is_valid"][i] = False
                    event["reason"][i] = "energy_difference_too_large"
                    logger.warning(
                        f"uid {uid} has energy percent difference too large: {percent_diff}"
                    )
                    processed_indices.append(i)
                    continue

                is_duplicate = any(
                    abs(median_energy - energy) < c.DIFFERENCE_THRESHOLD
                    for energy in unique_energies
                )
                event["is_duplicate"][i] = is_duplicate

                if not is_duplicate:
                    unique_energies.add(median_energy)
                    valid_unique_count += 1
                    if valid_unique_count == TOP_K:
                        processed_indices.append(i)
                        break

            processed_indices.append(i)

        except Exception as e:
            logger.error(f"Failed to parse miner data for uid {uid} with error: {e}")
            continue

    # Add the checkpoint files to the files dictionary
    files.update(checkpoint_files)

    # Update event with only the processed entries
    if processed_indices:
        # Get the data for processed indices
        processed_data = [sorted_data[i] for i in processed_indices]
        # Unzip the processed data
        (
            reported_energies,
            responses,
            uids,
            evaluators,
            seed,
            files,
            process_md_output_time,
            axons,
        ) = map(list, zip(*processed_data))

    # Update event dictionary with processed data
    event.update(
        {
            "seed": seed,
            "files": files,
            "process_md_output_time": process_md_output_time,
            "reported_energy": reported_energies,
        }
    )

    # Calculate final energies for valid and non-duplicate responses
    for idx, (is_valid, is_duplicate) in enumerate(
        zip(event["is_valid"], event["is_duplicate"])
    ):
        if is_valid and not is_duplicate:
            # If the reason == skip, then "checked_energy" is the miner log file energy
            energies[idx] = np.median(
                event["checked_energy_final"][idx][-c.ENERGY_WINDOW_SIZE :]
            )

    return energies, event
