import time
import copy
from typing import List, Dict, Any

import numpy as np
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
    miner_registry: MinerRegistry,
):
    """Evaluates the miner's response and updates the miner registry."""

    for uid, resp in zip(uids, responses):
        miner_files = {}
        try:
            if resp.dendrite.status_code != 200:
                logger.info(
                    f"uid {uid} responded with status code {resp.dendrite.status_code}"
                )
                continue

            start_time = time.time()
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

            miner_files["best_cpt"] = (
                evaluator.checkpoint_path
                if hasattr(evaluator, "checkpoint_path")
                else ""
            )
            miner_files["folded_pdb"] = (
                evaluator.folded_pdb_path
                if hasattr(evaluator, "folded_pdb_path")
                else ""
            )
            miner_files["system_config"] = (
                evaluator.system_config_path
                if hasattr(evaluator, "system_config_path")
                else ""
            )
            miner_files["log_file_path"] = (
                evaluator.log_file_path if hasattr(evaluator, "log_file_path") else ""
            )
            miner_files["state_xml_path"] = (
                evaluator.state_xml_path if hasattr(evaluator, "state_xml_path") else ""
            )

            miner_registry.registry[uid].logs["can_process"] = can_process
            miner_registry.registry[uid].logs[
                "reported_energy"
            ] = evaluator.get_reported_energy()
            miner_registry.registry[uid].logs["evaluator"] = evaluator
            miner_registry.registry[uid].logs["seed"] = resp.miner_seed
            miner_registry.registry[uid].logs["files"] = miner_files
            miner_registry.registry[uid].logs["process_md_output_time"] = (
                time.time() - start_time
            )
            miner_registry.registry[uid].logs["axon"] = resp.axon

        except Exception as e:
            # If any of the above methods have an error, we will catch here.
            logger.error(f"Failed to parse miner data for uid {uid} with error: {e}")
            continue

    return miner_registry


async def run_evaluation_validation_pipeline(
    validator: "Validator",
    protein: Protein,
    responses: List[JobSubmissionSynapse],
    uids: List[int],
    miner_registry: MinerRegistry,
    job_type: str,
    job_id: str,
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
    energies = np.zeros(len(uids))

    # Get initial evaluations
    miner_registry = evaluate(protein, responses, uids, job_type, miner_registry)

    all_miner_logs: Dict[int, Dict[str, Any]] = miner_registry.get_all_miner_logs()
    sorted_dict = dict(
        sorted(all_miner_logs.items(), key=lambda item: item[1]["reported_energy"])
    )

    valid_unique_count = 0
    processed_uids = []
    unique_energies = set()  # Track unique energy values

    # Process responses until we get TOP_K valid non-duplicate ones or run out of responses
    for uid, miner_data in sorted_dict.items():
        try:
            axon = miner_data["axon"]
            reported_energy = miner_data["reported_energy"]
            evaluator: BaseEvaluator = miner_data["evaluator"]

            if reported_energy == 0:
                continue

            ns_computed = miner_data["ns_computed"]

            # Get the miner's credibility for this task.
            validation_probability = miner_registry.get_validation_probability(
                miner_uid=uid, task=job_type
            )

            # Calculate the probability of validation based on the miner's credibility
            start_time = time.time()
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
                checked_energies = {}
                miner_energies = {}
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
                checkpoint_path,
            ) in evaluator.intermediate_checkpoint_files.items():
                miner_registry.registry[uid].logs["files"][
                    f"checkpoint_{checkpoint_num}"
                ] = checkpoint_path

            is_valid: bool = median_energy != 0.0

            # Update event dictionary for this index
            miner_registry.registry[uid].logs["is_run_valid_time"] = (
                time.time() - start_time
            )
            miner_registry.registry[uid].logs["reason"] = reason
            miner_registry.registry[uid].logs["is_valid"] = is_valid
            miner_registry.registry[uid].logs["ns_computed"] = float(ns_computed)
            miner_registry.registry[uid].logs["checked_energies"] = checked_energies
            miner_registry.registry[uid].logs["miner_energies"] = miner_energies

            percent_diff = (
                abs((median_energy - reported_energy) / reported_energy) * 100
            )

            if is_valid:
                if percent_diff > c.ANOMALY_THRESHOLD:
                    miner_registry.registry[uid].logs["is_valid"] = False
                    miner_registry.registry[uid].logs[
                        "reason"
                    ] = "energy_difference_too_large"
                    logger.warning(
                        f"uid {uid} has energy percent difference too large: {percent_diff}"
                    )
                    processed_uids.append(uid)
                    continue

                is_duplicate = any(
                    abs(median_energy - energy) < c.DIFFERENCE_THRESHOLD
                    for energy in unique_energies
                )
                miner_registry.registry[uid].logs["is_duplicate"] = is_duplicate

                if not is_duplicate:
                    unique_energies.add(median_energy)
                    valid_unique_count += 1
                    if valid_unique_count == TOP_K:
                        processed_uids.append(uid)
                        break

            processed_uids.append(uid)

        except Exception as e:
            logger.error(f"Failed to parse miner data for uid {uid} with error: {e}")
            continue

    # Update event with only the processed entries
    if len(processed_uids) > 0:
        # The information from all processed_uids
        event = {}
        for uid in processed_uids:
            event[uid] = copy.deepcopy(miner_registry.registry[uid].logs)

    for uid, miner_logs in event.items():
        if miner_logs["is_valid"] and not miner_logs["is_duplicate"]:
            energies[uid] = np.median(
                miner_logs["checked_energies"]["final"][-c.ENERGY_WINDOW_SIZE :]
            )

    return energies, event
