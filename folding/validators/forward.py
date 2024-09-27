import time
from tqdm import tqdm
import bittensor as bt
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import traceback
from folding.validators.protein import Protein
from folding.utils.logging import log_event
from folding.validators.reward import get_energies
from folding.protocol import PingSynapse, JobSubmissionSynapse

from folding.utils.ops import (
    select_random_pdb_id,
    load_pdb_ids,
    get_response_info,
    load_pkl,
)
from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY
from folding.validators.hyperparameters import HyperParameters

ROOT_DIR = Path(__file__).resolve().parents[2]
PDB_IDS = load_pdb_ids(
    root_dir=ROOT_DIR, filename="pdb_ids.pkl"
)  # TODO: Currently this is a small list of PDBs without MISSING flags.


def run_ping_step(self, uids: List[int], timeout: float) -> Dict:
    """Report a dictionary of ping information from all miners that were
    randomly sampled for this batch.
    """
    axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = PingSynapse()

    bt.logging.info(f"Pinging {len(axons)} uids")
    responses: List[PingSynapse] = self.dendrite.query(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
    )

    ping_report = defaultdict(list)
    for resp in responses:
        ping_report["miner_status"].append(resp.can_serve)
        ping_report["reported_compute"].append(resp.available_compute)

    return ping_report


def run_step(
    self,
    protein: Protein,
    uids: List[int],
    timeout: float,
    mdrun_args="",  # TODO: Remove this
) -> Dict:
    start_time = time.time()

    if protein is None:
        event = {
            "block": self.block,
            "step_length": time.time() - start_time,
            "energies": [],
            "active": False,
        }
        return event

    # Get the list of uids to query for this step.
    axons = [self.metagraph.axons[uid] for uid in uids]

    synapse = JobSubmissionSynapse(
        pdb_id=protein.pdb_id,
        md_inputs=protein.md_inputs,
        pdb_contents=protein.pdb_contents,
        system_config=protein.system_config.to_dict(),
    )

    # Make calls to the network with the prompt - this is synchronous.
    bt.logging.info("⏰ Waiting for miner responses ⏰")
    responses: List[JobSubmissionSynapse] = self.dendrite.query(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        deserialize=True,  # decodes the bytestream response inside of md_outputs.
    )

    response_info = get_response_info(responses=responses)

    # There are hotkeys that have decided to stop serving. We need to remove them from the store.
    responses_serving = []
    active_uids = []
    for ii, state in enumerate(response_info["response_miners_serving"]):
        if state:
            responses_serving.append(responses[ii])
            active_uids.append(uids[ii])

    event = {
        "block": self.block,
        "step_length": time.time() - start_time,
        "uids": active_uids,
        "energies": [],
        **response_info,
    }

    if len(responses_serving) == 0:
        bt.logging.warning(
            f"❗ No miners serving pdb_id {synapse.pdb_id}... Making job inactive. ❗"
        )
        return event

    energies, energy_event = get_energies(
        protein=protein, responses=responses_serving, uids=active_uids
    )

    # Log the step event.
    event.update({"energies": energies.tolist(), **energy_event})

    if len(protein.md_inputs) > 0:
        event["md_inputs"] = list(protein.md_inputs.keys())
        event["md_inputs_sizes"] = list(map(len, protein.md_inputs.values()))

    return event


def parse_config(config) -> Dict[str, str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """

    exclude_in_hp_search = {}

    if config.protein.ff is not None:
        exclude_in_hp_search["FF"] = config.protein.ff
    if config.protein.water is not None:
        exclude_in_hp_search["WATER"] = config.protein.water
    if config.protein.box is not None:
        exclude_in_hp_search["BOX"] = config.protein.box

    return exclude_in_hp_search


def create_new_challenge(self, exclude: List) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_challenge function.

    Args:
        exclude (List): list of pdb_ids to exclude from the search

    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """
    while True:
        forward_start_time = time.time()

        # Select a random pdb
        pdb_id = self.config.protein.pdb_id or select_random_pdb_id(
            PDB_IDS=PDB_IDS, exclude=exclude
        )

        # Perform a hyperparameter search until we find a valid configuration for the pdb
        bt.logging.warning(f"Attempting to prepare challenge for pdb {pdb_id}")
        event = try_prepare_challenge(config=self.config, pdb_id=pdb_id)

        if event.get("validator_search_status"):
            return event
        else:
            # forward time if validator step fails
            event["hp_search_time"] = time.time() - forward_start_time

            # only log the event if the simulation was not successful
            log_event(self, event, failed=True)
            bt.logging.error(
                f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
            )
            exclude.append(pdb_id)


def try_prepare_challenge(config, pdb_id: str) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """

    exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters(exclude=exclude_in_hp_search)

    bt.logging.info(f"Searching parameter space for pdb {pdb_id}")
    protein = None
    for tries in tqdm(
        range(hp_sampler.TOTAL_COMBINATIONS), total=hp_sampler.TOTAL_COMBINATIONS
    ):
        hp_sampler_time = time.time()

        event = {"hp_tries": tries}
        sampled_combination: Dict = hp_sampler.sample_hyperparameters()

        if config.protein.ff is not None:
            if (
                config.protein.ff is not None
                and config.protein.ff not in FORCEFIELD_REGISTRY
            ):
                raise ValueError(
                    f"Forcefield {config.protein.ff} not found in FORCEFIELD_REGISTRY"
                )

        if config.protein.water is not None:
            if (
                config.protein.water is not None
                and config.protein.water not in FORCEFIELD_REGISTRY
            ):
                raise ValueError(
                    f"Water {config.protein.water} not found in FORCEFIELD_REGISTRY"
                )

        hps = {
            "ff": config.protein.ff or sampled_combination["FF"],
            "water": config.protein.water or sampled_combination["WATER"],
            "box": config.protein.box or sampled_combination["BOX"],
        }

        protein = Protein(pdb_id=pdb_id, config=config.protein, **hps)

        try:
            protein.setup_simulation()

            if protein.init_energy > 0:
                raise ValueError(
                    f"Initial energy is positive: {protein.init_energy}. Simulation failed."
                )

        except Exception:
            # full traceback
            bt.logging.error(traceback.format_exc())
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event.update(hps)  # add the dictionary of hyperparameters to the event
            event["hp_sample_time"] = time.time() - hp_sampler_time
            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            event["epsilon"] = protein.epsilon

            if "validator_search_status" not in event:
                bt.logging.warning("✅✅ Simulation ran successfully! ✅✅")
                event["validator_search_status"] = True  # simulation passed!
                # break out of the loop if the simulation was successful
                break

            if tries == 10:
                bt.logging.error(f"Max tries reached for pdb_id {pdb_id} ❌❌")
                return event

    return event
