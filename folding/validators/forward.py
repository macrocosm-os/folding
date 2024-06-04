import os
import time
import torch
from tqdm import tqdm
import bittensor as bt
from pathlib import Path
from typing import List, Dict

from folding.validators.protein import Protein
from folding.utils.logging import log_event
from folding.validators.reward import get_energies
from folding.protocol import FoldingSynapse

from folding.utils.ops import select_random_pdb_id, load_pdb_ids, get_response_info
from folding.validators.hyperparameters import HyperParameters


from bittensor import dendrite

ROOT_DIR = Path(__file__).resolve().parents[2]
PDB_IDS = load_pdb_ids(
    root_dir=ROOT_DIR, filename="pdb_ids.pkl"
)  # TODO: Currently this is a small list of PDBs without MISSING flags.


def run_step(
    self,
    protein: Protein,
    uids: List[int],
    timeout: float,
    mdrun_args="",  #'-ntomp 64' #limit the number of threads to 64
):
    start_time = time.time()

    # Get the list of uids to query for this step.
    axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = FoldingSynapse(
        pdb_id=protein.pdb_id, md_inputs=protein.md_inputs, mdrun_args=mdrun_args
    )

    # Make calls to the network with the prompt - this is synchronous.
    bt.logging.warning("waiting for responses....")
    responses: List[FoldingSynapse] = self.dendrite.query(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        deserialize=True,  # decodes the bytestream response inside of md_outputs.
    )

    # For now we just want to get the losses, we are not rewarding yet
    # TODO: reframe the rewarding classes to just return the loss (e.g energy) for each response
    # We need to be super careful that the shape of losses is the same as the shape of the uids (becuase re refer to things downstream by index and assign rewards to the hotkey at that index)
    energies = get_energies(protein=protein, responses=responses, uids=uids)
    response_info = get_response_info(responses=responses)

    # # Log the step event.
    event = {
        "block": self.block,
        "step_length": time.time() - start_time,
        "uids": uids,
        "energies": energies.tolist(),
        **response_info,
    }

    if len(protein.md_inputs) > 0:
        event["md_inputs"] = list(protein.md_inputs.keys())
        event["md_inputs_sizes"] = list(map(len, protein.md_inputs.values()))

    return event


def parse_config(config) -> List[str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """
    ff = config.protein.ff
    water = config.protein.water
    box = config.protein.box
    exclude_in_hp_search = []

    if ff is not None:
        exclude_in_hp_search.append("FF")
    if water is not None:
        exclude_in_hp_search.append("WATER")
    if box is not None:
        exclude_in_hp_search.append("BOX")

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

        if event.get("validator_search_status") == True:
            return event
        else:
            # forward time if validator step fails
            event["hp_search_time"] = time.time() - forward_start_time

            # only log the event if the simulation was not successful
            log_event(self, event)
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
    for tries in tqdm(
        range(hp_sampler.TOTAL_COMBINATIONS), total=hp_sampler.TOTAL_COMBINATIONS
    ):
        hp_sampler_time = time.time()

        event = {"hp_tries": tries}
        try:
            sampled_combination: Dict = hp_sampler.sample_hyperparameters()
            hps = {
                "ff": config.protein.ff or sampled_combination["FF"],
                "water": config.protein.water or sampled_combination["WATER"],
                "box": config.protein.box or sampled_combination["BOX"],
                # "BOX_DISTANCE": sampled_combination["BOX_DISTANCE"], #TODO: Add this to the downstream logic.
            }

            protein = Protein(pdb_id=pdb_id, config=config.protein, **hps)
            protein.setup_simulation()

        except Exception as E:
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

    return event
