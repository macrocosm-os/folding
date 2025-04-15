import os
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from async_timeout import timeout
from folding.utils.s3_utils import upload_to_s3
from folding.validators.protein import Protein
from folding.utils.logging import log_event
from folding.validators.reward import get_energies
from folding.protocol import JobSubmissionSynapse, DFTJobSubmissionSynapse
import asyncio
from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY
from folding.validators.hyperparameters import HyperParameters
from folding.utils.ops import (
    load_and_sample_random_pdb_ids,
    get_response_info,
    OpenMMException,
    RsyncException,
    to_psi4_geometry_string,
    parse_custom_xyz,
)
from folding.utils.logger import logger
from folding.utils.uids import get_all_miner_uids


ROOT_DIR = Path(__file__).resolve().parents[2]


async def run_dft_step(self, job_event: Dict):
    """
    Send a DFT job to the miners and return the response.
    """
    uids = get_all_miner_uids(
        self.metagraph,
        self.config.neuron.vpermit_tao_limit,
        include_serving_in_check=False,
    )

    axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = DFTJobSubmissionSynapse(
        job_id=job_event["job_id"],
        geometry=job_event["geometry"],
    )

    responses: List[DFTJobSubmissionSynapse] = await self.dendrite.forward(
        axons=axons,
        synapse=synapse,
        timeout=10,
        deserialize=True,  # decodes the bytestream response inside of md_outputs.
    )

    response_info = get_response_info(responses=responses)
    return response_info


async def run_step(
    self,
    protein: Protein,
    timeout: float,
    job_type: str,
    job_id: str,
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

    # Get all uids on the network that are NOT validators.
    # the .is_serving flag means that the uid does not have an axon address.
    uids = get_all_miner_uids(
        self.metagraph,
        self.config.neuron.vpermit_tao_limit,
        include_serving_in_check=False,
    )

    axons = [self.metagraph.axons[uid] for uid in uids]

    system_config = protein.system_config.to_dict()
    system_config["seed"] = None  # We don't want to pass the seed to miners.

    synapse = JobSubmissionSynapse(
        pdb_id=protein.pdb_id,
        job_id=job_id,
    )

    # Make calls to the network with the prompt - this is synchronous.
    logger.info("⏰ Waiting for miner responses ⏰")
    responses: List[JobSubmissionSynapse] = await self.dendrite.forward(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        deserialize=True,  # decodes the bytestream response inside of md_outputs.
    )

    response_info = get_response_info(responses=responses)

    event = {
        "block": self.block,
        "step_length": time.time() - start_time,
        "uids": uids,
        "energies": [],
        **response_info,
    }

    energies, energy_event = await get_energies(
        validator=self,
        protein=protein,
        responses=responses,
        axons=axons,
        job_id=job_id,
        uids=uids,
        miner_registry=self.miner_registry,
        job_type=job_type,
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


# TODO: We need to be able to create a bunch of different challenges.
async def create_new_challenge(self, job_type: str, exclude: List) -> Dict:
    """Create a new challenge by sampling a random pdb_id and running a hyperparameter search
    using the try_prepare_md_challenge function.

    Args:
        exclude (List): list of pdb_ids to exclude from the search

    Returns:
        Dict: event dictionary containing the results of the hyperparameter search
    """
    if job_type == "md":
        while True:
            forward_start_time = time.time()
            if self.RSYNC_EXCEPTION_COUNT > 10:
                self.config.protein.pdb_id = None
                self.config.protein.input_source = "rcsb"

            if self.config.protein.pdb_id is not None:
                pdb_id = self.config.protein.pdb_id
            else:
                pdb_id, input_source = load_and_sample_random_pdb_ids(
                    root_dir=ROOT_DIR,
                    filename="pdb_ids.pkl",
                    input_source=self.config.protein.input_source,
                    exclude=exclude,
                )
                self.config.protein.input_source = input_source

            # Perform a hyperparameter search until we find a valid configuration for the pdb
            logger.info(f"Attempting to prepare challenge for pdb {pdb_id}")
            event = await try_prepare_md_challenge(
                self, config=self.config, pdb_id=pdb_id
            )
            event["input_source"] = self.config.protein.input_source

            if event.get("validator_search_status"):
                return event
            else:
                # forward time if validator step fails
                event["hp_search_time"] = time.time() - forward_start_time

                # only log the event if the simulation was not successful
                log_event(self, event, failed=True)
                logger.debug(
                    f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
                )
                exclude.append(pdb_id)

    elif job_type == "dft":
        while True:
            try:
                event = await try_prepare_dft_challenge(self)
                return event
            except Exception as e:
                logger.error(f"Error preparing DFT challenge: {e}")
                continue


def create_random_modifications_to_system_config(config) -> Dict:
    """create modifications of the desired parameters.

    Looks at the base bittensor config and parses parameters that we have deemed to be
    valid for random sampling. This is to increase the problem space for miners.
    """
    sampler = lambda min_val, max_val: round(np.random.uniform(min_val, max_val), 2)

    system_kwargs = {"temperature": sampler(200, 400), "friction": sampler(0.9, 1.1)}

    for param in system_kwargs.keys():
        if config.protein[param] is not None:
            system_kwargs[param] = config.protein[param]
            continue

    return system_kwargs


async def try_prepare_md_challenge(self, config, pdb_id: str) -> Dict:
    """Attempts to setup a simulation environment for the specific pdb & config
    Uses a stochastic sampler to find hyperparameters that are compatible with the protein
    """
    logger.info(f"Searching parameter space for pdb {pdb_id}")

    exclude_in_hp_search = parse_config(config)
    hp_sampler = HyperParameters(exclude=exclude_in_hp_search)

    # Create random modifications of parameters that are inside the function.
    system_kwargs = create_random_modifications_to_system_config(config=config)

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

        protein = Protein(
            pdb_id=pdb_id,
            config=config.protein,
            system_kwargs=system_kwargs,
            **hps,
        )

        try:
            async with timeout(300):
                await protein.setup_simulation()

            if protein.init_energy > 0:
                raise ValueError(
                    f"Initial energy is positive: {protein.init_energy}. Simulation failed."
                )

        except asyncio.TimeoutError as e:
            logger.info(f"Timeout occurred during simulation setup: {e}")
            event["validator_search_status"] = False
            tries = 10

        except OpenMMException as e:
            logger.info(f"OpenMMException occurred: init_energy is NaN {e}")
            event["validator_search_status"] = False

        except RsyncException as e:
            self.RSYNC_EXCEPTION_COUNT += 1
            event["validator_search_status"] = False

        except Exception as e:
            # full traceback
            logger.info(e)
            event["validator_search_status"] = False

        finally:
            event["pdb_id"] = pdb_id
            event["job_type"] = "SyntheticMD"
            event.update(hps)  # add the dictionary of hyperparameters to the event
            event["hp_sample_time"] = time.time() - hp_sampler_time
            event["pdb_complexity"] = [dict(protein.pdb_complexity)]
            event["init_energy"] = protein.init_energy
            event["epsilon"] = protein.epsilon
            event["system_kwargs"] = system_kwargs
            event["s3_links"] = {
                "testing": "testing"
            }  # overwritten below if s3 logging is on.

            if "validator_search_status" not in event:
                if not config.s3.off:
                    try:
                        logger.info(f"Uploading to {self.handler.bucket_name}")
                        s3_links = await upload_to_s3(
                            handler=self.handler,
                            pdb_location=protein.pdb_location,
                            simulation_cpt=protein.simulation_cpt,
                            validator_directory=protein.validator_directory,
                            pdb_id=pdb_id,
                            VALIDATOR_ID=self.validator_hotkey_reference,
                        )
                        event["s3_links"] = s3_links
                        event["validator_search_status"] = True  # simulation passed!
                        logger.success("✅✅ Simulation ran successfully! ✅✅")
                    except Exception as e:
                        logger.warning(f"Error uploading files to s3: {e}")
                        continue

                # break out of the loop if the simulation was successful
                break

            if tries == 10:
                logger.debug(f"Max tries reached for pdb_id {pdb_id} ❌❌")
                return event

    return event


async def try_prepare_dft_challenge(self) -> Dict:
    """Attempts to prepare a DFT challenge by sampling a random element from the dataset."""
    path = os.path.join(ROOT_DIR, "dsgdb9nsd.xyz")

    # Ideally, we are going to ping the endpoint ('https://springernature.figshare.com/ndownloader/files/3195389')
    elements = []  # This is going to be ~130,000 elements
    for file in os.listdir(path):
        if file.endswith(".xyz"):
            elements.append(file)

    # randomly sample a single element from the list
    element = random.choice(elements)

    # parse the element
    parsed_data = parse_custom_xyz(os.path.join(path, element))
    geometry = to_psi4_geometry_string(parsed_data["atoms"].to_numpy())

    return {
        "element": element,
        "geometry": geometry,
    }
