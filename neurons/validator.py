# The MIT License (MIT)
# Copyright © 2024 Macrocosmos

import os
import time
import random
import asyncio
import traceback

from itertools import chain
from datetime import datetime
from typing import Any, Dict, List, Tuple

import netaddr
import requests
import torch
import numpy as np
import pandas as pd
from async_timeout import timeout
import tenacity


import folding.utils.constants as c
from folding.base.reward import BatchRewardInput
from folding.base.reward import BaseReward, RewardEvent
from folding.base.validator import BaseValidatorNeuron
from folding.rewards.md_rewards import REWARD_REGISTRY

# import base validator class which takes care of most of the boilerplate
from folding.store import Job, SQLiteJobStore
from folding.utils.logger import logger
from folding.utils.logging import log_event
from folding.utils.uids import get_all_miner_uids
from folding.utils.s3_utils import (
    upload_output_to_s3,
    upload_to_s3,
    DigitalOceanS3Handler,
)
from folding.validators.forward import create_new_challenge, run_ping_step, run_step
from folding.validators.protein import Protein
from folding.registries.miner_registry import MinerRegistry
from folding.organic.api import start_organic_api


class Validator(BaseValidatorNeuron):
    """
    Protein folding validator neuron. This neuron is responsible for validating the folding of proteins by querying the miners and updating the scores based on the responses.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.load_state()

        # Sample all the uids on the network, and return only the uids that are non-valis.
        logger.info("Determining all miner uids...⏳")
        self.all_miner_uids: List = get_all_miner_uids(self)

        # If we do not have any miner registry saved to the machine, create.
        if not hasattr(self, "miner_registry"):
            self.miner_registry = MinerRegistry(miner_uids=self.all_miner_uids)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        self.store = SQLiteJobStore()
        self.wandb_run_start = None
        self.RSYNC_EXCEPTION_COUNT = 0

        self.validator_hotkey_reference = self.wallet.hotkey.ss58_address[:8]

        # The last time that we checked the global job pool.
        self.last_time_checked = datetime.now()
        self.last_time_created_jobs = datetime.now()

        if not self.config.s3.off:
            try:
                self.handler = DigitalOceanS3Handler(
                    bucket_name=self.config.s3.bucket_name,
                )
            except ValueError as e:
                raise f"Failed to create S3 handler, check your .env file: {e}"

    def get_uids(self, hotkeys: List[str]) -> List[int]:
        """Returns the uids corresponding to the hotkeys.
        It is possible that some hotkeys have been dereg'd,
        so we need to check for them in the metagraph.

        Args:
            hotkeys (List[str]): List of hotkeys

        Returns:
            List[int]: List of uids
        """
        return [
            self.metagraph.hotkeys.index(hotkey)
            for hotkey in hotkeys
            if hotkey in self.metagraph.hotkeys
            and self.metagraph.axons[self.metagraph.hotkeys.index(hotkey)].is_serving
        ]

    async def forward(self, job: Job, first: bool = False) -> dict:
        """Carries out a query to the miners to check their progress on a given job (pdb) and updates the job status based on the results.

        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores

        Args:
            job (Job): Job object containing the pdb and hotkeys
        """

        protein = await Protein.from_job(job=job, config=self.config.protein)

        uids = self.get_uids(hotkeys=job.hotkeys)

        logger.info("Running run_step...⏳")
        return await run_step(
            self,
            protein=protein,
            uids=uids,
            timeout=self.config.neuron.timeout,
            job_id=job.job_id,
            best_submitted_energy=job.best_loss,
            job_type=job.job_type,
            first=first,
        )

    async def ping_all_miners(
        self,
        exclude_uids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Sample ALL (non-excluded) miner uids and return the list of uids
        that can serve jobs.

        Args:
            exclude_uids (List[int]): uids to exclude from the uids search

        Returns:
           Tuple(List,List): Lists of active and inactive uids
        """

        current_miner_uids = list(
            set(self.all_miner_uids).difference(set(exclude_uids))
        )

        ping_report = await run_ping_step(
            self, uids=current_miner_uids, timeout=self.config.neuron.ping_timeout
        )
        can_serve = ping_report["miner_status"]  # list of booleans

        active_uids = np.array(current_miner_uids)[can_serve].tolist()
        return active_uids

    async def sample_random_uids(
        self,
        num_uids_to_sample: int,
        exclude_uids: List[int] = None,
    ) -> List[int]:
        """Helper function to sample a batch of uids on the network, determine their serving status,
        and sample more until a desired number of uids is found.

        Args:
            num_uids_to_sample (int): The number of uids to sample.
            exclude_uids (List[int], optional): List of uids to exclude from the sampling. Defaults to None.

        Returns:
            List[int]: A list of random uids.
        """

        if exclude_uids is not None:
            all_miner_uids = list(
                set(self.all_miner_uids).difference(set(exclude_uids))
            )
        else:
            all_miner_uids = self.all_miner_uids

        return random.sample(all_miner_uids, num_uids_to_sample)

    async def get_valid_uids(self) -> List[int]:
        """get valid uids to work on a job by sampling random uids and excluding active jobs.

        Returns:
            valid_uids: List of uids
        """
        active_jobs = self.store.get_queue(
            ready=False, validator_hotkey=self.wallet.hotkey.ss58_address
        ).queue
        active_hotkeys = [j.hotkeys for j in active_jobs]  # list of lists
        active_hotkeys = list(chain.from_iterable(active_hotkeys))
        exclude_uids = self.get_uids(hotkeys=active_hotkeys)

        valid_uids = await self.sample_random_uids(
            num_uids_to_sample=self.config.neuron.sample_size, exclude_uids=exclude_uids
        )

        return valid_uids

    async def add_job(
        self, job_event: dict[str, Any], uids: List[int] = None, protein: Protein = None
    ) -> bool:
        """Add a job to the job store while also checking to see what uids can be assigned to the job.
        If uids are not provided, then the function will sample random uids from the network.

        Args:
            job_event (dict[str, Any]): parameters that are needed to make the job.
            uids (List[int], optional): List of uids that can be assigned to the job. Defaults to None.
        """
        start_time = time.time()

        if uids is not None:
            valid_uids = uids
        else:
            valid_uids = await self.get_valid_uids()

        job_event["uid_search_time"] = time.time() - start_time
        selected_hotkeys = [self.metagraph.hotkeys[uid] for uid in valid_uids]

        if len(valid_uids) >= self.config.neuron.sample_size:
            # If the job is organic, we still need to run the setup simulation to create the files needed for the job.
            if job_event.get("is_organic"):
                self.config.protein.input_source = job_event["source"]
                protein = Protein(**job_event, config=self.config.protein)

                try:
                    job_event["pdb_id"] = job_event["pdb_id"]
                    job_event["job_type"] = "OrganicMD"
                    job_event["pdb_complexity"] = [dict(protein.pdb_complexity)]
                    job_event["init_energy"] = protein.init_energy
                    job_event["epsilon"] = protein.epsilon
                    job_event["s3_links"] = {
                        "testing": "testing"
                    }  # overwritten below if s3 logging is on.
                    async with timeout(300):
                        logger.info(
                            f"setup_simulation for organic query: {job_event['pdb_id']}"
                        )
                        await protein.setup_simulation()
                        logger.success(
                            f"✅✅ organic {job_event['pdb_id']} simulation ran successfully! ✅✅"
                        )

                    if protein.init_energy > 0:
                        logger.error(
                            f"Initial energy is positive: {protein.init_energy}. Simulation failed."
                        )
                        job_event["active"] = False

                    if not self.config.s3.off:
                        try:
                            logger.info(f"Uploading to {self.handler.bucket_name}")
                            s3_links = await upload_to_s3(
                                handler=self.handler,
                                pdb_location=protein.pdb_location,
                                simulation_cpt=protein.simulation_cpt,
                                validator_directory=protein.validator_directory,
                                pdb_id=job_event["pdb_id"],
                                VALIDATOR_ID=self.validator_hotkey_reference,
                            )
                            job_event["s3_links"] = s3_links
                            logger.success("✅✅ Simulation ran successfully! ✅✅")
                        except Exception as e:
                            logger.error(f"Error in uploading to S3: {e}")
                            logger.error("❌❌ Simulation failed! ❌❌")
                            job_event["active"] = False

                except Exception as e:
                    job_event["active"] = False
                    logger.error(f"Error in setting up organic query: {e}")

            logger.info(f"Inserting job: {job_event['pdb_id']}")
            try:
                job = self.store.upload_job(
                    event=job_event,
                    hotkeys=selected_hotkeys,
                    keypair=self.wallet.hotkey,
                    gjp_address=self.config.neuron.gjp_address,
                )

                job_event["job_id"] = await self.store.confirm_upload(job_id=job.job_id)

                if hasattr(job_event, "job_id") and job_event["job_id"] is None:
                    raise ValueError("job_id is None")

                logger.success("Job was uploaded successfully!")
                if job.active:
                    await self.forward(job=job, first=True)

                self.last_time_created_jobs = datetime.now()

                # TODO: return job_id
                return True
            except Exception as e:
                logger.warning(f"Error uploading job: {traceback.format_exc()}")
                job_event["job_id"] = None

                return False

        else:
            logger.warning(
                f"Not enough available uids to create a job. Requested {self.config.neuron.sample_size}, but number of valid uids is {len(valid_uids)}... Skipping until available"
            )
            return False

    async def add_k_synthetic_jobs(self, k: int):
        """Creates new synthetic jobs and assigns them to available workers. Updates DB with new records.
        Each "job" is an individual protein folding challenge that is distributed to the miners.

        Args:
            k (int): The number of jobs create and distribute to miners.
        """

        # Deploy K number of unique pdb jobs, where each job gets distributed to self.config.neuron.sample_size miners
        for ii in range(k):
            logger.info(f"Adding job: {ii+1}/{k}")

            # This will change on each loop since we are submitting a new pdb to the batch of miners
            exclude_pdbs = self.store.get_all_pdbs()
            job_event: Dict = await create_new_challenge(self, exclude=exclude_pdbs)

            await self.add_job(job_event=job_event)
            await asyncio.sleep(0.01)

    async def update_scores_wrapper(
        self, rewards: torch.FloatTensor, hotkeys: List[str]
    ):
        """Wrapper function to update the scores of the miners based on the rewards they received."""
        uids = self.get_uids(hotkeys=hotkeys)
        await self.update_scores(
            rewards=rewards,
            uids=uids,
        )

    async def update_job(self, job: Job):
        """Updates the job status based on the event information

        TODO: we also need to remove hotkeys that have not participated for some time (dereg or similar)
        """

        apply_pipeline = False
        energies = torch.Tensor(job.event["energies"])

        for uid, reason in zip(job.event["uids"], job.event["reason"]):
            # If there is an exploit on the cpt file detected via the state-checkpoint, reduce score.
            if reason == "state-checkpoint":
                logger.warning(
                    f"Reducing uid {uid} score, State-checkpoint check failed."
                )
                self.scores[uid] = 0.5 * self.scores[uid]

            credibility = [0.0] if reason != "valid" else [1.0]
            self.miner_registry.add_credibilities(
                miner_uid=uid, task=job.job_type, credibilities=credibility
            )
            self.miner_registry.update_credibility(miner_uid=uid, task=job.job_type)

        best_index = np.argmin(energies)
        best_loss = energies[best_index].item()  # item because it's a torch.tensor
        best_hotkey = job.hotkeys[best_index]

        await job.update(
            loss=best_loss,
            hotkey=best_hotkey,
        )

        # If no miners respond appropriately, the energies will be all zeros
        if (energies == 0).all():
            # All miners not responding but there is at least ONE miner that did in the past. Give them rewards.
            if job.best_loss < 0:
                apply_pipeline = True
                logger.warning(
                    f"Received all zero energies for {job.pdb_id} but stored best_loss < 0... Applying reward pipeline."
                )
        else:
            apply_pipeline = True
            logger.success("Non-zero energies received. Applying reward pipeline.")

        if apply_pipeline:
            model: BaseReward = REWARD_REGISTRY[job.job_type](priority=job.priority)
            reward_event: RewardEvent = await model.forward(
                data=BatchRewardInput(
                    energies=energies,
                    top_reward=c.TOP_SYNTHETIC_MD_REWARD,
                    job=job,
                ),
            )

            job.computed_rewards = reward_event.rewards.numpy().tolist()

        else:
            job.computed_rewards = [0.0] * len(job.hotkeys)
            logger.warning(
                f"All energies zero for job {job.pdb_id} and job has never been updated... Skipping"
            )

        async def prepare_event_for_logging(event: Dict):
            for key, value in event.items():
                if isinstance(value, pd.Timedelta):
                    event[key] = value.total_seconds()
            return event

        event = job.model_dump()
        simulation_event = event.pop("event")  # contains information from hp search
        merged_events = simulation_event | event  # careful: this overwrites.

        event = await prepare_event_for_logging(merged_events)

        # If the job is finished, remove the pdb directory
        pdb_location = None
        folded_protein_location = None
        protein = await Protein.from_job(job=job, config=self.config.protein)

        if protein is not None:
            if job.active is True:
                if event["updated_count"] == 1:
                    pdb_location = protein.pdb_location

                protein.get_miner_data_directory(event["best_hotkey"])
                folded_protein_location = os.path.join(
                    protein.miner_data_directory, f"{protein.pdb_id}_folded.pdb"
                )

        else:
            logger.error(f"Protein.from_job returns NONE for protein {job.pdb_id}")

        # Remove these keys from the log because they polute the terminal.
        log_event(
            self,
            event=event,
            pdb_location=pdb_location,
            folded_protein_location=folded_protein_location,
        )

        # Only upload the best .cpt files to S3 if the job is inactive
        if job.active is False:
            output_links = []
            for idx, best_cpt_file in enumerate(job.event["best_cpt"]):
                # If the best_cpt_file is empty, we will append an empty string to the output_links list.
                if best_cpt_file == "":
                    output_links.append("")
                    continue

                output_link = await upload_output_to_s3(
                    handler=self.handler,
                    output_file=best_cpt_file,
                    pdb_id=job.pdb_id,
                    miner_hotkey=job.hotkeys[idx],
                    VALIDATOR_ID=self.validator_hotkey_reference,
                )

                output_links.append(output_link)

            job.best_cpt_links = output_links

        # Finally, we update the job in the store regardless of what happened.
        self.store.update_gjp_job(
            job=job,
            gjp_address=self.config.neuron.gjp_address,
            keypair=self.wallet.hotkey,
            job_id=job.job_id,
        )

        merged_events.pop("checked_energy")
        merged_events.pop("miner_energy")
        logger.success(f"Event information: {merged_events}")

        if protein is not None and job.active is False:
            protein.remove_pdb_directory()

    async def create_synthetic_jobs(self):
        """
        Creates jobs and adds them to the queue.
        """

        while True:
            try:
                logger.info("Starting job creation loop.")
                queue = self.store.get_queue(
                    ready=False, validator_hotkey=self.wallet.hotkey.ss58_address
                )
                if queue.qsize() < self.config.neuron.queue_size:
                    # Potential situation where (sample_size * queue_size) > available uids on the metagraph.
                    # Therefore, this product must be less than the number of uids on the metagraph.
                    if (
                        self.config.neuron.sample_size * self.config.neuron.queue_size
                    ) > self.metagraph.n:
                        raise ValueError(
                            f"sample_size * queue_size must be less than the number of uids on the metagraph ({self.metagraph.n})."
                        )

                    logger.debug("✅ Creating jobs! ✅")
                    # Here is where we select, download and preprocess a pdb
                    # We also assign the pdb to a group of workers (miners), based on their workloads

                    await self.add_k_synthetic_jobs(
                        k=self.config.neuron.queue_size - queue.qsize()
                    )

                    logger.info(
                        f"Sleeping {self.config.neuron.synthetic_job_interval} seconds before next job creation loop."
                    )
                else:
                    logger.info(
                        "Job queue is full. Sleeping 60 seconds before next job creation loop."
                    )

            except Exception as e:
                logger.error(f"Error in create_jobs: {traceback.format_exc()}")

            await asyncio.sleep(self.config.neuron.synthetic_job_interval)

    async def update_jobs(self):
        while True:
            try:
                # Wait at the beginning of update_jobs since we want to avoid attemping to update jobs before we get data back.
                await asyncio.sleep(self.config.neuron.update_interval)

                logger.info("Updating jobs.")

                for job in self.store.get_queue(
                    ready=True, validator_hotkey=self.wallet.hotkey.ss58_address
                ).queue:
                    # Here we straightforwardly query the workers associated with each job and update the jobs accordingly
                    job_event = await self.forward(job=job)

                    # If we don't have any miners reply to the query, we will make it inactive.
                    if len(job_event["energies"]) == 0:
                        job.active = False
                        self.store.update_gjp_job(
                            job=job,
                            gjp_address=self.config.neuron.gjp_address,
                            keypair=self.wallet.hotkey,
                            job_id=job.job_id,
                        )
                        continue

                    if isinstance(job.event, str):
                        job.event = eval(job.event)  # if str, convert to dict.

                    job.event.update(job_event)
                    # Determine the status of the job based on the current energy and the previous values (early stopping)
                    # Update the DB with the current status
                    await self.update_job(job=job)
                logger.info(f"step({self.step}) block({self.block})")

            except Exception as e:
                logger.error(f"Error in update_jobs: {traceback.format_exc()}")

            self.step += 1

            logger.info(
                f"Sleeping {self.config.neuron.update_interval} seconds before next job update loop."
            )

    async def read_and_update_rewards(self):
        """Read the rewards from the inactive jobs and update the scores of the miners
        using EMA.
        """
        inactive_jobs_queue = self.store.get_inactive_queue(
            last_time_checked=self.last_time_checked.strftime("%Y-%m-%dT%H:%M:%S")
        )
        self.last_time_checked = datetime.now()

        if inactive_jobs_queue.qsize() == 0:
            logger.info("No inactive jobs to update.")
            return

        logger.info(f"number of jobs to eval: {inactive_jobs_queue.qsize()}")
        while (
            not inactive_jobs_queue.qsize() == 0
        ):  # recommended to use qsize() instead of empty()
            inactive_job = inactive_jobs_queue.get()
            logger.info(f"Updating scores for job: {inactive_job.pdb_id}")
            if inactive_job.computed_rewards is None:
                logger.warning(
                    f"Computed rewards are None for job: {inactive_job.pdb_id}"
                )
                continue

            await self.update_scores_wrapper(
                rewards=torch.Tensor(inactive_job.computed_rewards),
                hotkeys=inactive_job.hotkeys,
            )
            await asyncio.sleep(0.01)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    )
    async def start_organic_api(self):
        try:
            logger.info("Starting organic API")
            external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
            netaddr.IPAddress(external_ip)
            previous_commit = self.subtensor.get_commitment(
                self.config.netuid, self.uid
            )
            logger.info(f"Previous commitment: {previous_commit}")
            commitment = f"http://{external_ip}:{self.config.neuron.organic_api.port}"

            if previous_commit != commitment:
                serve_success = self.subtensor.commit(
                    wallet=self.wallet,
                    netuid=self.config.netuid,
                    data=commitment,
                )
                logger.debug(f"Serve success: {serve_success}")
            else:
                logger.info("No need to commit again")

            await start_organic_api(self._organic_scoring, self.config)
        except Exception as e:
            logger.error(f"Error in start_organic_api: {traceback.format_exc()}")
            raise e

    async def reward_loop(self):
        logger.info("Starting reward loop.")
        while True:
            try:
                await asyncio.sleep(60)
                await self.read_and_update_rewards()
            except Exception as e:
                logger.error(f"Error in reward_loop: {traceback.format_exc()}")

    async def sync_loop(self):
        logger.info("Starting sync loop.")
        while True:
            seconds_per_block = 12
            try:
                await asyncio.sleep(self.config.neuron.epoch_length * seconds_per_block)
                self.sync()
            except Exception as e:
                logger.error(f"Error in sync_loop: {traceback.format_exc()}")

    async def monitor_db(self):
        """
        Monitors the database for any changes.
        """
        while True:
            try:
                await asyncio.sleep(300)
                try:
                    outdated = await self.store.monitor_db()
                except Exception as e:
                    logger.error(f"Error in monitor_db: {traceback.format_exc()}")
                    await self.start_rqlite()

                if outdated:
                    logger.error("Database is outdated. Restarting rqlite.")
                    await self.start_rqlite()
                else:
                    logger.debug("Database is up-to-date.")
            except Exception as e:
                logger.error(f"Error in monitor_db: {traceback.format_exc()}")

    async def monitor_validator(self):
        while True:
            await asyncio.sleep(3600)
            # if no jobs have been created in the last 12 hours, shutdown the validator
            if (datetime.now() - self.last_time_created_jobs).seconds > 43200:
                logger.error(
                    "No jobs have been created in the last 12 hours. Restarting validator."
                )
                self.should_exit = True

            block_difference = (
                self.metagraph.block - self.metagraph.neurons[self.uid].last_update
            )
            if block_difference > 3 * self.config.neuron.epoch_length:
                logger.error(
                    f"Haven't set blocks in {block_difference} blocks. Restarting validator."
                )
                self.should_exit = True

    async def __aenter__(self):
        await self.start_rqlite()
        await asyncio.sleep(10)  # Wait for rqlite to start

        self.loop.create_task(self.sync_loop())
        self.loop.create_task(self.update_jobs())
        self.loop.create_task(self.create_synthetic_jobs())
        self.loop.create_task(self.reward_loop())
        self.loop.create_task(self.monitor_db())
        if self.config.neuron.organic_enabled:
            logger.info("Starting organic scoring loop.")
            self.loop.create_task(self._organic_scoring.start_loop())
            self.loop.create_task(self.start_organic_api())
        self.loop.create_task(self.monitor_validator())
        self.is_running = True
        logger.debug("Starting validator in background thread.")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.is_running = False
            self.loop.stop()
            logger.debug("Stopped")


async def main():
    async with Validator() as v:
        while v.is_running and not v.should_exit:
            await asyncio.sleep(30)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
