# The MIT License (MIT)
# Copyright © 2024 Macrocosmos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import re
import time
import random
import numpy as np
from loguru import logger
from itertools import chain
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
import asyncio

from async_timeout import timeout

from folding.utils.uids import get_random_uids
from folding.rewards.reward_pipeline import reward_pipeline
from folding.validators.forward import create_new_challenge, run_step, run_ping_step
from folding.validators.protein import Protein

# import base validator class which takes care of most of the boilerplate
from folding.store import Job, SQLiteJobStore
from folding.base.validator import BaseValidatorNeuron
from folding.utils.logging import log_event


class Validator(BaseValidatorNeuron):
    """
    Protein folding validator neuron. This neuron is responsible for validating the folding of proteins by querying the miners and updating the scores based on the responses.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.load_state()

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # TODO: Change the store to SQLiteJobStore if you want to use SQLite
        self.store = SQLiteJobStore()
        self.mdrun_args = self.parse_mdrun_args()

        # Sample all the uids on the network, and return only the uids that are non-valis.
        logger.info("Determining all miner uids...⏳")
        self.all_miner_uids: List = get_random_uids(
            self, k=int(self.metagraph.n), exclude=None
        ).tolist()

        self.wandb_run_start = None
        self.RSYNC_EXCEPTION_COUNT = 0

    def parse_mdrun_args(self) -> str:
        mdrun_args = ""

        # There are unwanted keys in mdrun_args, like __is_set. Remove all of these
        filtered_args = {
            key: value
            for key, value in self.config.mdrun_args.items()
            if not re.match(r"^[^a-zA-Z0-9]", key)
        }

        for arg, value in filtered_args.items():
            if value is not None:
                mdrun_args += f"-{arg} {value} "

        return mdrun_args

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

    async def forward(self, job: Job) -> dict:
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
            mdrun_args=self.mdrun_args,
            best_submitted_energy=job.best_loss,
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
        exclude_uids: List[int],
    ) -> List[int]:
        """Helper function to sample a batch of uids on the network, determine their serving status,
        and sample more until a desired number of uids is found.

        Args:
            num_uids_to_sample (int): The number of uids to sample.
            exclude_uids (List[int]): A list of uids that should be excluded from sampling.

        Returns:
            List[int]: A list of responding and free uids.
        """
        exclude_uids = []
        active_uids = await self.ping_all_miners(exclude_uids=exclude_uids)

        if len(active_uids) > num_uids_to_sample:
            return random.sample(active_uids, num_uids_to_sample)

        elif len(active_uids) <= num_uids_to_sample:
            return active_uids

    async def get_valid_uids(self) -> List[int]:
        """get valid uids to work on a job by sampling random uids and excluding active jobs.

        Returns:
            valid_uids: List of uids
        """
        active_jobs = self.store.get_queue(ready=False).queue
        active_hotkeys = [j.hotkeys for j in active_jobs]  # list of lists
        active_hotkeys = list(chain.from_iterable(active_hotkeys))
        exclude_uids = self.get_uids(hotkeys=active_hotkeys)

        valid_uids = await self.sample_random_uids(
            num_uids_to_sample=self.config.neuron.sample_size,
            exclude_uids=exclude_uids,
        )

        return valid_uids

    async def add_job(self, job_event: dict[str, Any], uids: List[int] = None) -> bool:
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
                    async with timeout(180):
                        logger.info(
                            f"setup_simulation for organic query: {job_event['pdb_id']}"
                        )
                        await protein.setup_simulation()
                        logger.success(
                            f"✅✅ organic {job_event['pdb_id']} simulation ran successfully! ✅✅"
                        )

                    if protein.init_energy > 0:
                        raise ValueError(
                            f"Initial energy is positive: {protein.init_energy}. Simulation failed."
                        )

                except Exception as e:
                    logger.error(f"Error in setting up organic query: {e}")

            logger.info(f"Inserting job: {job_event['pdb_id']}")
            self.store.insert(
                pdb=job_event["pdb_id"],
                ff=job_event["ff"],
                water=job_event["water"],
                box=job_event["box"],
                hotkeys=selected_hotkeys,
                epsilon=job_event["epsilon"],
                system_kwargs=job_event["system_kwargs"],
                event=job_event,
            )

            return True
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

    async def update_job(self, job: Job):
        """Updates the job status based on the event information

        TODO: we also need to remove hotkeys that have not participated for some time (dereg or similar)
        """

        top_reward = 0.80
        apply_pipeline = False

        # There could be hotkeys that have decided to stop serving. We need to remove them from the store.
        serving_hotkeys = []
        for ii, state in enumerate(job.event["response_miners_serving"]):
            if state:
                serving_hotkeys.append(job.hotkeys[ii])

        energies = torch.Tensor(job.event["energies"])
        rewards = torch.zeros(len(energies))  # one-hot per update step

        best_index = np.argmin(energies)
        best_loss = energies[best_index].item()  # item because it's a torch.tensor
        best_hotkey = serving_hotkeys[best_index]

        await job.update(
            hotkeys=serving_hotkeys,
            loss=best_loss,
            hotkey=best_hotkey,
        )

        # If no miners respond appropriately, the energies will be all zeros
        if (energies == 0).all():
            # All miners not responding but there is at least ONE miner that did in the past. Give them rewards.
            if job.best_loss < 0:
                apply_pipeline = True
                logger.warning(
                    f"Received all zero energies for {job.pdb} but stored best_loss < 0... Applying reward pipeline."
                )
        else:
            apply_pipeline = True
            logger.success("Non-zero energies received. Applying reward pipeline.")

        if apply_pipeline:
            rewards: torch.Tensor = await reward_pipeline(
                energies=energies,
                rewards=rewards,
                top_reward=top_reward,
                job=job,
            )

            uids = self.get_uids(hotkeys=job.hotkeys)
            await self.update_scores(
                rewards=rewards,
                uids=uids,  # pretty confident these are in the correct order.
            )
        else:
            logger.warning(
                f"All energies zero for job {job.pdb} and job has never been updated... Skipping"
            )

        # Finally, we update the job in the store regardless of what happened.
        self.store.update(job=job)

        async def prepare_event_for_logging(event: Dict):
            for key, value in event.items():
                if isinstance(value, pd.Timedelta):
                    event[key] = value.total_seconds()
            return event

        event = job.to_dict()
        simulation_event = event.pop("event")  # contains information from hp search
        merged_events = simulation_event | event  # careful: this overwrites.

        merged_events["rewards"] = list(
            rewards.numpy()
        )  # add the rewards to the logging event.

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
            logger.error(f"Protein.from_job returns NONE for protein {job.pdb}")

        # Remove these keys from the log because they polute the terminal.
        log_event(
            self,
            event=event,
            pdb_location=pdb_location,
            folded_protein_location=folded_protein_location,
        )

        merged_events.pop("checked_energy")
        merged_events.pop("miner_energy")
        logger.success(f"Event information: {merged_events}")

        if protein is not None and job.active is False:
            protein.remove_pdb_directory()

    async def sync_loop(self):
        logger.info("Starting sync loop.")
        while True:
            self.sync()
            seconds_per_block = 12
            await asyncio.sleep(self.config.neuron.epoch_length * seconds_per_block)

    async def create_synthetic_jobs(self):
        """
        Creates jobs and adds them to the queue.
        """

        while True:
            try:
                logger.info("Starting job creation loop.")
                queue = self.store.get_queue(ready=False)
                if queue.qsize() < self.config.neuron.queue_size:
                    # Potential situation where (sample_size * queue_size) > available uids on the metagraph.
                    # Therefore, this product must be less than the number of uids on the metagraph.
                    if (
                        self.config.neuron.sample_size * self.config.neuron.queue_size
                    ) > self.metagraph.n:
                        raise ValueError(
                            f"sample_size * queue_size must be less than the number of uids on the metagraph ({self.metagraph.n})."
                        )

                    logger.debug(f"✅ Creating jobs! ✅")
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
                logger.error(f"Error in create_jobs: {e}")

            await asyncio.sleep(self.config.neuron.synthetic_job_interval)

    async def update_jobs(self):
        while True:
            try:
                # Wait at the beginning of update_jobs since we want to avoid attemping to update jobs before we get data back.
                await asyncio.sleep(self.config.neuron.update_interval)

                logger.info("Updating jobs.")
                logger.info(f"step({self.step}) block({self.block})")

                for job in self.store.get_queue(ready=False).queue:
                    # Remove any deregistered hotkeys from current job. This will update the store when the job is updated.
                    if not job.check_for_available_hotkeys(self.metagraph.hotkeys):
                        self.store.update(job=job)
                        continue

                    # Here we straightforwardly query the workers associated with each job and update the jobs accordingly
                    job_event = await self.forward(job=job)

                    # If we don't have any miners reply to the query, we will make it inactive.
                    if len(job_event["energies"]) == 0:
                        job.active = False
                        self.store.update(job=job)
                        continue

                    if isinstance(job.event, str):
                        job.event = eval(job.event)  # if str, convert to dict.

                    job.event.update(job_event)
                    # Determine the status of the job based on the current energy and the previous values (early stopping)
                    # Update the DB with the current status
                    await self.update_job(job=job)
            except Exception as e:
                logger.error(f"Error in update_jobs: {e}")

            self.step += 1

            logger.info(
                f"Sleeping {self.config.neuron.update_interval} seconds before next job update loop."
            )

    async def __aenter__(self):
        self.loop.create_task(self.sync_loop())
        self.loop.create_task(self.create_synthetic_jobs())
        self.loop.create_task(self.update_jobs())
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
            logger.debug("Stopped")


async def main():
    async with Validator() as v:
        while v.is_running and not v.should_exit:
            await asyncio.sleep(30)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
