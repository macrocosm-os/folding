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

import re
import time
import numpy as np
from typing import Dict
from itertools import chain
from typing import List

import torch
import pandas as pd
import bittensor as bt

from folding.store import PandasJobStore
from folding.utils.uids import get_random_uids
from folding.rewards.reward_pipeline import reward_pipeline
from folding.validators.forward import create_new_challenge, run_step
from folding.validators.protein import Protein

# import base validator class which takes care of most of the boilerplate
from folding.store import Job
from folding.base.validator import BaseValidatorNeuron
from folding.utils.logging import log_event


class Validator(BaseValidatorNeuron):
    """
    Protein folding validator neuron. This neuron is responsible for validating the folding of proteins by querying the miners and updating the scores based on the responses.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        # TODO: Change the store to SQLiteJobStore if you want to use SQLite
        self.store = PandasJobStore()
        self.mdrun_args = self.parse_mdrun_args()

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

    def forward(self, job: Job) -> dict:
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

        protein = Protein.from_job(job=job, config=self.config.protein)

        uids = [self.metagraph.hotkeys.index(hotkey) for hotkey in job.hotkeys]
        # query the miners and get the rewards for their responses
        # Check check_uid_availability to ensure that the hotkeys are valid and active
        bt.logging.info("⏰ Waiting for miner responses ⏰")
        return run_step(
            self,
            protein=protein,
            uids=uids,
            timeout=self.config.neuron.timeout,
            mdrun_args=self.mdrun_args,
        )

    def get_pdbs_to_exclude(self) -> List[str]:
        # Set of pdbs that are currently in the process of running + old submitted simulations.
        return list(self.store._db.index)

    def add_jobs(self, k: int):
        """Creates new jobs and assigns them to available workers. Updates DB with new records.
        Each "job" is an individual protein folding challenge that is distributed to the miners.

        Args:
            k (int): The number of jobs create and distribute to miners.
        """

        # Deploy K number of unique pdb jobs, where each job gets distributed to self.config.neuron.sample_size miners
        for ii in range(k):
            bt.logging.info(f"Adding job: {ii+1}/{k}")

            # This will change on each loop since we are submitting a new pdb to the batch of miners
            exclude_pdbs = self.get_pdbs_to_exclude()

            # selects a new pdb, downloads data, preprocesses and gets hyperparams.
            job_event: Dict = create_new_challenge(self, exclude=exclude_pdbs)

            # assign workers to the job (hotkeys)
            active_jobs = self.store.get_queue(ready=False).queue
            active_hotkeys = [j.hotkeys for j in active_jobs]  # list of lists
            active_hotkeys = list(chain.from_iterable(active_hotkeys))
            exclude_uids = [
                self.metagraph.hotkeys.index(hotkey) for hotkey in active_hotkeys
            ]

            uids = get_random_uids(
                self, self.config.neuron.sample_size, exclude=exclude_uids
            )

            selected_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]

            # TODO: We can pass other custom stuff like update_interval, max_time_no_improvement and min_updates to control length
            # update_interval = 60 * np.log10(job['pdb_length'])
            # min_updates = 10
            # max_time_no_improvement = min_updates * update_interval

            if len(selected_hotkeys) > 0:
                self.store.insert(
                    pdb=job_event["pdb_id"],
                    ff=job_event["ff"],
                    water=job_event["water"],
                    box=job_event["box"],
                    hotkeys=selected_hotkeys,
                    epsilon=job_event["epsilon"],
                    event=job_event,
                )

    def update_job(self, job: Job):
        """Updates the job status based on the event information

        TODO: we also need to remove hotkeys that have not participated for some time (dereg or similar)
        """

        top_reward = 0.80
        apply_pipeline = False

        energies = torch.Tensor(job.event["energies"])
        rewards = torch.zeros(len(energies))  # one-hot per update step

        # TODO: we need to get the commit and gro hashes from the best hotkey
        commit_hash = ""  # For next time
        gro_hash = ""  # For next time

        # If no miners respond appropriately, the energies will be all zeros
        if (energies == 0).all():
            # All miners not responding but there is at least ONE miner that did in the past. Give them rewards.
            if job.best_loss < np.inf:
                apply_pipeline = True
                bt.logging.warning(
                    f"Received all zero energies for {job.pdb} but stored best_loss < np.inf... Giving rewards."
                )

            if (
                pd.Timestamp.now().floor("s") - job.created_at
                > job.max_time_no_improvement
            ):
                if isinstance(job.best_loss_at, pd._libs.tslibs.nattype.NaTType):
                    # means that nothing has been sampled from any miners and not updated.
                    # apply_pipeline could be True here, so we still apply rewards one last time.
                    bt.logging.warning(
                        f"Job {job.pdb} has not been updated since creation. Removing from queue."
                    )
                    job.active = False
        else:
            apply_pipeline = True
            bt.logging.success("Non-zero energies received. Applying reward pipeline.")

        if apply_pipeline:
            best_index = np.argmin(energies)
            best_loss = energies[best_index].item()  # item because it's a torch.tensor
            best_hotkey = job.hotkeys[best_index]

            # This does a bunch of important things:
            # 1. checks if best loss in this round is the best loss overall and updates the best_hotkey and hashes accordingly
            # 2. Potentially applies early stopping criteria
            # 3. Ensures that all timestamps and counters are updated correctly
            job.update(
                loss=best_loss,
                hotkey=best_hotkey,
                commit_hash=commit_hash,
                gro_hash=gro_hash,
            )

            rewards: torch.Tensor = reward_pipeline(
                energies=energies, rewards=rewards, top_reward=top_reward, job=job
            )

            uids = [self.metagraph.hotkeys.index(hotkey) for hotkey in job.hotkeys]
            self.update_scores(
                rewards=rewards,
                uids=uids,  # pretty confident these are in the correct order.
            )
        else:
            bt.logging.warning(
                f"All energies zero for job {job.pdb} and job has never been updated... Skipping"
            )

        # Finally, we update the job in the store regardless of what happened.
        self.store.update(job=job)

        def prepare_event_for_logging(event: Dict):
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

        bt.logging.success(f"Event information: {merged_events}")
        log_event(self, event=prepare_event_for_logging(merged_events))


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as v:
        while v.is_running and not v.should_exit:
            # bt.logging.info(
            #     f"Validator running:: network: {v.subtensor.network} | block: {v.block} | step: {v.step} | uid: {v.uid} | last updated: {v.block-v.metagraph.last_update[v.uid]} | vtrust: {v.metagraph.validator_trust[v.uid]:.3f} | emission {v.metagraph.emission[v.uid]:.3f}"
            # )
            time.sleep(30)
