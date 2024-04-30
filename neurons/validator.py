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


import time
import numpy as np
from typing import Dict
import bittensor as bt

from folding.store import PandasJobStore
from folding.utils.uids import get_random_uids
from folding.validators.forward import forward, create_new_challenge, run_step
from folding.validators.protein import Protein

# import base validator class which takes care of most of the boilerplate
from folding.store import Job
from folding.base.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    """
    Protein folding validator neuron. This neuron is responsible for validating the folding of proteins by querying the miners and updating the scores based on the responses.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.warning(
            "VALIDATOR LOAD_STATE DOES NOT WORK... SKIPPING BaseValidatorNeuron.load_state()"
        )

        # TODO: Change the store to SQLiteJobStore if you want to use SQLite
        self.store = PandasJobStore(db_path=self.config.neuron.db_path_location)

    async def forward(self, job: Job):
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

        # TODO: the command below should correctly prepare the md_inputs to point at the current best gro files (+ others)
        protein = Protein.from_pdb(pdb_id=job.pdb)
        uids = [self.metagaph.hotkeys.index(hotkey) for hotkey in job.hotkeys]
        # query the miners and get the rewards for their responses
        # Check check_uid_availability to ensure that the hotkeys are valid and active
        return await run_step(
            self,
            protein=protein,
            uids=uids,
            timeout=self.config.neuron.timeout,
        )

    def add_jobs(self, k):
        """Creates new jobs and assigns them to available workers. Updates DB with new records"""
        exclude_pdbs = self.store.get_queue(ready=False).index.tolist()

        for i in range(k):
            # selects a new pdb, downloads data, preprocesses and gets hyperparams.
            job_event: Dict = create_new_challenge(self, exclude=exclude_pdbs)

            # assign workers to the job (hotkeys)
            active_jobs = self.store.get_queue(ready=False)
            # exclude hotkeys that are already in use by this validator (a single job)
            active_hotkeys = active_jobs["hotkeys"].explode().unique().tolist()
            exclude_uids = [
                self.metagaph.hotkeys.index(hotkey) for hotkey in active_hotkeys
            ]

            uids = get_random_uids(
                self, self.config.neuron.sample_size, exclude=exclude_uids
            )

            selected_hotkeys = [self.metagraph.hotkeys[uid] for uid in uids]

            # TODO: We can pass other custom stuff like update_interval, max_time_no_improvement and min_updates to control length
            # update_interval = 60 * np.log10(job['pdb_length'])
            # min_updates = 10
            # max_time_no_improvement = min_updates * update_interval

            self.store.insert(pdb=job_event["pdb_id"], hotkeys=selected_hotkeys)

    def update_job(self, job: Job, event: Dict):
        """Updates the job status based on the event information

        TODO: we also need to remove hotkeys that have not participated for some time (dereg or similar)
        """

        # set updated_at to most recent query
        # maybe attach some metadata about the specific checkpoints that are the current head
        loss = event["best_loss"]
        hotkey = event["best_hotkey"]
        commit_hash = ""  # For next time
        gro_hash = ""  # For next time

        # check if early stopping criteria is met
        job.update(loss=loss, hotkey=hotkey, commit_hash=commit_hash, gro_hash=gro_hash)

        self.store.update(job=job)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as v:
        while v.is_running and not v.should_exit:
            bt.logging.info(
                f"Validator running:: network: {v.subtensor.network} | block: {v.block} | step: {v.step} | uid: {v.uid} | last updated: {v.block-v.metagraph.last_update[v.uid]} | vtrust: {v.metagraph.validator_trust[v.uid]:.3f} | emission {v.metagraph.emission[v.uid]:.3f}"
            )
            time.sleep(15)
