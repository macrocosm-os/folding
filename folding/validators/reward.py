import time
from typing import List, Dict

import numpy as np
import bittensor as bt

from folding.validators.protein import Protein
from folding.utils.ops import get_energy_from_simulation
from folding.protocol import JobSubmissionSynapse


class RewardPipeline:
    def __init__(
        self, protein: Protein, responses: List[JobSubmissionSynapse], uids: List[int]
    ):
        self.protein = protein
        self.responses = responses
        self.uids = uids

        self.energies = np.zeros(len(uids))

        self.event = {}
        self.event["is_valid"] = [False] * len(uids)
        self.event["checked_energy"] = [0] * len(uids)
        self.event["reported_energy"] = [0] * len(uids)
        self.event["miner_energy"] = [0] * len(uids)
        self.event["rmsds"] = [0] * len(uids)
        self.event["process_md_output_time"] = [0] * len(uids)
        self.event["is_run_valid"] = [0] * len(uids)

        self.packages = [None] * len(uids)
        self.miner_states = [None] * len(uids)

    def process_energies(self):
        for i, (uid, resp) in enumerate(zip(self.uids, self.responses)):
            try:
                start_time = time.time()

                can_process, package = self.protein.process_md_output(
                    md_output=resp.md_output,
                    hotkey=resp.axon.hotkey,
                    state=resp.miner_state,
                    seed=resp.miner_seed,
                )

                self.packages[i] = package
                self.miner_states[i] = resp.miner_state

                self.event["process_md_output_time"][i] = time.time() - start_time

                if not can_process:
                    continue

                if resp.dendrite.status_code != 200:
                    bt.logging.info(
                        f"uid {uid} responded with status code {resp.dendrite.status_code}"
                    )
                    continue

                energy = get_energy_from_simulation(package["simulation"])

                # Catching edge case where energy is 0
                if energy == 0:
                    continue

                self.energies[i] = energy

            except Exception as E:
                # If any of the above methods have an error, we will catch here.
                bt.logging.error(
                    f"Failed to parse miner data for uid {uid} with error: {E}"
                )
                continue

    def check_run_validities(self):
        start_time = time.time()

        # Checking the naive case where all energies are 0.
        if sum(self.energies) == 0:
            return False

        # Iterate over the energies from lowest to highest.
        for index in np.argsort(self.energies):
            package = self.packages[index]

            if package is None:
                continue

            is_valid, checked_energy, miner_energy = self.protein.is_run_valid(
                state=self.miner_states[index], **package
            )
            self.event["is_run_valid_time"][index] = time.time() - start_time

            self.event["checked_energy"][index] = checked_energy
            self.event["miner_energy"][index] = miner_energy
            self.event["is_valid"][index] = is_valid
            self.event["reported_energy"][index] = float(self.energies[index])

            # If the run is valid, then we can presume that all other simulations do not need to be considered for competition.
            if not is_valid:
                self.energies[index] = 0
                continue
            else:
                break
