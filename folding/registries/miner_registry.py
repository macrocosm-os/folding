from typing import List, Callable
from itertools import chain

import numpy as np
import folding.utils.constants as c
from folding.registries.evaluation_registry import EVALUATION_REGISTRY


class MinerRegistry:
    """
    Class to handling holding the scores, credibilities, and other attributes of miners.
    """

    def __init__(self, miner_uids: List[int]):
        self.tasks: List[str] = EVALUATION_REGISTRY.keys()
        self.registry = dict.fromkeys(miner_uids)

        for miner_uid in miner_uids:
            self.registry[miner_uid] = {}
            self.registry[miner_uid]["overall_credibility"] = c.STARTING_CREDIBILITY
            for task in self.tasks:
                self.registry[miner_uid][task] = {
                    "credibility": c.STARTING_CREDIBILITY,
                    "credibilities": [],
                    "score": 0.0,
                    "results": [],
                }

    def add_results(self, miner_uid: int, task: str, results: List[Callable]):
        """adds scores to the miner registry

        Args:
            miner_uid (int):
            task (str): name of the task the miner completed
            results (List[Callable]): a list of Callables that represent the scores the miner received
        """
        self.registry[miner_uid][task]["results"].extend(results)

    def add_credibilities(self, miner_uid: int, task: str, credibilities: List[float]):
        """adds credibilities to the miner registry

        Args:
            miner_uid (int):
            task (str): name of the task the miner completed
            credibilities (List[float]): a list of credibilities the miner received
        """

        self.registry[miner_uid][task]["credibilities"].append(credibilities)

    def update_credibility(self, miner_uid: int, task: str):
        """
        Updates the credibility of a miner based:
        1. The credibility of the miner's previous results. Intially set as STARTING_CREDIBILITY
        2. The credibility of the miner's current results.
        3. The number of previous and current entries to act as a weighting factor
        4. The EMA with credibility_alpha as the smoothing factor
        """

        task_credibilities = list(chain.from_iterable(self.registry[miner_uid][task]["credibilities"]))

        current_credibility = np.mean(task_credibilities)
        previous_credibility = self.registry[miner_uid][task]["credibility"]

        # Use EMA to update the miner's credibility.
        self.registry[miner_uid][task]["credibility"] = (
            c.CREDIBILITY_ALPHA * current_credibility + (1 - c.CREDIBILITY_ALPHA) * previous_credibility
        )

        # Reset the credibilities.
        self.registry[miner_uid][task]["credibilities"] = []

        all_credibilities = []
        for task in self.tasks:
            all_credibilities.append(self.registry[miner_uid][task]["credibility"])

        # Your overall credibility is the minimum of all the credibilities.
        self.registry[miner_uid]["overall_credibility"] = min(all_credibilities)

    def reset(self, miner_uid: int) -> None:
        """Resets the score and credibility of miner 'uid'."""
        for task in self.tasks:
            self.registry[miner_uid][task]["credibility"] = c.STARTING_CREDIBILITY
            self.registry[miner_uid][task]["credibilities"] = []
            self.registry[miner_uid][task]["score"] = 0.0
            self.registry[miner_uid][task]["results"] = []
