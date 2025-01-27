from typing import List, Callable

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

    def compute_results(self, miner_uid: int, task: str) -> List[float]:
        """computes the results of the miner

        Args:
            miner_uid (int): the miner's unique identifier
            task (str): the task the miner completed
        """
        computed_results = []
        for result in self.registry[miner_uid][task]["results"]:
            computed_results.append(result.calculate_reward())

        return computed_results

    def add_credibilities(self, miner_uid: int, task: str, credibilities: List[float]):
        """adds credibilities to the miner registry

        Args:
            miner_uid (int):
            task (str): name of the task the miner completed
            credibilities (List[float]): a list of credibilities the miner received
        """
        self.registry[miner_uid][task]["credibilities"].extend(credibilities)

    def update_credibility(self, miner_uid: int, task: str):
        """
        Updates the credibility of a miner based on the score they received
        using a mean across all credibilities for the desired task.
        """

        current_credibility = np.mean(self.registry[miner_uid][task]["credibilities"])
        previous_credibility = self.registry[miner_uid][task]["credibility"]

        # Use EMA to update the miner's credibility.
        self.registry[miner_uid][task]["credibility"] = (
            c.CREDIBILITY_ALPHA * current_credibility + (1 - c.CREDIBILITY_ALPHA) * previous_credibility
        )

    def reset(self, miner_uid: int) -> None:
        """Resets the score and credibility of miner 'uid'."""
        for task in self.tasks:
            self.registry[miner_uid][task]["credibility"] = c.STARTING_CREDIBILITY
            self.registry[miner_uid][task]["credibilities"] = []
            self.registry[miner_uid][task]["score"] = 0.0
            self.registry[miner_uid][task]["results"] = []
