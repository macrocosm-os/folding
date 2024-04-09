import random
from typing import List, Dict


class HyperParameters:
    def __init__(self, pdb_id: str) -> None:
        """Sample hyperparameters for protein folding for a specific pdb_id

        Args:
            pdb_id (str): id for the protein.
        """
        self.pdb_id: str = pdb_id

        self.FF: List[str] = ["charmm27", "amber03"]
        self.BOX_TYPE: List[str] = ["dodecahedron", "octahedron", "cubic"]
        self.WATER: List = ["tip3p", "tip4p", "spce"]
        self.BOX_DISTANCE: float = 1.0

        self.TOTAL_COMBINATIONS = len(self.FF) * len(self.BOX_TYPE) * len(self.WATER)

        # Sample a combinations without replacement
        self.sampled_combinations: List[Dict[str, str]] = []
        self.remaining_combinations = [
            {
                "FF": ff,
                "BOX_TYPE": box_type,
                "WATER": water,
                "BOX_DISTANCE": str(self.BOX_DISTANCE),
            }
            for ff in self.FF
            for box_type in self.BOX_TYPE
            for water in self.WATER
        ]

        # Randomly shuffle the parameter space to no bias simulations.
        random.shuffle(self.remaining_combinations)

    def sample_hyperparameters(self) -> Dict[str, str]:
        # Check if all combinations have been sampled, if so, reset
        if len(self.sampled_combinations) == self.TOTAL_COMBINATIONS:
            return Exception(
                "All combinations have been sampled, reset the class to sample again."
            )

        # Sample hyperparameter set and add to set of sampled.
        sampled_combination = self.remaining_combinations.pop(0)
        self.sampled_combinations.append(sampled_combination)

        return sampled_combination
