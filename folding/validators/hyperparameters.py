import random
from itertools import product
from typing import List, Dict, Union

import bittensor as bt


class HyperParameters:
    def __init__(self, exclude: Union[Dict[str, str], List[str]] = None) -> None:
        """Sample hyperparameters for protein folding for a specific pdb_id

        Args:
            exclude (List[str], optional): List of hyperparameters to exclude. Defaults to None.
                needs to be either ['FF','BOX','WATER'] to exclude a specific hyperparameter, or
                you can exclude a specific value(s) by passing {'FF': 'charmm27', 'BOX_TYPE': 'dodecahedron'}.
        """

        # Need to download files for specific inputs.
        self.FF: List[str] = ["charmm27", "amber03"]
        self.BOX: List[str] = ["dodecahedron", "octahedron", "cubic"]
        self.WATER: List = ["tip3p", "spce"]
        self.BOX_DISTANCE: List[float] = [1.0]

        self.parameter_set = {
            "FF": self.FF,
            "BOX": self.BOX,
            "WATER": self.WATER,
            "BOX_DISTANCE": self.BOX_DISTANCE,
        }

        self.exclude: List[str] = exclude or []

        self.create_parameter_space()
        self.setup_combinations()

    def create_parameter_space(self):
        """
        Make the parameter space based on the hyperparameters, and those to exclude.
        If exclude is an empty list, we do nothing.
        """
        if isinstance(self.exclude, dict):
            for key, value in self.exclude.items():
                key = key.upper()
                if key in self.parameter_set.keys():
                    try:
                        self.parameter_set[key].remove(value)
                    except:
                        bt.logging.warning(
                            f"Value {value} not found in {key} parameter set. Skipping..."
                        )
                else:
                    bt.logging.error(
                        f"Parameter {key} not found in parameter set. Only FF, BOX_TYPE, and/or WATER are allowed."
                    )

        elif isinstance(
            self.exclude, list
        ):  # The case where you want to remove entire parameters (['FF'])
            self.exclude = [string.upper() for string in self.exclude]

            for param in self.exclude:
                try:
                    self.parameter_set.pop(param)
                except KeyError:
                    bt.logging.error(
                        f"Parameter {param} not found in parameter set. Only FF, BOX_TYPE, and/or WATER are allowed."
                    )

    def setup_combinations(self):
        """
        Samples all possible combinations of hyperparameters for protein folding.
        Remove excluded hyperparameters from the list of possible combinations.
        """
        self.sampled_combinations: List[Dict[str, str]] = []

        # Create a list of tuples, and then corresponds them in dictionary format.
        self.all_combinations = list(product(*self.parameter_set.values()))
        self.all_combinations = [
            {key: value for key, value in zip(self.parameter_set.keys(), combination)}
            for combination in self.all_combinations
        ]

        self.TOTAL_COMBINATIONS = len(self.all_combinations)

        # Randomly shuffle the parameter space to no bias simulations.
        random.shuffle(self.all_combinations)

    def sample_hyperparameters(self) -> Dict[str, str]:
        """Return a dictionary of pdb_id and sampled hyperparameters."""

        if len(self.sampled_combinations) == self.TOTAL_COMBINATIONS:
            return Exception(
                "All combinations have been sampled, reset the class to sample again."
            )

        # Sample hyperparameter set and add to set of sampled.
        sampled_combination = self.all_combinations.pop(0)
        self.sampled_combinations.append(sampled_combination)

        return sampled_combination
