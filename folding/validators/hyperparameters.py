import random
from itertools import product
from typing import List, Dict, Union

import bittensor as bt

from folding.utils.openmm_forcefields import FORCEFIELD_REGISTERY


class HyperParameters:
    def __init__(self, exclude: Union[Dict[str, str], List[str]] = None) -> None:
        """Sample hyperparameters for protein folding for a specific pdb_id

        Args:
            exclude (List[str], optional): List of hyperparameters to exclude. Defaults to None.
                needs to be either ['FF','WATER'] to exclude a specific hyperparameter, or
                you can exclude a specific value(s) by passing {'FF': 'charmm27', 'BOX_TYPE': 'dodecahedron'}.
        """

        self.sampled_combinations: List[Dict[str, str]] = []
        self.all_combinations: List[Dict[str, str]] = []

        # Need to download files for specific inputs.
        fields = [field() for field in FORCEFIELD_REGISTERY.values()]
        self.exclude: List[str] = exclude or []

        initial_search: List[Dict[str, str]] = []

        for field in fields:
            self.FF: List[str] = field.forcefields
            self.WATER: List[str] = field.waters
            self.BOX = ["cubic", "dodecahedron", "octahedron"]

            parameter_set = {
                "FF": self.FF,
                "WATER": self.WATER,
                "BOX": self.BOX,
            }

            self.create_parameter_space()
            self.all_combinations.extend(
                self.setup_combinations(parameter_set=parameter_set)
            )

            initial_search.append(field.recommended_configuration)

        random.shuffle(self.all_combinations)

        self.all_combinations = initial_search + self.all_combinations
        self.TOTAL_COMBINATIONS = len(self.all_combinations)

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

    def setup_combinations(
        self, parameter_set: Dict[str, List[str]]
    ) -> List[Dict[str, str]]:
        """
        Samples all possible combinations of hyperparameters for protein folding.
        Remove excluded hyperparameters from the list of possible combinations.
        """

        # Create a list of tuples, and then corresponds them in dictionary format.
        all_combinations = list(product(*parameter_set.values()))
        all_combinations = [
            {key: value for key, value in zip(parameter_set.keys(), combination)}
            for combination in all_combinations
        ]

        return all_combinations

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
