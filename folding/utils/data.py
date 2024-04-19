import os
from typing import Dict

import pandas as pd
from folding.utils.ops import run_cmd_commands


class DataExtractor:
    """
    A class that containing methods to extract various types of data from simulation files using GROMACS commandsds.

    Methods:
        potential_energy(): Extracts potential energy data from the simulation.
        temperature(): Extracts temperature data from the simulation.
        pressure(): Extracts pressure data from the simulation.
        density(): Extracts density data from the simulation.
        rmsd(): Extracts RMSD data from the simulation.
    """

    def __init__(self, miner_data_directory: str, validator_data_directory: str):
        """Data extraction class to obtain the necessary information needed for the reward stack

        Args:
            miner_data_directory (str): location of the validators own miner directory (ex: /root/folding/data/5oxe/validator/5CUU1KW2)
            validator_data_directory (str): ex: '/root/folding/data/5oxe/validator'
        """
        self.validator_data_directory = validator_data_directory
        self.miner_data_directory = miner_data_directory

        self.data = {}
        self.commands = []

        self.pipelines = [
            dict(
                method=self.get_energy,
                kwargs=dict(name="energy", input_file_name="em.edr"),
            ),
            dict(
                method=self.get_energy,
                kwargs=dict(name="temperature", input_file_name="em.edr"),
            ),
            dict(
                method=self.get_energy,
                kwargs=dict(name="pressure", input_file_name="npt.edr"),
            ),
            dict(
                method=self.get_energy,
                kwargs=dict(name="density", input_file_name="npt.edr"),
            ),
            dict(
                method=self.get_energy,
                kwargs=dict(
                    name="prod_energy",
                    input_file_name="md_0_1.edr",
                ),
            ),
            dict(
                method=self.get_rmsd,
                kwargs=dict(
                    name="rmsd",
                    data_type="Potential",
                    input_file_name="md_0_1_center.xtc",
                ),
            ),
        ]

    def extract(self, filepath: str, names=["step", "default-name"]):
        return pd.read_csv(filepath, sep="\s+", header=None, names=names)

    def forward(self) -> Dict:
        for pipe in self.pipelines:
            for method, kwargs in pipe.items():
                method(**kwargs)

        return self.data

    def get_energy(
        self,
        name: str,
        input_file_name: str,
        data_type: str = "Potential",
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        output_data_location = os.path.join(self.miner_data_directory, f"{name}.xvg")

        # The em.edr file is something that the validator INITIALLY creates, before sending anything to the miner.
        # Therefore, we know the location is in the validator directory.
        input_file_parent = (
            self.validator_data_directory
            if input_file_name == "em.edr"
            else self.miner_data_directory
        )

        # Hardcoded constraints on prod_energy
        cmd = f"{data_type}\n0\n" if name == "prod_energy" else f"echo '{data_type}'"
        command = [
            f"{cmd} | {base_command} -f {input_file_parent}/{input_file_name} -o {output_data_location} {xvg_command}"
        ]

        run_cmd_commands(
            command
        )  # necessary to run because we need to create the file before extraction

        self.data[name] = self.extract(
            filepath=output_data_location, names=["step", f"{name}-energy"]
        )

    def get_rmsd(
        self,
        name: str,
        data_type: str,
        xvg_command: str = "-xvg none",
        **kwargs,
    ):
        output_data_location = os.path.join(
            self.miner_data_directory, f"{data_type}.xvg"
        )

        command = [
            f"echo '4 4' | gmx rms -s {self.miner_data_directory}/md_0_1.tpr -f {self.miner_data_directory}/md_0_1_center.xtc -o {output_data_location} -tu ns {xvg_command}"
        ]

        run_cmd_commands(command)

        self.data[name] = self.extract(
            filepath=output_data_location, names=["step", "rmsd"]
        )
