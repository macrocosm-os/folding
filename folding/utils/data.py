import os
import pandas as pd
import bittensor as bt

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

    def extract(self, filepath: str, names=["step", "default-name"]):
        return pd.read_csv(filepath, sep="\s+", header=None, names=names)

    # def energy(
    #     self,
    #     data_type: str,
    #     output_path: str = None,
    #     base_command: str = "gmx energy",
    #     xvg_command: str = "-xvg none",
    # ):
    #     if output_path is None:
    #         output_path = self.miner_data_directory

    #     output_data_location = os.path.join(output_path, f"{data_type}.xvg")
    #     command = [
    #         f"echo '{data_type}' | {base_command} -f {self.validator_data_directory}/em.edr -o {output_data_location} {xvg_command}"
    #     ]
    #     run_cmd_commands(command)

    #     self.data["energy"] = self.extract(
    #         filepath=output_data_location, names=["step", "energy"]
    #     )

    def temperature(
        self,
        data_type: str,
        output_path: str = None,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        if output_path is None:
            output_path = self.miner_data_directory

        output_data_location = os.path.join(output_path, f"{data_type}.xvg")
        command = [
            f"echo Potential | {base_command} -f {self.validator_data_directory}/em.edr -o {output_data_location} {xvg_command} -b 20"
        ]
        run_cmd_commands(command)

        self.data["temperature"] = self.extract(
            filepath=output_data_location, names=["step", "temperature"]
        )

    def pressure(
        self,
        data_type: str,
        output_path: str = None,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        if output_path is None:
            output_path = self.miner_data_directory

        output_data_location = os.path.join(output_path, f"{data_type}.xvg")
        command = [
            f"echo Potential | {base_command} -f {output_path}/npt.edr -o {output_data_location} {xvg_command}"
        ]
        run_cmd_commands(command)

        self.data["pressure"] = self.extract(
            filepath=output_data_location, names=["step", "pressure"]
        )

    def density(
        self,
        data_type: str,
        output_path: str = None,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        if output_path is None:
            output_path = self.miner_data_directory

        output_data_location = os.path.join(output_path, f"{data_type}.xvg")
        command = [
            f"echo '{data_type}' | {base_command} -f {output_path}/npt.edr -o {output_data_location} {xvg_command}"
        ]
        run_cmd_commands(command)

        self.data["density"] = self.extract(
            filepath=output_data_location, names=["step", "density"]
        )

    def energy(
        self,
        data_type: str,
        output_path: str = None,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        if output_path is None:
            output_path = self.miner_data_directory

        xvg_name = "rerun_energy_extracted.xvg"
        output_data_location = os.path.join(output_path, xvg_name)
        command = [
            f"printf '{data_type}\n0\n' | {base_command} -f {output_path}/rerun_energy.edr -o {output_data_location} {xvg_command}"
        ]
        run_cmd_commands(command)

        self.data["energy"] = self.extract(
            filepath=output_data_location, names=["step", "energy"]
        )

    def rmsd(self, output_path: str = None, xvg_command: str = "-xvg none"):
        if output_path is None:
            output_path = self.miner_data_directory

        xvg_name = "rmsd_xray.xvg"
        output_data_location = os.path.join(output_path, xvg_name)
        command = [
            f"echo '4 4' | gmx rms -s {output_path}/md_0_1.tpr -f {output_path}/md_0_1_center.xtc -o {output_data_location} -tu ns {xvg_command}"
        ]
        run_cmd_commands(command)

        self.data["rmsd"] = self.extract(
            filepath=output_data_location, names=["step", "rmsd"]
        )

    def rerun_potential(self, output_path: str = None):
        if output_path is None:
            output_path = self.validator_data_directory

        xvg_name = "potential_rerun.xvg"
        output_data_location = os.path.join(output_path, xvg_name)
        command = [
            f"! echo 'Potential' | gmx energy -f {output_path}/rerun_calculation.edr -o {output_path}/{xvg_name}"
        ]

        run_cmd_commands(command)

        self.data["potential_rerun"] = self.extract(
            filepath=output_data_location, names=["step", "rerun_potential_energy"]
        )
