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

    def __init__(self):
        pass

    def extract(filepath: str, names=["step", "energy"]):
        return pd.read_csv(filepath, sep="\s+", header=None, names=names)

    def energy(
        self,
        data_type: str,
        path: str,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        output_data_location = os.path.join(path, f"{data_type}.xvg")
        command = [
            f"printf '{data_type}\n0\n' | {base_command} -f {path}/em.edr -o {data_type}.xvg {xvg_command}"
        ]
        run_cmd_commands(command)

        return self.extract(filepath=output_data_location, names=["step", "energy"])

    def temperature(
        self,
        data_type: str,
        path: str,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        output_data_location = os.path.join(path, f"{data_type}.xvg")
        command = [
            f"echo '{data_type}' | {base_command} -f {path}/em.edr -o {output_data_location} {xvg_command} -b 20"
        ]
        run_cmd_commands(command)

        return self.extract(
            filepath=output_data_location, names=["step", "temperature"]
        )

    def pressure(
        self,
        data_type: str,
        path: str,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        output_data_location = os.path.join(path, f"{data_type}.xvg")
        command = [
            f"echo '{data_type}' | {base_command} -f {path}/npt.edr -o {data_type}.xvg {xvg_command}"
        ]
        run_cmd_commands(command)

        return self.extract(filepath=output_data_location, names=["step", "pressure"])

    def density(
        self,
        data_type: str,
        path: str,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        output_data_location = os.path.join(path, f"{data_type}.xvg")
        command = [
            f"echo '{data_type}' | {base_command} -f {path}/npt.edr -o {data_type}.xvg {xvg_command}"
        ]
        run_cmd_commands(command)

        return self.extract(filepath=output_data_location, names=["step", "density"])

    def prod_energy(
        self,
        data_type: str,
        path: str,
        base_command: str = "gmx energy",
        xvg_command: str = "-xvg none",
    ):
        xvg_name = "potential_production_run.xvg"
        output_data_location = os.path.join(path, xvg_name)
        command = [
            f"printf '{data_type}\n0\n' | {base_command} -f {path}/md_0_1.edr -o {xvg_name} {xvg_command}"
        ]
        run_cmd_commands(command)

        return self.extract(
            filepath=output_data_location, names=["step", "temperature"]
        )

    def rmsd(self, path: str, xvg_command: str = "-xvg none"):
        xvg_name = "rmsd_xray.xvg"
        output_data_location = os.path.join(path, xvg_name)
        command = [
            f"echo '4 4' | gmx rms -s {path}/md_0_1.tpr -f md_center.xtc -o {xvg_name} -tu ns {xvg_command}"
        ]
        run_cmd_commands(command)

        return self.extract(filepath=output_data_location, names=["step", "rmsd"])
