from .ops import run_cmd_commands



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
         
    def energy(self, data_type, path, base_command: str = "gmx energy", xvg_command: str = "-xvg none"):
        command = [f"printf '{data_type}\n0\n' | {base_command} -f {path}/em.edr -o {data_type}.xvg {xvg_command}"] 
        run_cmd_commands(command)

    def temperature(self, data_type, path, base_command: str = "gmx energy", xvg_command: str = "-xvg none"):
        command = [f"echo '{data_type}' | {base_command} -f {path}/em.edr -o {data_type}.xvg {xvg_command} -b 20"]
        run_cmd_commands(command)

    def pressure(self, data_type, path, base_command: str = "gmx energy", xvg_command: str = "-xvg none"):
        command = [f"echo '{data_type}' | {base_command} -f {path}/npt.edr -o {data_type}.xvg {xvg_command}"]
        run_cmd_commands(command)

    def density(self, data_type, path, base_command: str = "gmx energy", xvg_command: str = "-xvg none"):
        command = [f"echo '{data_type}' | {base_command} -f {path}/npt.edr -o {data_type}.xvg {xvg_command}"]
        run_cmd_commands(command)

    def prod_energy(self, data_type, path, base_command: str = "gmx energy", xvg_command: str = "-xvg none"):
        command = [f"printf '{data_type}\n0\n' | {base_command} -f {path}/md_0_1.edr -o potential_production_run.xvg {xvg_command}"] 
        run_cmd_commands(command)

    def rmsd(self, path, xvg_command: str = "-xvg none"):
        command = [f"echo '4 4' | gmx rms -s {path}/md_0_1.tpr -f md_center.xtc -o rmsd_xray.xvg -tu ns {xvg_command}"]
        run_cmd_commands(command)
