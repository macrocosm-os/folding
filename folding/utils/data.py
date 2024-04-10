from ops import run_cmd_commands



class DataExtractor: 
    def __init__(self, data_type, base_command, xvg_command):
        self.data_type = data_type
        self.base_command = base_command # 'gmx energy'
        self.xvg_command = xvg_command # '-xvg none' 
    
    def potential_energy(self):
        # self.data_type = "Potential"
        command = f"printf '{self.data_type}\n0\n' | {self.base_command} -f em.edr -o {self.data_type}.xvg {self.xvg_command}" 
        run_cmd_commands(command)

    def temperature(self):
        # self.data_type = "T-rest"
        command = f"echo '{self.data_type}' | {self.base_command} -f em.edr -o {self.data_type}.xvg {self.xvg_command} -b 20"
        run_cmd_commands(command)

    def pressure(self):
        # self.data_type = "Pressure"
        command = f"echo '{self.data_type}' | {self.base_command} -f npt.edr -o {self.data_type}.xvg {self.xvg_command}"
        run_cmd_commands(command)

    def density(self):
        # self.data_type = "Density"
        command = f"echo '{self.data_type}' | {self.base_command} -f npt.edr -o {self.data_type}.xvg {self.xvg_command}"
        run_cmd_commands(command)

    def rmsd(self):
        # self.data_type = "RMSD"
        command = f"echo '4 4' | gmx rms -s em.tpr -f md_center.xtc -o rmsd_xray.xvg -tu ns {self.xvg_command}"
        run_cmd_commands(command)



    # after the validator gets the responses, computes (extracts) the data 
    # create a data parsing class with methods you can call. so we can pass in any file type
    # leverage the run_cmd_commands() from ops 

    # function that specifies what we want, then the run cmd. 
