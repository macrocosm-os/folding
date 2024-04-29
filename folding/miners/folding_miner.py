import os
import glob
import base64
import time
import psutil
import concurrent.futures

import bittensor as bt
from typing import Dict, List, Tuple

# import base miner class which takes care of most of the boilerplate
from folding.base.miner import BaseMinerNeuron
from folding.protocol import FoldingSynapse

from folding.utils.ops import run_cmd_commands, check_if_directory_exists

# root level directory for the project (I HATE THIS)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def attach_files_to_synapse(synapse: FoldingSynapse, data_directory:str, state:str, desired_files:List[str] = None) -> FoldingSynapse:
    """load the output files as bytes and add to synapse.md_output

    Args:
        synapse (FoldingSynapse): Recently received synapse object
        data_directory (str): directory where the miner is holding the necessary data for the validator.
        state (str): the current state of the simulation
        desired_files (List[str], optional): List of files to attach to the synapse. Defaults to None.
    
    state is either:
     1. nvt
     2. npt
     3. md_0_1

    State depends on the current state of the simulation (controlled in GromacsExecutor.run() method)

    Returns:
        FoldingSynapse: synapse with md_output attached
    """

    if desired_files is None:
        state_files = os.path.join(data_directory, f"{state}") #The current state of the simulation. Likely to always be the last state
        desired_files = glob.glob(f'{state_files}*') + glob.glob(f"{data_directory}/*.edr") #Grab all the state_files and the edr files

    else:
        #If we pass a list of files, we will attach the latest checkpoint file as well.
        latest_cpt_file = max(glob.glob("*.cpt"), key=os.path.getctime) #TODO: This is default behaviour, but maybe we shouldn't do this? 
        desired_files.append(latest_cpt_file)

        desired_files = [os.path.join(data_directory, file) for file in desired_files] #Explicitly add the data_directory to the files

    bt.logging.info(f"Desired files to send to the validator: {desired_files}")
    for filename in desired_files:
        bt.logging.info(f"Attaching file: {filename!r} to synapse.md_output")
        try:
            with open(filename, "rb") as f:
                synapse.md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            bt.logging.error(f"Failed to read file {filename!r} with error: {e}")

    bt.logging.success(
        f"Finished running GROMACS simulation for protein: {synapse.pdb_id} and attached {len(synapse.md_output)} files to synapse.md_output"
    )

    return synapse

class FoldingMiner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(FoldingMiner, self).__init__(config=config)

        self.executors = {}  # Maps pdb_ids to GromacsExecutors 
        self.futures = {}    # Maps pdb_ids to futures
        self.max_num_processes = psutil.cpu_count(logical=False)  # Only physical cores
        self.max_workers=self.max_num_processes - 1 #subtract one to ensure that we are not using all the processors. 

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) #remove one for safety

    def compute_itermediate_gro(self, synapse:FoldingSynapse, data_directory:str, state:str) -> FoldingSynapse:
        """Compute the intermediate gro file from the xtc file. 

        Args:
            synapse (FoldingSynapse): Recently received synapse object
            data_directory (str): directory where the miner is holding the necessary intermediate data.
            state (str): the current state of the simulation

        Returns:
            FoldingSynapse: synapse with md_output attached
        """
        file_location = os.path.join(data_directory, f"{state}") 
        gro_filename = f'{state}_intermediate.gro'
        command = [f"gmx trjconv -s {file_location}.tpr -f {file_location}.xtc -o {gro_filename} -dump -1"] #TODO: Could have an internal counter to show how many times we have been queried.

        run_cmd_commands(commands=command, suppress_cmd_output=self.config.neuron.suppress_cmd_output)
        return attach_files_to_synapse(synapse=synapse, data_directory = data_directory, state=state, desired_files=[gro_filename])


    def configure_commands(self, mdrun_args: str) -> Dict[str, List[str]]:
        commands = [
            "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr",  # Temperature equilibration
            "gmx mdrun -deffnm nvt " + mdrun_args,

            "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr",  # Pressure equilibration
            "gmx mdrun -deffnm npt " + mdrun_args,

            f"gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr",  # Production run
            f"gmx mdrun -deffnm md_0_1 " + mdrun_args,

            f"echo '1\n1\n' | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_center.xtc -center -pbc mol",
        ]


        #These are rough identifiers for the different states of the simulation
        state_commands = {
            "nvt": commands[:2],
            "npt": commands[2:4],
            "md_0_1": commands[4:],
        }

        return state_commands

    async def forward(self, synapse: FoldingSynapse) -> FoldingSynapse:
        """The main async function that is called by the dendrite to run the simulation.

        Returns:
            FoldingSynapse: synapse with md_output attached
        """

        #If we are already running a process with the same identifier, return intermediate information
        if synapse.pdb_id in self.executors:
            simulation_state = self.executors[synapse.pdb_id].get_state()
            data_directory = self.executors[synapse.pdb_id].output_dir
            return self.compute_itermediate_gro(synapse=synapse, data_directory=data_directory, state = simulation_state)
            
        # Check if the number of active processes is less than the number of CPUs
        if len(self.executors) >= self.max_workers:
            bt.logging.warning("Cannot start new process: CPU limit reached.")
            return synapse # Return the synapse as is
        
        state_commands = self.configure_commands(mdrun_args=synapse.mdrun_args)

        #Create the job and submit it to the executor
        gromax_executor = GromacsExecutor(synapse = synapse)
        future = self.executor.submit(gromax_executor.run, synapse, state_commands, self.config)

        self.executors[synapse.pdb_id] = gromax_executor
        self.futures[synapse.pdb_id] = future

        #Here we will always check the total number of processes running and return the final synapse if we are done with a process.
        while True:
            if len(self.executors) == 0: 
                break

            futures = self.futures
            for pdb_id, future in futures.items():
                if future.done():
                    gromacs_executor = self.executors[pdb_id] 
                    final_synapse = gromacs_executor.final_synapse

                    del gromacs_executor #no need to keep the data in memory. 
                    del self.futures[pdb_id] #remove any artifact of the future. 
                    return final_synapse

    
class GromacsExecutor():
    def __init__(self, synapse:FoldingSynapse) -> None:
        
        self.pdb_id = synapse.pdb_id
        self.output_dir = os.path.join(
                    ROOT_DIR,
                    "data",
                    synapse.axon.hotkey[:8],
                    synapse.pdb_id,
                )
    
    def run(self, synapse: FoldingSynapse, commands:Dict, config:Dict) -> FoldingSynapse:
        """run method to handle the processing of the gromacs simulation.

        Args:
            synapse (FoldingSynapse): synapse object that contains the information for the simulation
            commands (Dict): Dict of lists of commands that we are meant to run in the executor
            config (Dict): configuration file for the miner

        Returns:
            FoldingSynapse: The ORIGINAL synapse with the md_output attached
        """
        bt.logging.info(
            f"Running GROMACS simulation for protein: {synapse.pdb_id} with files {synapse.md_inputs.keys()} mdrun_args: {synapse.mdrun_args}"
        )

        synapse.md_output = {}

        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=self.output_dir)
        os.chdir(self.output_dir) #TODO: will this be a problem with many processes?

        # The following files are required for GROMACS simulations and are recieved from the validator
        for filename, content in synapse.md_inputs.items():
            # Write the file to the output directory
            with open(filename, "w") as file:
                bt.logging.info(f"\nWriting {filename} to {self.output_dir}")
                file.write(content)

        for state, commands in commands.items():
            bt.logging.info(f"Running {state} commands")
            self.state = state

            run_cmd_commands(
                commands=commands, suppress_cmd_output=config.neuron.suppress_cmd_output
            )

        self.final_synapse = attach_files_to_synapse(synapse=synapse, data_directory=self.output_dir, state=self.state) 

    def get_state(self):
        return self.state #retuns which protion of the simulation stack the miner is in. 