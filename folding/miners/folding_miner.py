import os
import time
import glob
import base64
import psutil
import concurrent.futures
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass
from utils.monitor import auto_plot
import bittensor as bt

# import base miner class which takes care of most of the boilerplate
from folding.base.miner import BaseMinerNeuron
from folding.protocol import FoldingSynapse
from folding.utils.ops import run_cmd_commands, check_if_directory_exists

# root level directory for the project (I HATE THIS)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_PATH = os.path.join(ROOT_DIR, "data")


def attach_files_to_synapse(
    synapse: FoldingSynapse,
    data_directory: str,
    state: str,
    desired_files: List[str] = None,
) -> FoldingSynapse:
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
     4. finished

    State depends on the current state of the simulation (controlled in GromacsExecutor.run() method)

    Returns:
        FoldingSynapse: synapse with md_output attached
    """

    synapse.md_output = {}  # ensure that the initial state is empty

    try:
        if desired_files is None:
            state_files = os.path.join(
                data_directory, f"{state}"
            )  # The current state of the simulation. Likely to always be the last state
            desired_files = glob.glob(f"{state_files}*") + glob.glob(
                f"{data_directory}/*.edr"
            )  # Grab all the state_files and the edr files

        else:
            # If we pass a list of files, we will attach the latest checkpoint file as well.
            latest_cpt_file = max(
                glob.glob("*.cpt"), key=os.path.getctime
            )  # TODO: This is default behaviour, but maybe we shouldn't do this?

            desired_files.append(latest_cpt_file)

            desired_files = [
                os.path.join(data_directory, file) for file in desired_files
            ]  # Explicitly add the data_directory to the files

        bt.logging.info(f"Desired files to send to the validator: {desired_files}")
        for filename in desired_files:
            bt.logging.info(f"Attaching file: {filename!r} to synapse.md_output")
            try:
                with open(filename, "rb") as f:
                    synapse.md_output[filename] = base64.b64encode(f.read())
            except Exception as e:
                bt.logging.error(f"Failed to read file {filename!r} with error: {e}")

        bt.logging.success(
            f"Attached {len(synapse.md_output)} files to synapse.md_output for protein: {synapse.pdb_id}"
        )

    except Exception as e:
        bt.logging.error(f"Failed to attach files with error: {e}")

    finally:
        return synapse  # either return the synapse wth the md_output attached or the synapse as is.


class FoldingMiner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(FoldingMiner, self).__init__(config=config)

        def nested_dict():
            return defaultdict(
                lambda: None
            )  # allows us to set the desired attribute to anything.

        self.simulations = defaultdict(
            nested_dict
        )  # Maps pdb_ids to the current state of the simulation

        self.max_num_processes = psutil.cpu_count(logical=False)  # Only physical cores
        self.max_workers = (
            self.max_num_processes - 1
        )  # subtract one to ensure that we are not using all the processors.

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )  # remove one for safety

    def compute_itermediate_gro(
        self, synapse: FoldingSynapse, data_directory: str, state: str
    ) -> FoldingSynapse:
        """Compute the intermediate gro file from the xtc file.

        Args:
            synapse (FoldingSynapse): Recently received synapse object
            data_directory (str): directory where the miner is holding the necessary intermediate data.
            state (str): the current state of the simulation

        Returns:
            FoldingSynapse: synapse with md_output attached
        """
        file_location = os.path.join(data_directory, f"{state}")
        gro_filename = f"{state}_intermediate.gro"
        command = [
            f"gmx trjconv -s {file_location}.tpr -f {file_location}.xtc -o {gro_filename} -dump -1"
        ]  # TODO: Could have an internal counter to show how many times we have been queried.

        run_cmd_commands(
            commands=command, suppress_cmd_output=self.config.neuron.suppress_cmd_output
        )

        return attach_files_to_synapse(
            synapse=synapse,
            data_directory=data_directory,
            state=state,
            desired_files=[gro_filename],
        )

    def configure_commands(self, mdrun_args: str) -> Dict[str, List[str]]:
        # get the monitoring commands
        monitoring_commands = self.configure_monitoring_commands()

        commands = [
            "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr",  # Temperature equilibration
            "gmx mdrun -deffnm nvt " + mdrun_args,
            "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr",  # Pressure equilibration
            "gmx mdrun -deffnm npt " + mdrun_args,
            f"gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr",  # Production run
            f"gmx mdrun -deffnm md_0_1 " + mdrun_args,
            f"echo '1\n1\n' | gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_center.xtc -center -pbc mol",
        ]

        # These are rough identifiers for the different states of the simulation
        state_commands = {
            "nvt": commands[:2],
            "npt": commands[2:4],
            "md_0_1": commands[4:],
        }

        # Append the monitoring commands to each state
        for state, cmds in state_commands.items():
            # check if the last command in the state is still running. 
            monitoring_block = f"while pgrep -f '{cmds[-1]}' > /dev/null; do {monitoring_commands[state][0]}; sleep 1; done"
            cmds.append(monitoring_block)

        return state_commands

    def configure_monitoring_commands(self):
        commands = [
            "echo 'Potential' | gmx energy -f nvt.edr -o 'nvt_potential.xvg' -nobackup", # -nobackup overwrites the output file
            "echo 'Potential' | gmx energy -f npt.edr -o 'npt_potential.xvg' -nobackup",
            "echo 'Potential' | gmx energy -f md_0_1.edr -o 'md_0_1_potential.xvg' -nobackup",
        ]
        state_commands_for_monitoring = {
            "nvt": commands[:1],
            "npt": commands[1:2],
            "md_0_1": commands[2:],
        }

        return state_commands_for_monitoring
    

    def forward(self, synapse: FoldingSynapse) -> FoldingSynapse:
        """
        The main async function that is called by the dendrite to run the simulation.
        There are a set of default behaviours the miner should carry out based on the form the synapse comes in as:

            1. If the synapse has a pdb_id that is already being processed, return the intermediate gro file
            2. If the synapse md_inputs contains a ckpt file, then we are expected to either accept/reject a simulation rebase.
            3. If none of the above conditions are met, we start a new simulation.
                - If the number of active processes is less than the number of CPUs and the pdb_id is unique, start a new process

        Returns:
            FoldingSynapse: synapse with md_output attached
        """

        # If we are already running a process with the same identifier, return intermediate information
        if synapse.pdb_id in self.simulations:
            simulation = self.simulations[synapse.pdb_id]

            if simulation["executor"].get_state() == "finished":
                final_synapse = attach_files_to_synapse(
                    synapse=synapse,
                    data_directory=simulation["output_dir"],
                    state="md_0_1",  # attach the last state of the simulation as not files are labelled as 'finished'
                )

                # This will run if final_synapse exists.
                del self.simulations[
                    synapse.pdb_id
                ]  # Remove the simulation from the list
                return final_synapse

            return self.compute_itermediate_gro(
                synapse=synapse,
                data_directory=simulation["output_dir"],
                state=simulation["executor"].get_state(),
            )

        elif synapse.pdb_id in os.listdir(BASE_DATA_PATH):
            # If we have a pdb_id in the data directory, we can assume that the simulation has been run before
            # and we can return the files from the last simulation. This only works if you have kept the data.
            output_dir = os.path.join(BASE_DATA_PATH, synapse.pdb_id)

            return attach_files_to_synapse(
                synapse=synapse, data_directory=output_dir, state="md_0_1"
            )

        # Check if the number of active processes is less than the number of CPUs
        if len(self.simulations) >= self.max_workers:
            bt.logging.warning("Cannot start new process: CPU limit reached.")
            return synapse  # Return the synapse as is

        state_commands = self.configure_commands(mdrun_args=synapse.mdrun_args)

        # Create the job and submit it to the executor
        gromax_executor = GromacsExecutor(pdb_id=synapse.pdb_id)

        future = self.executor.submit(
            gromax_executor.run,
            synapse.md_inputs,
            state_commands,
            self.config.neuron.suppress_cmd_output,
        )

        self.simulations[synapse.pdb_id]["executor"] = gromax_executor
        self.simulations[synapse.pdb_id]["future"] = future
        self.simulations[synapse.pdb_id]["output_dir"] = output_dir


class GromacsExecutor:
    def __init__(self, pdb_id: str) -> None:
        self.pdb_id = pdb_id
        self.state: str = None
        self.state_file_name = f"{pdb_id}_state.txt"

        self.output_dir = os.path.join(BASE_DATA_PATH, self.pdb_id)

    def run(
        self,
        md_inputs: Dict,
        commands: Dict,
        suppress_cmd_output: bool = True,
    ) -> FoldingSynapse:
        """run method to handle the processing of the gromacs simulation.

        Args:
            synapse (FoldingSynapse): synapse object that contains the information for the simulation
            commands (Dict): Dict of lists of commands that we are meant to run in the executor
            config (Dict): configuration file for the miner

        Returns:
            FoldingSynapse: The ORIGINAL synapse with the md_output attached
        """
        bt.logging.info(
            f"Running GROMACS simulation for protein: {self.pdb_id} with files {md_inputs.keys()}"
        )

        # Make sure the output directory exists and if not, create it
        check_if_directory_exists(output_directory=self.output_dir)
        os.chdir(self.output_dir)  # TODO: will this be a problem with many processes?

        # The following files are required for GROMACS simulations and are recieved from the validator
        for filename, content in md_inputs.items():
            # Write the file to the output directory
            with open(filename, "w") as file:
                bt.logging.info(f"\nWriting {filename} to {self.output_dir}")
                file.write(content)

        for state, commands in commands.items():
            bt.logging.info(f"Running {state} commands")

            with open(self.state_file_name, "w") as f:
                f.write(f"{state}\n")
                plot_state = f.read().strip()
                
            run_cmd_commands(commands=commands, suppress_cmd_output=suppress_cmd_output)
            
            if plot_state in ["nvt", "npt", "md_0_1"]:
                auto_plot(data_location=self.output_dir, file_name=f"{state}_potential.xvg", output_file=f"{state}_potential.png")

        bt.logging.success(f"✅Finished GROMACS simulation for protein: {self.pdb_id}✅")

        state = "finished"
        with open(self.state_file_name, "w") as f:
            f.write(f"{state}\n")

    def get_state(self) -> str:
        """get_state reads a txt file that contains the current state of the simulation"""
        with open(os.path.join(self.output_dir, self.state_file_name), "r") as f:
            lines = f.readlines()

            if lines:
                return lines[-1].strip()  # return the last line of the file
            return None


class MockGromacsExecutor(GromacsExecutor):
    def __init__(self, pdb_id: str, output_dir: str) -> None:
        super().__init__(pdb_id=pdb_id)
        self.required_values = set(["init", "wait", "finished"])
        self.output_dir = output_dir

    def run(self, total_wait_time: int = 1):
        start_time = time.time()

        bt.logging.success(f"✅ MockGromacsExecutor.run is running ✅")
        check_if_directory_exists(output_directory=self.output_dir)

        store = os.path.join(self.output_dir, self.state_file_name)
        states = ["init", "wait", "finished"]

        itermediate_interval = total_wait_time / len(states)

        for state in states:
            bt.logging.info(f"Running state: {state}")
            state_time = time.time()
            with open(store, "w") as f:
                f.write(f"{state}\n")

            time.sleep(itermediate_interval)
            bt.logging.info(f"Total state_time: {time.time() - state_time}")

        bt.logging.warning(f"Total run method time: {time.time() - start_time}")
